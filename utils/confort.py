import calplot
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from pythermalcomfort.models import adaptive_ashrae, utci
from pythermalcomfort.utilities import running_mean_outdoor_temperature
from shiny import App, render, ui
import shinyswatch
from utils.config import load_settings
from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib
from components.explorador import panel_estadistica, panel_explorador


def get_ASHRAE_55_temperatures(df_ancho,t_column,w_column):
        
    df_filtrado = df_ancho[[t_column,w_column]].copy()
    df_filtrado = df_filtrado.resample('d').mean()

    # Inicializamos una lista para guardar los resultados
    t_running_mean_list = []

    # Iteramos sobre el índice (fechas) del DataFrame
    for fecha_referencia in df_filtrado.index:
        # Filtrar los últimos 7 días antes de la fecha actual (excluyendo la fecha actual)
        ventana_7dias = df_filtrado.loc[
            (df_filtrado.index < fecha_referencia) &
            (df_filtrado.index >= fecha_referencia - pd.Timedelta(days=30))
        ]

        # Ordenar y convertir a lista
        temp_array = ventana_7dias.sort_index(ascending=False)[t_column].tolist()

        # Calcular temperatura media móvil si hay suficientes datos, si no, poner NaN
        if len(temp_array) >= 7:
            t_rm = running_mean_outdoor_temperature(temp_array)
        else:
            t_rm = float('nan')

        t_running_mean_list.append(t_rm)

    # Añadir la columna al DataFrame
    df_filtrado['t_running_mean'] = t_running_mean_list

    def parse_adaptive_ashrae_output(output_str):
        lines = output_str.strip().split("\n")
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Convertir a tipo adecuado
                if value.lower() == "false":
                    value = False
                elif value.lower() == "true":
                    value = True
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Mantener como string si no es número ni booleano

                data[key] = value

        return data

    # Aplicar adaptive_ashrae fila por fila
    resultados = df_filtrado.apply(
        lambda row: parse_adaptive_ashrae_output(
            str(adaptive_ashrae(
                tdb=row[t_column],
                tr=row[t_column],
                t_running_mean=row["t_running_mean"],
                v=row[w_column]
            ))
        ),
        axis=1,
        result_type="expand"
    )
    # Unir los resultados al DataFrame original
    df_filtrado = pd.concat([df_filtrado, resultados], axis=1)

    return df_filtrado

def amplitud_zona_confort_Morillon(oscilacion_media_anual):
        delta = oscilacion_media_anual
        if delta < 13:
            amplitud = 2.5
        elif 13 <= delta < 16:
            amplitud = 3.0
        elif 16 <= delta < 19:
            amplitud = 3.5
        elif 19 <= delta < 24:
            amplitud = 4.0
        elif 24 <= delta < 28:
            amplitud = 4.5
        elif 28 <= delta < 33:
            amplitud = 5.0
        elif 33 <= delta < 38:
            amplitud = 5.5
        elif 38 <= delta < 45:
            amplitud = 6.0
        elif 45 <= delta < 52:
            amplitud = 6.5
        else:
            amplitud = 7.0

        return amplitud

def temp_neutralidad_Morillon(data, t_column):

    resultados = {}

    # Asegura que el índice sea datetime y esté ordenado
    data = data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex")

    # Agrupa los datos por año
    for year, group in data.groupby(data.index.year):

        # Máximos diarios y su media mensual
        temp_max_diaria = group[t_column].resample('D').max()
        temp_max_media_mensual = temp_max_diaria.resample('ME').mean()
        t_max = temp_max_media_mensual.max()

        # Mínimos diarios y su media mensual
        temp_min_diaria = group[t_column].resample('D').min()
        temp_min_media_mensual = temp_min_diaria.resample('ME').mean()
        t_min = temp_min_media_mensual.min()

        oscilacion_media_anual = t_max - t_min

        temp_media_anual = group[t_column].mean()
        temp_neutralidad = 17.6 + 0.31 * temp_media_anual

        # Clasificación de la amplitud
        amplitud = amplitud_zona_confort_Morillon(oscilacion_media_anual)

        resultados[year] = {
            "temp_neutralidad": round(temp_neutralidad, 2),
            "amplitud_zona_confort": amplitud,
            # "oscilacion_media_anual": round(oscilacion_media_anual, 2)
        }

    return resultados


def plot_t_comfort_plotly(
    df_ancho,
    columnas,
    tdb_column,
    w_column,
    tmp_cmf_column='tmp_cmf',
    tmp_cmf_80_low_column='tmp_cmf_80_low',
    tmp_cmf_80_up_column='tmp_cmf_80_up',
    title='Temperatura exterior vs confort adaptativo (ASHRAE)'
    ):

    df = get_ASHRAE_55_temperatures(df_ancho, tdb_column, w_column)
    resultados_anuales = temp_neutralidad_Morillon(df_ancho, t_column = tdb_column)

    # Asignar valores a las nuevas columnas por año
    for year, valores in resultados_anuales.items():
        mask = df.index.year == year
        df.loc[mask, 'temp_neutralidad_Morillon'] = valores['temp_neutralidad']
        df.loc[mask, 'Lim_inf_Morillon'] = valores['temp_neutralidad'] + valores['amplitud_zona_confort']/2
        df.loc[mask, 'Lim_sup_Morillon'] = valores['temp_neutralidad'] - valores['amplitud_zona_confort']/2

    fig = go.Figure()

    nombres = {
        tdb_column: 'Tdb_day_avg',
        'tmp_cmf': 'tcmf_day_avg',
        'tmp_cmf_80_low': 'lim inf ASHRAE 55, 80',
        'tmp_cmf_80_up': 'lim sup ASHRAE 55, 80',
        'tmp_cmf_90_low': 'lim inf ASHRAE 55, 90',
        'tmp_cmf_90_up': 'lim sup ASHRAE 55, 90',
        'Lim_inf_Morillon' : 'Lim_inf_Morillon',
        'temp_neutralidad_Morillon' : 'temp_neutralidad_Morillon',
        'Lim_sup_Morillon' : 'Lim_sup_Morillon',
    }

    colores = {
        tdb_column: 'blue',
        'tmp_cmf': 'green',
        'tmp_cmf_80_low': 'red',
        'tmp_cmf_80_up': 'red',
        'tmp_cmf_90_low': 'orange',
        'tmp_cmf_90_up': 'orange',
        'Lim_inf_Morillon' : 'hotpink',
        'temp_neutralidad_Morillon' : 'purple',
        'Lim_sup_Morillon' : 'hotpink',
    }

    dash_style = {
        tdb_column: 'solid',
        'tmp_cmf': 'solid',
        'tmp_cmf_80_low': 'dash',
        'tmp_cmf_80_up': 'dash',
        'tmp_cmf_90_low': 'dash',
        'tmp_cmf_90_up': 'dash',
        'Lim_inf_Morillon' : 'dash',
        'temp_neutralidad_Morillon' : 'solid',
        'Lim_sup_Morillon' : 'dash',
    }

    fig = go.Figure()

    for col_name, display_name in columnas.items():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col_name],
            mode='lines',
            name=display_name,
            line=dict(
                color=colores.get(col_name, 'black'),
                dash=dash_style.get(col_name, 'solid')
            )
        ))

    fig.update_layout(
        title='Temperatura exterior vs confort adaptativo (ASHRAE)',
        xaxis_title='Fecha',
        yaxis_title='Temperatura (°C)',
        legend=dict(
            orientation='v',
            yanchor='bottom',
            y=0.01,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgrey',
            borderwidth=1
        ),
        template='plotly_white',
        height=500
    )

    fig.show()
    # return df


def plot_heatmap_zona_confort_Morillon(data, resultados_anuales, periodo, col_temp, modo="mes"):
    data = data.copy()
        # Asegúrate de que el índice es un DateTimeIndex
    data.index = pd.to_datetime(data.index)
    # Crear un índice horario continuo desde el mínimo al máximo
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='h')
    # Reindexar y llenar los huecos con NaN
    data = data.reindex(full_index)

    # Validación del índice
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("El índice del DataFrame debe ser un DateTimeIndex.")
    if col_temp not in data.columns:
        raise ValueError(f"La columna '{col_temp}' no existe en el DataFrame.")
    if data.empty:
        raise ValueError("El DataFrame de entrada está vacío.")

    # Validación del periodo
    if isinstance(periodo, int):
        periodo = [periodo]
    if not all(isinstance(a, int) for a in periodo):
        raise ValueError("Los valores del parámetro 'periodo' deben ser enteros (años).")
    
        # Filtrar resultados disponibles
    disponibles = set(resultados_anuales.keys())
    seleccion = [a for a in periodo if a in disponibles]
    no_encontrados = [a for a in periodo if a not in disponibles]
    if not seleccion:
        raise ValueError(f"Ninguno de los años especificados en 'periodo' se encuentra en los resultados: {no_encontrados}")

    # Calcular promedios
    amplitudes = [resultados_anuales[a]["amplitud_zona_confort"] for a in seleccion]
    neutralidades = [resultados_anuales[a]["temp_neutralidad"] for a in seleccion]
    amplitud_zona_confort = np.mean(amplitudes)
    temp_neutralidad = np.mean(neutralidades)
        # Límites de confort
    lim_inferior = temp_neutralidad - 0.5 * amplitud_zona_confort
    lim_superior = temp_neutralidad + 0.5 * amplitud_zona_confort


    # Etiquetas de meses
    meses_dict = {
        '01': 'Ene.', '02': 'Feb.', '03': 'Mar.', '04': 'Abr.',
        '05': 'May.', '06': 'Jun.', '07': 'Jul.', '08': 'Ago.',
        '09': 'Sep.', '10': 'Oct.', '11': 'Nov.', '12': 'Dic.'
    }
    # Colormap y matriz de datos
    colors = ['blue', 'white', 'red']
    cmap = ListedColormap(colors)
    cmap.set_bad('black')

    if modo == "mes":
        data = data[data.index.year.isin(seleccion)]
        horas = data.index.floor("H").strftime("%H:%M")
        meses = data.index.strftime("%m")
        matriz = data[col_temp].groupby([meses, horas]).mean().unstack().T
        col_etiquetas = matriz.columns.tolist()
        etiquetas_x = [meses_dict.get(m, m) for m in col_etiquetas]

    elif modo == "semana":
        data = data[data.index.year.isin(seleccion)]
        semanas = data.index.to_series().dt.to_period("W")
        horas = data.index.floor("H").strftime("%H:%M")
        etiquetas = semanas.astype(str)
        matriz = data[col_temp].groupby(by=[etiquetas, horas]).mean().unstack().T
        col_etiquetas = matriz.columns.tolist()

        inicio_semana = semanas.dt.start_time
        meses_semana = inicio_semana.dt.month.astype(str).str.zfill(2)
        semana_a_mes = meses_semana.map(meses_dict)
        semana_a_mes.index = etiquetas
        semana_a_mes = semana_a_mes[~semana_a_mes.index.duplicated()]

        meses_ordenados = [semana_a_mes.get(sem, '') for sem in col_etiquetas]
        etiquetas_x = []
        ultimo = None
        for mes in meses_ordenados:
            if mes != ultimo:
                etiquetas_x.append(mes)
                ultimo = mes
            else:
                etiquetas_x.append("")
    else:
        raise ValueError("El modo debe ser 'mes' o 'semana'")

    if matriz.isna().all().all():
        raise ValueError("La matriz de datos está vacía o llena de NaNs.")

    min_val = matriz.min().min()
    max_val = matriz.max().max()
    bounds = sorted([min_val - 1, lim_inferior, lim_superior, max_val + 1])
    norm = BoundaryNorm(bounds, cmap.N)

    # Gráfico
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(matriz.values, aspect='auto', cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, boundaries=bounds, ticks=[lim_inferior, temp_neutralidad, lim_superior])
    cbar.set_ticklabels([
        f'''< {lim_inferior:.1f}''',
        f'''{temp_neutralidad:.1f}''',
        f'''> {lim_superior:.1f}'''
    ])
    cbar.ax.tick_params(which='both', direction='out', top=True, bottom=True)

    ax.set_xticks(np.arange(len(col_etiquetas)))
    ax.set_xticklabels(etiquetas_x, rotation=0, fontsize=10)

    ax.set_yticks(np.arange(len(matriz.index)))
    ax.set_yticklabels(matriz.index, fontsize=8)

    ax.set_ylabel("Hora del día")
    ax.set_xlabel(f"{modo.capitalize()} del año")
    plt.title(f"Zona de confort térmico ({', '.join(map(str, seleccion))})", fontsize=12, fontweight="bold")

    leyenda_patches = [
        Patch(facecolor='red', edgecolor='black', label='Disconfort\ntérmico\ncálido'),
        Patch(facecolor='white', edgecolor='black', label='Zona de\nconfort'),
        Patch(facecolor='blue', edgecolor='black', label='Disconfort\ntérmico\nfrío'),
        Patch(facecolor='black', edgecolor='black', label='Datos\nfaltantes\n(NaN)'),
    ]

    ax.legend(
        handles=leyenda_patches,
        loc='center left',
        bbox_to_anchor=(1.13, 0.5),
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    plt.tight_layout()
    plt.show()


def DDH_calc(weather_df_1, t_column,w_column, start_time, end_time, modelo):
    """
    Calcula los grados-día de enfriamiento (DDH_cold) y calefacción (DDH_heat)
    acumulados por día, conservando un índice temporal adecuado.

    Parámetros:
        weather_df_1 (DataFrame): Debe tener una columna de temperatura y un índice tipo DateTimeIndex.
        t_column (str): Nombre de la columna de temperatura.
        t_neutralidad (float): Temperatura base de confort.

    Retorna:
        DataFrame: Índice de fechas y columnas DDH_heat y DDH_cold.
    """

    if not isinstance(weather_df_1.index, pd.DatetimeIndex):
        raise ValueError("El índice de weather_df debe ser un DatetimeIndex.")

    weather_df = weather_df_1[[t_column, w_column]].copy()
    weather_df = weather_df.loc[start_time:end_time]
    weather_df = weather_df.resample('h').mean()

    if modelo == 'Morillon': 
        # Copia para no alterar el original
        periodo = [int(start_time.split('-')[0]),int(end_time.split('-')[0])]
        resultados_anuales = temp_neutralidad_Morillon(weather_df_1, t_column)

        # Esto es para calcular los promedios para la temperatura de confort y la amplitud de esta 
        disponibles = set(resultados_anuales.keys())
        seleccion = [a for a in periodo if a in disponibles]
        no_encontrados = [a for a in periodo if a not in disponibles]
        if not seleccion:
            raise ValueError(f"Ninguno de los años especificados en 'periodo' se encuentra en los resultados: {no_encontrados}")


        # amplitud_zona_confort = np.mean([resultados_anuales[a]["amplitud_zona_confort"] for a in seleccion])
        temp_neutralidad = np.mean([resultados_anuales[a]["temp_neutralidad"] for a in seleccion])

        # Calcular grados-día por fila
        weather_df["DDH_heat"] = (weather_df[t_column] - temp_neutralidad).clip(lower=0)
        weather_df["DDH_cold"] = (temp_neutralidad - weather_df[t_column]).clip(lower=0)

    elif modelo == 'ASHRAE_55':
        df = get_ASHRAE_55_temperatures(weather_df, t_column, w_column)

        # Asegurar que los índices sean datetime
        df.index = pd.to_datetime(df.index)
        weather_df.index = pd.to_datetime(weather_df.index)

        # Crear columna con solo la fecha (sin hora) para ambos dataframes
        df['fecha_sin_hora'] = df.index.date
        weather_df['fecha_sin_hora'] = weather_df.index.date

        # Crear diccionario {fecha: tmp_cmf}
        tmp_cmf_dict = df.set_index('fecha_sin_hora')['tmp_cmf'].to_dict()

        # Crear la nueva columna con la diferencia
        weather_df['DDH_heat'] = (weather_df['AirTC_Avg'] - weather_df['fecha_sin_hora'].map(tmp_cmf_dict)).clip(lower=0)
        weather_df['DDH_cold'] = (weather_df['fecha_sin_hora'].map(tmp_cmf_dict) - weather_df['AirTC_Avg']).clip(lower=0)
        weather_df = weather_df.drop(['AirTC_Avg','WS_ms_Avg','fecha_sin_hora'], axis=1)
    
    # Agrupación diaria
    def suma_nan_si_presente(x):
        return np.nan if x.isna().any() else x.sum()

    weather_df = weather_df.groupby(weather_df.index.date)[["DDH_heat", "DDH_cold"]].agg(suma_nan_si_presente)
    weather_df.index = pd.to_datetime(weather_df.index)

    full_index = pd.date_range(start=weather_df.index.min(),
                           end=weather_df.index.max(),
                           freq='D')

    # Reindexar el DataFrame, insertando NaN donde falten datos
    weather_df = weather_df.reindex(full_index)
    
    return weather_df


def graficar_DDH_por_periodos(weather_df_1, t_column, w_column, periodos, modelo):
    """
    Calcula y grafica los DDH_heat y DDH_cold para periodos dados.

    Parámetros:


    Retorna:
    - Figura de Plotly con gráfica de barras.
    """
    periodo_DDH = []
    for inicio, fin in periodos:
        periodo_DDH.append(int(inicio.split('-')[0]))
        periodo_DDH.append(int(fin.split('-')[0]))

    start_time = str(min(periodo_DDH))
    end_time = str(max(periodo_DDH))
    t_column = 'AirTC_Avg'

    DDH = DDH_calc(weather_df_1 = weather_df_1, t_column = t_column, w_column = w_column, start_time = start_time, end_time = end_time, modelo = modelo)

    for i, (inicio, fin) in enumerate(periodos):
        df_rango = DDH.loc[inicio:fin]
        porcentaje_nan = (df_rango.isna().sum() / len(df_rango)) * 100

        print(f"🟠 Precaución, de {inicio} a {fin}:")
        for col, val in porcentaje_nan.items():
            if pd.isna(val):
                val = 100.0  # Asignar 100 si el valor es NaN
            print(f"  → {col} tiene {val:.2f}% de datos faltantes")
        print('Esto podría comprometer los resultados\n')

    datos = []
    for inicio, fin in periodos:
        df_periodo = DDH.loc[inicio:fin]
        suma = df_periodo.sum()
        datos.append({'Periodo': f'{inicio} a {fin}', 'Tipo': 'DDH_heat', 'Valor': suma['DDH_heat']})
        datos.append({'Periodo': f'{inicio} a {fin}', 'Tipo': 'DDH_cold', 'Valor': suma['DDH_cold']})
    for d in datos:
        if d['Valor'] == 0.0:
            print(f"No hay datos para el período {d['Periodo']} ({d['Tipo']})")


    df_grafico = pd.DataFrame(datos)

    # Crear la gráfica
    fig = px.bar(
        df_grafico,
        x='Periodo',
        y='Valor',
        color='Tipo',
        barmode='group',
        labels={'Valor': 'Grados Día'},
        title='Comparación de DDH Heat y Cold por período',
        color_discrete_map={
            'Heat': 'red',
            'Cold': 'blue'
        }
    )

    return fig

# ---------- FUNCIÓN PARA CALCULAR UTCI ----------
def UTCI(row, t_col="AirTC_Avg", ws_col="WS_ms_Avg", rh_col="RH"):
    try:
        return utci(
            tdb=row[t_col],
            tr=row[t_col],
            v=row[ws_col],
            rh=row[rh_col]
        )["utci"]
    except:
        return np.nan

# ---------- CATEGORIZACIÓN UTCI ----------
def categorizar_utci(valor):
    if pd.isna(valor):
        return np.nan
    if valor <= -40:
        return 'Frío extremo'
    elif -40 < valor <= -27:
        return 'Frío fuerte'
    elif -27 < valor <= -13:
        return 'Frío moderado'
    elif -13 < valor <= 9:
        return 'Sin estrés'
    elif 9 < valor <= 26:
        return 'Calor moderado'
    elif 26 < valor <= 32:
        return 'Calor fuerte'
    elif valor > 32:
        return 'Calor extremo'
    else:
        return 'Sin categoría'
    
def plot_UTCI(data, rangos):
    data = data.resample('h').mean()

    # ---------- PREPARACIÓN DE DATOS ----------
    data["UTCI"] = data.apply(UTCI, axis=1)
    data["Categoría"] = data["UTCI"].apply(categorizar_utci)

    # ---------- DEFINIR RANGOS DE FECHAS ----------
    # Colores por rango
    colores = ['#1f77b4', '#ff7f0e']  # Puedes agregar más si hay más rangos

    # ---------- GRAFICAR Y CALCULAR ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    categorias = ['Frío extremo', 'Frío fuerte', 'Frío moderado', 'Sin estrés', 'Calor moderado', 'Calor fuerte', 'Calor extremo']

    # Dataframe acumulador para los conteos
    conteos_por_rango = pd.DataFrame(index=categorias)

    # Recorremos cada rango
    for i, (inicio, fin) in enumerate(rangos):
        df_rango = data.loc[inicio:fin]
        porcentaje_nan = df_rango["UTCI"].isna().mean() * 100

        print(f"🟠 Rango {inicio} a {fin}:")
        print(f"Este rango tiene {porcentaje_nan:.2f}% de datos faltantes, lo cual podría comprometer los resultados.\n")

        conteo = df_rango["Categoría"].value_counts()
        conteo = conteo.reindex(categorias, fill_value=0)
        conteos_por_rango[f'Rango {i+1}'] = conteo

    # ---------- GRAFICAR BARRAS AGRUPADAS ----------
    conteos_por_rango.plot(kind='bar', color=colores, ax=ax)
    ax.set_title("Cantidad de horas por categoría de estrés térmico (UTCI)")
    ax.set_ylabel("Horas")
    ax.set_xlabel("Categoría")
    ax.legend([f"{inicio} a {fin}" for inicio, fin in rangos])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return None



