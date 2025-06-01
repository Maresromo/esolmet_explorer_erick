# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch 
import plotly.express as px

from matplotlib.colors import LinearSegmentedColormap
import calplot
import numpy as np; np.random.seed(sum(map(ord, 'calplot')))
import pandas as pd
from matplotlib.patches import Patch 
import plotly.express as px


def temp_neutralidad_oscilacion_media_anual(data):
    temp_max_diaria = data['AirTC_Avg'].resample('D').max()
    temp_max_media_mensual = temp_max_diaria.resample('M').mean()

    temp_min_diaria = data['AirTC_Avg'].resample('D').min()
    temp_min_media_mensual = temp_min_diaria.resample('M').mean()

    oscilacion_media_anual = float((temp_max_media_mensual - temp_min_media_mensual).mean())

    temp_media_anual = float(data['AirTC_Avg'].resample('Y').mean())
    
    # Es así, o se calcula con un minimo y un máximo?
    temp_neutralidad = 17.6 + 0.31*float(temp_media_anual)

    return temp_neutralidad, oscilacion_media_anual


def amplitud_zona_confort(oscilacion_media_anual):
    if oscilacion_media_anual < 13:
        return 2.5
    elif 13 <= oscilacion_media_anual < 16:
        return 3.0
    elif 16 <= oscilacion_media_anual < 19:
        return 3.5
    elif 19 <= oscilacion_media_anual < 24:
        return 4.0
    elif 24 <= oscilacion_media_anual < 28:
        return 4.5
    elif 28 <= oscilacion_media_anual < 33:
        return 5.0
    elif 33 <= oscilacion_media_anual < 38:
        return 5.5
    elif 38 <= oscilacion_media_anual < 45:
        return 6.0
    elif 45 <= oscilacion_media_anual < 52:
        return 6.5
    else:  # delta_ta >= 52
        return 7.0


def DDH_calc(weather_df_1, t_column, t_neutralidad, modo="dia"):
    """
    Calcula los grados-día de enfriamiento (DDH_cold) y calefacción (DDH_heat)
    acumulados por día o por mes, conservando un índice temporal adecuado.

    Parámetros:
        weather_df (DataFrame): Debe tener una columna de temperatura y un índice tipo DateTimeIndex.
        col_temp (str): Nombre de la columna de temperatura.
        t_cooling (float): Temperatura base para enfriamiento.
        t_heating (float): Temperatura base para calefacción.
        modo (str): 'dia' para agrupación diaria, 'mes' para agrupación mensual.

    Retorna:
        DataFrame: Índice de fechas o meses y columnas DDH_heat y DDH_cold.
    """

    if not isinstance(weather_df_1.index, pd.DatetimeIndex):
        raise ValueError("El índice de weather_df debe ser un DatetimeIndex.")

    # Copia para no alterar el original
    weather_df = pd.DataFrame()
    weather_df[t_column] = weather_df_1[t_column].resample('1H').mean().copy()

    # Calcular grados-día por fila
    weather_df["DDH_heat"] = (weather_df[t_column] - t_neutralidad).clip(lower=0)
    weather_df["DDH_cold"] = (t_neutralidad - weather_df[t_column]).clip(lower=0)

    # Agrupación
    def suma_nan_si_presente(x):
        # Si hay al menos un NaN en el grupo, retorna NaN
        if x.isna().any():
            return np.nan
        else:
            return x.sum()

    if modo == "dia":
        resultado = weather_df.groupby(weather_df.index.date)[["DDH_heat", "DDH_cold"]].agg(suma_nan_si_presente)
        resultado.index = pd.to_datetime(resultado.index)

    elif modo == "mes":
        resultado = weather_df.groupby(weather_df.index.to_period("M"))[["DDH_heat", "DDH_cold"]].agg(suma_nan_si_presente)
        resultado.index = resultado.index.to_timestamp()

    else:
        raise ValueError("El modo debe ser 'dia' o 'mes'.")

    return resultado


def plot_heatmap_zona_confort(data, amplitud_zona_confort, temp_neutralidad, modo="mes", col_temp="AirTC_Avg"):
    meses_dict = {
        '01': 'Ene.', '02': 'Feb.', '03': 'Mar.', '04': 'Abr.',
        '05': 'May.', '06': 'Jun.', '07': 'Jul.', '08': 'Ago.',
        '09': 'Sep.', '10': 'Oct.', '11': 'Nov.', '12': 'Dic.'
    }

    # Verificaciones
    if col_temp not in data.columns:
        raise ValueError(f"La columna '{col_temp}' no existe en el DataFrame.")
    if data.empty:
        raise ValueError("El DataFrame de entrada está vacío.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("El índice del DataFrame debe ser un DateTimeIndex.")

    lim_inferior = temp_neutralidad - 0.5 * amplitud_zona_confort
    lim_superior = temp_neutralidad + 0.5 * amplitud_zona_confort

    # Colormap con color especial para NaN (negro)
    colors = ['blue', 'white', 'red']
    cmap = ListedColormap(colors)
    cmap.set_bad('black')  # <- Color para NaN

    if modo == "mes":
        horas = data.index.floor("H").strftime("%H:%M")
        meses = data.index.strftime("%m")
        matriz = data[col_temp].groupby([meses, horas]).mean().unstack().T

        col_etiquetas = matriz.columns.tolist()
        etiquetas_x = [meses_dict[mes] for mes in col_etiquetas]

    elif modo == "semana":
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

    matriz_array = matriz.values  # para imshow se requiere array de numpy
    im = ax.imshow(matriz_array, aspect='auto', cmap=cmap, norm=norm)

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
    plt.title(f"Zona de confort térmico cada {modo}", fontsize=12, fontweight="bold")

    # === Leyenda personalizada con parches de colores ===
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
        # title='Interpretación\nde colores',
        fontsize=9,
        title_fontsize=10
    )

    plt.tight_layout()
    plt.show()

def graficar_DDH_por_periodos(data, t_column, t_neutralidad, periodos):
    """
    Calcula y grafica los DDH_heat y DDH_cold para periodos dados.

    Parámetros:
    - data: DataFrame con una columna de temperatura.
    - t_column: nombre de la columna de temperatura.
    - t_neutralidad: temperatura base para calcular DDH.
    - periodos: lista de tuplas (inicio, fin) con fechas en formato 'YYYY-MM-DD'.

    Retorna:
    - Figura de Plotly con gráfica de barras.
    """

    # Calcular los DDH con la función existente
    DDH = DDH_calc(weather_df_1=data, t_column=t_column, t_neutralidad=t_neutralidad, modo='dia')

    # Agrupar DDH por períodos definidos
    datos = []
    for inicio, fin in periodos:
        df_periodo = DDH.loc[inicio:fin]
        suma = df_periodo.sum()
        datos.append({'Periodo': f'{inicio} a {fin}', 'Tipo': 'DDH_heat', 'Valor': suma['DDH_heat']})
        datos.append({'Periodo': f'{inicio} a {fin}', 'Tipo': 'DDH_cold', 'Valor': suma['DDH_cold']})

    df_grafico = pd.DataFrame(datos)

    # Crear la gráfica
    fig = px.bar(df_grafico,
                 x='Periodo',
                 y='Valor',
                 color='Tipo',
                 barmode='group',
                 labels={'Valor': 'Grados Día'},
                 title='Comparación de DDH Heat y Cold por período')

    return fig

