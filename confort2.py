# %%
from matplotlib.colors import LinearSegmentedColormap
import calplot
import numpy as np; np.random.seed(sum(map(ord, 'calplot')))
import pandas as pd
from matplotlib.patches import Patch 
import plotly.express as px
from utils.data_processing import load_esolmet_data
from pythermalcomfort.models import utci



# %%
from shiny import App, ui, render
import shinyswatch  
from components.explorador import panel_explorador, panel_estadistica
from components.panels import panel_documentacion, panel_trayectoriasolar, panel_fotovoltaica, panel_eolica, panel_confort
from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib
from utils.confort import *
#import plotly.express as px
from utils.config import load_settings



# %%

#No voy a usar plotly por el momento hasta no tener idea de los datos
# Agregue una linea nueva

esolmet = load_esolmet_data()
data = esolmet.copy()
# data = data.dropna()
del data["RECORD"]
del data["Rain_mm_Tot"]

# %%
temp_neutralidad, oscilacion_media_anual = temp_neutralidad_oscilacion_media_anual(data, t_column='AirTC_Avg')

# %%
zona_confort = plot_heatmap_zona_confort(data, temp_neutralidad=temp_neutralidad, modo="mes", col_temp="AirTC_Avg", amplitud_zona_confort=amplitud_zona_confort(oscilacion_media_anual))
zona_confort

# %%
DDH = DDH_calc(weather_df_1 = data, t_column= "AirTC_Avg", t_neutralidad=temp_neutralidad, modo='dia')
DDH
# %%
# //////////////////////////////////////////////////////////////////////////////
# Esta raro pero creo que esta bien 
# DDH.loc['2023-06-11':'2023-06-11']

# prom = data.AirTC_Avg.resample('1H').mean()
# prom2 = prom.loc['2023-06-11':'2023-06-11']


# (prom2-temp_neutralidad).clip(lower=0).plot()
# (prom2-temp_neutralidad).clip(lower=0).sum()

# (temp_neutralidad-prom2).clip(lower=0).plot()
# (temp_neutralidad-prom2).clip(lower=0).sum()

# //////////////////////////////////////////////////////////////////////////////

# %%
# %%
# Define colormap: blanco (0) ➝ amarillo ➝ rojo (valores altos)
# seria bueno mostrar un boton para generar el promedio diario o el acumulado de todo 
# el periodo, así como los días que se tienen calculos de ese periodo
# y si se tiene un número de días menor al 100% de ese periodo, mostrar
# un aviso que diga "precación, si el número de días con datos es muy bajo,
# estos calculos podrían no reflejar la realidad"
import pandas as pd
import matplotlib.pyplot as plt
import calplot
from matplotlib.colors import LinearSegmentedColormap

# Crear mapa de color de blanco a rojo
cmap_custom = LinearSegmentedColormap.from_list('white_to_red', ['blue', 'red'])

# Asegúrate de que es una Serie con índice de fechas
serie = eventos['DDH_heat']
serie.index = pd.to_datetime(serie.index)

# Verifica que haya NaNs
print("Valores NaN:", serie.isna().sum())

# Crear figura
plt.figure(figsize=(16, 8))
calplot.calplot(
    serie,
    suptitle='Calendario para DDH_heat',
    cmap=cmap_custom,
    fillcolor='black'  # Esto colorea los NaNs
)
plt.show()


# %%

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
    weather_df[t_column] = weather_df_1[t_column].resample('h').mean().copy()

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
# %%
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
    fig = px.bar(
        df_grafico,
        x='Periodo',
        y='Valor',
        color='Tipo',
        barmode='group',
        labels={'Valor': 'Grados hora acumulados'},
        title='Comparación de DDH Heat y Cold por período',
        color_discrete_map={
            'DDH_heat': 'red',
            'DDH_cold': 'blue'
        }
    )

    return fig


# %%
periodos_ = [
    ('2023-01-01', '2023-06-30'),
    # ('2023-01-03', '2023-01-03'),
    ('2023-07-01', '2023-12-31')
]

graficar_DDH_por_periodos(data, t_column = 'AirTC_Avg', t_neutralidad= temp_neutralidad, periodos=periodos_)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pythermalcomfort.models import utci

esolmet = load_esolmet_data()
data = esolmet.copy()
# data = data.dropna()
del data["RECORD"]
del data["Rain_mm_Tot"]


data = data.resample('h').mean()

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

# ---------- PREPARACIÓN DE DATOS ----------
data["UTCI"] = data.apply(UTCI, axis=1)
data["Categoría"] = data["UTCI"].apply(categorizar_utci)

# ---------- DEFINIR RANGOS DE FECHAS ----------
rangos = [
    ("2023-01-01", "2023-06-30"),  # Verano
    ("2023-07-01", "2023-12-30")   # Invierno
]

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

# %%
data.Categoría.value_counts
# %%
