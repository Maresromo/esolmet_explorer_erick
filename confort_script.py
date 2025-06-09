# %%
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
from utils.confort import *
from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib
from components.explorador import panel_estadistica, panel_explorador
from components.panels import (
    panel_confort,
    panel_documentacion,
    panel_eolica,
    panel_fotovoltaica,
    panel_trayectoriasolar
)


# Falta hacer un heatmap en el que se muestren los grados hora de calentamiento y enfriamiento para cada mes y cada hora para poder 
# comparar los metodos entre si 


# %%
con = duckdb.connect('esolmet.db')
df = con.execute("SELECT * FROM lecturas").fetchdf()

df_ancho = df.pivot(index='fecha', columns='variable', values='valor')
t_column="AirTC_Avg"
w_column = 'WS_ms_Avg'

con.close()

# %%
# Fecha de referencia
prueba = get_ASHRAE_55_temperatures(df_ancho, t_column, w_column)

# %%
plot_t_comfort_plotly(df_ancho, {
    t_column: 'Tdb_day_avg',
    'tmp_cmf': 'tcmf_day_avg ASHRAE 55',
    'tmp_cmf_90_low': 'lim inf ASHRAE 55, 90%',
    'tmp_cmf_90_up': 'lim sup ASHRAE 55, 90%',
    'tmp_cmf_80_low': 'lim inf ASHRAE 55, 80%',
    'tmp_cmf_80_up': 'lim sup ASHRAE 55, 80%',
    'temp_neutralidad_Morillon' : 'temp_neutralidad_Morillon',
    'Lim_inf_Morillon' : 'Lim_inf_Morillon',
    'Lim_sup_Morillon' : 'Lim_sup_Morillon',
}, t_column, w_column)
# %%
plot_heatmap_zona_confort_Morillon(df_ancho, temp_neutralidad_Morillon(df_ancho, t_column), periodo=[2024], modo="mes", col_temp=t_column)
# %%
DDH = DDH_calc(weather_df_1 = df_ancho, t_column= "AirTC_Avg", w_column='WS_ms_Avg', start_time = '2023-10', end_time = '2025', modelo= 'ASHRAE_55')

# %%
periodos = [
    ('2023-03-01', '2023-04-30'),
    ('2024-01-03', '2024-10-03'),
    ('2024-11-01', '2025-12-31')
]

prueba = graficar_DDH_por_periodos(weather_df_1=df_ancho, t_column = 'AirTC_Avg', w_column='WS_ms_Avg', periodos=periodos, modelo= 'Morillon')
prueba

# %%
data = df_ancho.copy()
rangos = [
    ("2023-01-01", "2025-06-30"),  # Verano
    ("2024-07-01", "2024-12-30")   # Invierno
]
plot_UTCI(data, rangos)
# %%
