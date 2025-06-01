# %%
from matplotlib.colors import LinearSegmentedColormap
import calplot
import numpy as np; np.random.seed(sum(map(ord, 'calplot')))
import pandas as pd
from matplotlib.patches import Patch 
import plotly.express as px
from utils.data_processing import load_esolmet_data



# %%

#No voy a usar plotly por el momento hasta no tener idea de los datos
# Agregue una linea nueva

esolmet = load_esolmet_data()
data = esolmet.copy()
# data = data.dropna()
del data["RECORD"]
del data["Rain_mm_Tot"]

# %%
temp_neutralidad, oscilacion_media_anual = temp_neutralidad_oscilacion_media_anual(data)

# %%
zona_confort = plot_heatmap_zona_confort(data, temp_neutralidad=temp_neutralidad, modo="mes", col_temp="AirTC_Avg", amplitud_zona_confort=amplitud_zona_confort(oscilacion_media_anual))
zona_confort

# %%
DDH = DDH_calc(weather_df_1 = data, t_column= "AirTC_Avg", t_neutralidad=temp_neutralidad, modo='dia')
DDH
# %%
# //////////////////////////////////////////////////////////////////////////////
# Esta raro pero creo que esta bien 
DDH.loc['2023-06-11':'2023-06-11']

prom = data.AirTC_Avg.resample('1H').mean()
prom2 = prom.loc['2023-06-11':'2023-06-11']


(prom2-temp_neutralidad).clip(lower=0).plot()
(prom2-temp_neutralidad).clip(lower=0).sum()

(temp_neutralidad-prom2).clip(lower=0).plot()
(temp_neutralidad-prom2).clip(lower=0).sum()

# //////////////////////////////////////////////////////////////////////////////
# %%
periodos_ = [
    ('2023-01-01', '2023-01-01'),
    ('2023-01-03', '2023-01-03'),
    ('2023-06-03', '2023-06-03')
]
graficar_DDH_por_periodos(data, t_column = 'AirTC_Avg', t_neutralidad= temp_neutralidad, periodos=periodos_)
# %%
# Define colormap: blanco (0) ➝ amarillo ➝ rojo (valores altos)
cmap_custom = LinearSegmentedColormap.from_list(
    'white_to_red',
    ['white', 'red']
)

variable ='DDH_heat'
eventos = DDH_calc(weather_df_1 = data,
                    t_column= "AirTC_Avg", 
                    t_neutralidad=temp_neutralidad,
                    modo='dia')

# return 
plt.figure(figsize=(16,8))
calplot.calplot(eventos[variable],
                suptitle=f'Calendario para {variable}',
                cmap = cmap_custom)
