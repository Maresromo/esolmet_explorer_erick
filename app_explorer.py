from shiny import App, ui, render
import shinyswatch  
from components.explorador import panel_explorador, panel_estadistica
from components.panels import panel_documentacion, panel_trayectoriasolar, panel_fotovoltaica, panel_eolica
from components.confort import panel_confort
# from utils.confort import 
from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib
#import plotly.express as px
from utils.config import load_settings
# import duckdb
from shiny import render, ui
import plotly.graph_objects as go


variables, latitude, longitude, gmt, name, alias = load_settings()
# con = duckdb.connect('esolmet.db')

#No voy a usar plotly por el momento hasta no tener idea de los datos
# Agregue una linea nueva

# esolmet = load_esolmet_data() 

app_ui = ui.page_fillable(
    ui.navset_card_tab( 
        ui.nav_panel(
            "ESOLMET",
            ui.navset_card_tab(
                panel_explorador(),
                panel_estadistica(),
                id="esolmet_subtabs"
            )
        ),
        ui.nav_panel(
            'HERRAMIENTAS',
            ui.navset_card_tab(
                panel_trayectoriasolar(),
                panel_fotovoltaica(),
                panel_eolica(),
                panel_confort(),
                id="herramientas"
            )
        ),
    ),
    theme=shinyswatch.theme.spacelab
)


def server(input, output, session):

    @render.plot(alt='Irradiancia')
    def plot_matplotlib():
        return graficado_Is_matplotlib( input.fechas())
    
    # @render.plot(alt='prueba')
    # def plot_utci():
    #     return plot_utci_2( input.fechas_utci1())
    
    # output.grafico_confort = grafico_confort(input, output, session)



app = App(app_ui, server)


# # %%
# import duckdb

# con = duckdb.connect('esolmet.db')
# print(con.execute("SHOW TABLES").fetchall())
# con.close()

# %%
