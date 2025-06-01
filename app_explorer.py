from shiny import App, ui, render
import shinyswatch  
from components.explorador import panel_explorador, panel_estadistica
from components.panels import panel_documentacion, panel_trayectoriasolar, panel_fotovoltaica, panel_eolica, panel_confort
from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib
from utils.confort import *
#import plotly.express as px
from utils.config import load_settings

esolmet = load_esolmet_data()

variables, latitude, longitude, gmt, name = load_settings()

app_ui = ui.page_fluid(
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
        return graficado_Is_matplotlib(esolmet, input.fechas())


app = App(app_ui, server)

