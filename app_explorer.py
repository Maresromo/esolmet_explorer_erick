from shiny import App, ui, render
import shinyswatch
from components.explorador import panel_explorador, panel_estadistica
from components.panels import (
    panel_documentacion,
    panel_trayectoriasolar,
    panel_fotovoltaica,
    panel_eolica,
    # panel_confort,
)
from components.confort import panel_confort

from utils.confort import (
    graficar_DDH_por_periodos, 
    plot_heatmap_zona_confort_Morillon,
    plot_utci,
    plot_confort_adaptativo
)

# from utils.data_processing import load_esolmet_data
from utils.graficadores import graficado_Is_matplotlib

# import plotly.express as px
# import duckdb

# Cargar configuración solo si es necesario en el futuro
# from utils.config import load_settings
# con = duckdb.connect('esolmet.db')

# No voy a usar plotly por el momento hasta no tener idea de los datos
# Agregue una linea nueva

# esolmet = load_esolmet_data()

app_ui = ui.page_fillable(
    ui.navset_card_tab(
        ui.nav_panel(
            "ESOLMET",
            ui.navset_card_tab(
                panel_explorador(), panel_estadistica(), id="esolmet_subtabs"
            ),
        ),
        ui.nav_panel(
            "HERRAMIENTAS",
            ui.navset_card_tab(
                panel_trayectoriasolar(),
                panel_fotovoltaica(),
                panel_eolica(),
                panel_confort(),
                id="herramientas",
            ),
        ),
    ),
    theme=shinyswatch.theme.spacelab,
)


def server(input, output, session):

    @render.plot(alt="Irradiancia")
    def plot_matplotlib():
        return graficado_Is_matplotlib(input.fechas())

    @output
    @render.ui
    def plotly_plot():
        seleccion = input.lineas()
        columnas = []
        for item in seleccion:
            if item == "ASHRAE_80":
                columnas.extend(["tmp_cmf_80_low", "tmp_cmf_80_up"])
            elif item == "ASHRAE_90":
                columnas.extend(["tmp_cmf_90_low", "tmp_cmf_90_up"])
            elif item == "Morillon":
                columnas.extend(["Lim_inf_Morillon", "Lim_sup_Morillon"])
            else:
                columnas.append(item)

        fig = plot_confort_adaptativo(columnas)
        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.ui
    def ddh_plot():
        r1 = (input.rango_1_ddh()[0].strftime('%Y-%m-%d'), input.rango_1_ddh()[1].strftime('%Y-%m-%d'))
        r2 = (input.rango_2_ddh()[0].strftime('%Y-%m-%d'), input.rango_2_ddh()[1].strftime('%Y-%m-%d'))

        fig = graficar_DDH_por_periodos([r1, r2], modelo=input.modelo_ddh())
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    @output
    @render.plot
    def heatmap_plot():
        year_range = input.years_heatmap()
        years = list(range(year_range[0], year_range[1] + 1))
        return plot_heatmap_zona_confort_Morillon(years)

    @output
    @render.plot
    def utci_plot():
        r1 = (input.rango_1_utci()[0].strftime('%Y-%m-%d'), input.rango_1_utci()[1].strftime('%Y-%m-%d'))
        r2 = (input.rango_2_utci()[0].strftime('%Y-%m-%d'), input.rango_2_utci()[1].strftime('%Y-%m-%d'))
        return plot_utci(rangos=[r1, r2]) 
   
app = App(app_ui, server)
