from shiny import ui, render
from shiny.express import input, output
import duckdb
import pandas as pd
# from utils.confort import plot_t_comfort_plotly
from utils.config import load_settings
from shiny import render, ui
import plotly.graph_objects as go



# Cargar DataFrame global una sola vez

def panel_confort():
    return ui.nav_panel(
        "Confort termico",
        ui.input_date_range(
            "fechas_utci1",
            "Rango 1:",
            start="2023-11-01",
            end="2025-12-31",
            min="2010-01-01",
            max="2025-12-31",
            language="es",
            separator="a",
        ),
        # ui.output_plot("plot_utci",fill=True),
    )
