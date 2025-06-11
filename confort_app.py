from shiny import App, render, ui, reactive
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from utils.config import load_settings
from utils.confort import (
    graficar_DDH_por_periodos, 
    plot_heatmap_zona_confort_Morillon,
    plot_utci,
    plot_confort_adaptativo
)

# Cargar datos iniciales
con = duckdb.connect('esolmet.db')
df = con.execute("SELECT * FROM lecturas").fetchdf()
con.close()

df_ancho = df.pivot(index='fecha', columns='variable', values='valor')
df_ancho.index = pd.to_datetime(df_ancho.index)

# Renombrar columnas usando alias desde settings
_, _, _, _, _, alias = load_settings()
df_ancho = df_ancho.rename(columns=alias)

# Función para el gráfico UTCI

#  mostrar la temperatura de confort para el DDH, poner solo 2 para 
# Utilizar el slider range para seleccionar los anios 
# 

# Interfaz de usuario
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.style("""
            .responsive-container, .ddh-container, 
            .heatmap-container, .utci-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }
            .controls-card, .ddh-controls, 
            .heatmap-controls, .utci-controls {
                flex: 1 1 400px;
                min-width: 300px;
            }
            .plot-card, .ddh-plot, 
            .heatmap-plot, .utci-plot {
                flex: 2 1 600px;
                min-height: 500px;
            }
            @media (max-width: 1200px) {
                .responsive-container, .ddh-container, 
                .heatmap-container, .utci-container {
                    flex-direction: column;
                }
            }
        """)
    ),
    # Sección Gráfico de Confort
    ui.div(
        {"class": "responsive-container"},
        ui.card(
            {"class": "controls-card"},
            ui.h4("Configuración del gráfico de confort"),
            ui.input_date_range(
                "fechas_t_confort_plotly", 
                "Rango de fechas", 
                start=df_ancho.index.min().date(), 
                end=df_ancho.index.max().date(),
                width="100%"
            ),
            ui.input_checkbox_group(
                "lineas", 
                "Mostrar líneas:", 
                choices={
                    "Te": "Temperatura del aire promedio diario",
                    "tmp_cmf": "Temperatura de confort ASHRAE 55",                    
                    "ASHRAE_80": "Zona de confort ASHRAE 55 (80%)",
                    "ASHRAE_90": "Zona de confort ASHRAE 55 (90%)",
                    "temp_neutralidad_Morillon": "Temperatura de confort Morillón",
                    "Morillon": "Zona de confort Morillón"
                },
                selected=["Te", "ASHRAE_80"],
                width="100%"
            )

        ),
        ui.card(
            {"class": "plot-card"},
            ui.output_ui("plotly_plot")
        )
    ),
    # Sección DDH
    ui.div(
        {"class": "ddh-container"},
        ui.card(
            {"class": "ddh-controls"},
            ui.h4("Configuración de degree days hour (DDH)"),
            ui.input_date_range(
                "rango_1_ddh", 
                "Rango de fechas 1", 
                start=df_ancho.index.min().date(), 
                end=df_ancho.index.max().date(),
                width="100%"
            ),
            ui.input_date_range(
                "rango_2_ddh", 
                "Rango de fechas 2", 
                start=df_ancho.index.min().date(), 
                end=df_ancho.index.max().date(),
                width="100%"
            ),
            ui.input_radio_buttons(  
                "modelo_ddh",  
                "Selecciona un modelo para la temperatura de confort",  
                {"Morillon": "Morillón", "ASHRAE_55": "ASHRAE 55"},  
            )
        ),
        ui.card(
            {"class": "ddh-plot"},
            ui.output_ui("ddh_plot")
        )
    ),
    # Sección Heatmap
    ui.div(
        {"class": "heatmap-container"},
        ui.card(
            {"class": "heatmap-controls"},
            ui.h4("Configuración de heat map con modelo de Morillón"),
            ui.input_slider(
                "years_heatmap",
                "Selecciona el rango de años:",
                min=min(df_ancho.index.year),
                max=max(df_ancho.index.year),
                value=(max(df_ancho.index.year) - 5, max(df_ancho.index.year)),  # por ejemplo, últimos 5 años
                step=1,
                sep="",
                width="100%"
            ),
            ui.input_radio_buttons(  
                "modo_heatmap",  
                "Selecciona un modo",  
                {"mes": "Mensual","semana": "Semanal"},  
            )     
        ),
        ui.card(
            {"class": "heatmap-plot"},
            ui.output_plot("heatmap_plot")
        )
    ),
    # Sección UTCI
    ui.div(
        {"class": "utci-container"},
        ui.card(
            {"class": "utci-controls"},
            ui.h4("Configuración del Universal Thermal Climate Index (UTCI)"),
            ui.input_date_range(
                "rango_1_utci", 
                "Rango de fechas 1", 
                start=df_ancho.index.min().date(), 
                end=df_ancho.index.max().date(),
                width="100%"
            ),
            ui.input_date_range(
                "rango_2_utci", 
                "Rango de fechas 2", 
                start=df_ancho.index.min().date(), 
                end=df_ancho.index.max().date(),
                width="100%"
            ),        
        ),
        ui.card(
            {"class": "utci-plot"},
            ui.output_plot("utci_plot")
        )
    ),
    title="Visualización de Confort Térmico"
)

def server(input, output, session):
    @reactive.Calc
    def filtered_data():
        start_date = pd.to_datetime(input.fechas_t_confort_plotly()[0])
        end_date = pd.to_datetime(input.fechas_t_confort_plotly()[1])
        return df_ancho.loc[start_date:end_date]
    
    @output
    @render.ui
    def plotly_plot():
        df = filtered_data()
        seleccion = input.lineas()

        # Mapear selección de usuario a las columnas reales
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

        fig = plot_confort_adaptativo(df, columnas)

        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.ui
    def ddh_plot():
        rango1 = (input.rango_1_ddh()[0].strftime('%Y-%m-%d'), input.rango_1_ddh()[1].strftime('%Y-%m-%d'))
        rango2 = (input.rango_2_ddh()[0].strftime('%Y-%m-%d'), input.rango_2_ddh()[1].strftime('%Y-%m-%d'))

        fig = graficar_DDH_por_periodos(
            periodos=[rango1, rango2],
            modelo=input.modelo_ddh()
        )
        
        return ui.HTML(fig.to_html(full_html=False))
    
    @output
    @render.plot
    def heatmap_plot():
        year_range = input.years_heatmap()
        modo = input.modo_heatmap()

        # Crear la lista de años desde el rango
        years = list(range(year_range[0], year_range[1] + 1))

        fig = plot_heatmap_zona_confort_Morillon(
            years=years,
            modo=modo,
            data=df_ancho,
            col_temp='Te'
        )
        
        return fig

    
    @output
    @render.plot
    def utci_plot():
        # Obtener rangos de fechas
        rango1 = (input.rango_1_utci()[0].strftime('%Y-%m-%d'), input.rango_1_utci()[1].strftime('%Y-%m-%d'))
        rango2 = (input.rango_2_utci()[0].strftime('%Y-%m-%d'), input.rango_2_utci()[1].strftime('%Y-%m-%d'))
        
        # Generar gráfico UTCI
        fig = plot_utci(rangos=[rango1, rango2])
        
        return fig

app = App(app_ui, server)