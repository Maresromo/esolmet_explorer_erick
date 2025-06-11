from shiny import ui
from shinywidgets import output_widget
import faicons as fa  


def panel_estadistica():
    return ui.nav_panel(
        "Estadística",
        "Aquí irá tu contenido estadístico"
    )

def panel_trayectoriasolar():
    return ui.nav_panel(
        "SunPath",
        "Inserta aquí la figura de sunpath"
    )

def panel_fotovoltaica():
    return ui.nav_panel(
        "FotoVoltaica",
        "Inserta aquí la Produccion solar"
    )

def panel_confort():
    return ui.nav_panel(
        "Confort térmico",

        ui.input_date_range(
            "fechas_t_confort",
            "Fechas:",
            start="2023-11-01",
            end="2025-12-31",
            min="2010-01-01",
            max="2025-12-31",
            language="es",
            separator="a",
        ),

        # ui.input_checkbox_group(  
        #     "Ec_temp_conf",  
        #     "Ecuación temperatura de confort",  
        #     {  
        #         "Morillon": "Morillón",  
        #         "ASHRAE_55": "ASHRAE_55",  
        #     },  
        # ),  

        # ui.output_text("value"),  # si sigues usándolo

        ui.output_plot("grafico_confort_plotly")  # 🔽 Agregado aquí
    )



def panel_eolica():
    return ui.nav_panel(
        "Eolica",
        "Inserta aquí la Produccion eólica"
    )
def panel_documentacion():
    return ui.nav_panel(
        "Documentación",
        "Inserta aquí la documentación"
    )


def panel_subir_archivo():
    return ui.nav_panel(
        "Paso 1",
        ui.layout_columns(
            ui.card(
                ui.card_header("Archivo"),
                ui.input_file(
                    "archivo",
                    "Selecciona el archivo CSV",
                    button_label="Examinar",
                    placeholder="Sin archivo",
                    accept='.csv'
                ),
                ui.output_ui("upload_status"),
                ui.output_table("table_tests"),
            ),
            ui.card(
                ui.card_header("Gráfico"),
                output_widget("plot_plotly"),
                full_screen=True,
            ),
            col_widths=[3, 9],
        ),
    )


def panel_pruebas_archivo():
    return ui.nav_panel(
        "Paso 2",
        ui.layout_columns(
            ui.card(
                ui.card_header("Inconsistencias de radiación"),
                ui.output_data_frame("df_radiacion"),
            ),
            ui.card(
                ui.card_header("Gráfico de radiación"),
                output_widget("plot_radiacion"),
                full_screen=True,
            ),
            col_widths=[5, 7],
        ),
        # ui.layout_columns(
        #     ui.card(
        #         ui.card_header("Valores faltantes"),
        #         ui.output_plot("plot_missing"),
        #     ),
        #     ui.card(
        #         ui.card_header("Ubicación de NaN"),
        #         ui.output_data_frame("df_nans"),
        #     ),
        #     ui.card(
        #         ui.card_header("Ubicación de NaT"),
        #         ui.output_data_frame("df_nats"),
        #     ),
        #     col_widths=[6, 3, 3],
        # ),
    )


def panel_cargar_datos():
    return ui.nav_panel(
        "Paso 3",
        ui.card(
            ui.card_header("Datos preparados"),
            ui.card_body(
                ui.layout_column_wrap(
                    ui.div(
                        ui.p("Selecciona una acción para proceder."),
                        ui.output_ui("load_status"),
                        ui.output_ui("delete_status"),
                        class_="flex-grow-1"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "btn_load",
                            "Cargar en base de datos",
                            icon=fa.icon_svg("file-export"),
                            class_="btn btn-outline-success w-100 mb-2"
                        ),
                        ui.input_action_button(
                            "btn_delete",
                            "Eliminar base de datos",
                            icon=fa.icon_svg("trash"),
                            class_="btn btn-outline-danger w-100"
                        ),
                        class_="d-flex flex-column align-items-end",
                        style="min-width: 200px;"
                    ),
                    class_="d-flex gap-3 align-items-start"
                )
            )
        )
    )