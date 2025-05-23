import matplotlib.pyplot as plt

def graficado_Is_matplotlib(esolmet, fechas):
    fig, ax = plt.subplots()
    columnas = ["I_glo_Avg","I_dir_Avg"]
    ax.set_xlim(fechas[0], fechas[1])
    for columna in columnas:
        ax.plot(esolmet[columna],label=columna)
    ax.set_ylabel("Irradiancia [W/m2]")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend()
    return fig

def graficado_Todo_matplotlib(esolmet):
    fig, ax = plt.subplots()
    columnas = esolmet.columns
    for columna in columnas:
        ax.plot(esolmet[columna],label=columna)
    # ax.set_ylabel("Irradiancia [W/m2]")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend()
    return fig
