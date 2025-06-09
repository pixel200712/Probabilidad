import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from scipy import stats

# Cargar archivo Excel
df = pd.read_excel("Calificaciones_Xonacatlan.xlsx")
# Configurar Streamlit
st.set_page_config(layout="wide", page_title="Análisis de Calificaciones")
st.title("📊 Análisis de Calificaciones por Asignatura")

# Filtro de semestre
semestres = df["Semestre"].dropna().unique()
semestre_seleccionado = st.sidebar.selectbox("Selecciona un semestre", sorted(semestres))

# Filtro de carrera dinámico según semestre
carreras_filtradas = df[df["Semestre"] == semestre_seleccionado]["Carrera"].dropna().unique()
carrera_seleccionada = st.sidebar.selectbox("Selecciona una carrera", sorted(carreras_filtradas))

# Filtro de grupo dinámico según semestre y carrera
grupos_filtrados = df[
    (df["Semestre"] == semestre_seleccionado) &
    (df["Carrera"] == carrera_seleccionada)
]["Grupo"].dropna().unique()
grupo_seleccionado = st.sidebar.selectbox("Selecciona un grupo", sorted(grupos_filtrados))

# Filtrar el DataFrame base según todos los filtros
df_filtrado = df[
    (df["Semestre"] == semestre_seleccionado) &
    (df["Carrera"] == carrera_seleccionada) &
    (df["Grupo"] == grupo_seleccionado)
]

# Filtro de asignatura
todas_asignaturas = df_filtrado["Asignatura"].dropna().unique()
asignatura_seleccionada = st.sidebar.selectbox("Selecciona una asignatura", sorted(todas_asignaturas))

# Filtrado final
grupo_df = df_filtrado[df_filtrado["Asignatura"] == asignatura_seleccionada]

# Encabezado descriptivo
st.subheader(f"📘 {carrera_seleccionada} - {asignatura_seleccionada} | Grupo: {grupo_seleccionado} | Semestre: {semestre_seleccionado}")

# Colores para rangos
rango_colores = {
    '5-6': '#ff073a',  # rojo neón
    '6-7': '#ff9f1c',  # naranja neón
    '7-8': '#ffe066',  # amarillo neón
    '8-9': '#3cee54',  # verde neón
    '9-10': '#00FF00'  # verde brillante neón
}
rango_bins = [5, 6, 7, 8, 9, 10.1]
rango_labels = ['5-6', '6-7', '7-8', '8-9', '9-10']

calificaciones_dict = {}

cols = st.columns(2)  # Dividimos en 2 columnas horizontales

for idx, parcial in enumerate(['P1', 'P2']):
    calificaciones = grupo_df[parcial].dropna()
    calificaciones_dict[parcial] = calificaciones

    if calificaciones.empty:
        cols[idx].warning(f"⚠️ Estadísticas de {parcial}: No disponibles")
        continue

    # Cálculos
    media = calificaciones.mean()
    mediana = calificaciones.median()
    moda = stats.mode(calificaciones, nan_policy='omit', keepdims=True)[0][0]
    varianza = calificaciones.var()
    q1 = calificaciones.quantile(0.25)
    q2 = calificaciones.quantile(0.50)
    q3 = calificaciones.quantile(0.75)

    # HTML con estilos para las tablas
    tabla_html = f"""
    <style>
        .tabla-estadisticas {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        .tabla-estadisticas th, .tabla-estadisticas td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .tabla-estadisticas th {{
            background-color: #1f1f1f;
            color: #00ffd5;
        }}
        .tabla-estadisticas td {{
            color: #ffffff;
            background-color: #121212;
        }}
        .tabla-estadisticas tr:hover td {{
            background-color: #222;
        }}
    </style>

    <h4 style='color:#00ffd5;'>🔹 Estadísticas de {parcial}</h4>
    <table class="tabla-estadisticas">
        <thead>
            <tr>
                <th>📌 Medida</th>
                <th>🔢 Valor</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Media</td><td>{media:.2f}</td></tr>
            <tr><td>Mediana</td><td>{mediana:.2f}</td></tr>
            <tr><td>Moda</td><td>{moda:.2f}</td></tr>
            <tr><td>Varianza</td><td>{varianza:.2f}</td></tr>
            <tr><td>Q1 (25%)</td><td>{q1:.2f}</td></tr>
            <tr><td>Q2 (50%)</td><td>{q2:.2f}</td></tr>
            <tr><td>Q3 (75%)</td><td>{q3:.2f}</td></tr>
        </tbody>
    </table>
    """

    # Mostrar tabla en su columna correspondiente
    cols[idx].markdown(tabla_html, unsafe_allow_html=True)


# ----------- Histograma  ------------------
st.markdown("## 📊 Histograma Calificaciones")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#121212')  # fondo oscuro

for idx, parcial in enumerate(['P1', 'P2']):
    calificaciones = calificaciones_dict[parcial]
    if calificaciones.empty:
        axes[idx].set_title(f'{parcial} - Sin datos', color='white')
        axes[idx].axis('off')
        continue

    conteo, _ = np.histogram(calificaciones, bins=rango_bins)
    total = len(calificaciones)
    porcentajes = (conteo / total) * 100

    axes[idx].set_facecolor('#121212')  # fondo oscuro subplot
    barras = axes[idx].bar(rango_labels, conteo, color=[rango_colores[label] for label in rango_labels])

    # Porcentajes encima de cada barra
    for bar, pct in zip(barras, porcentajes):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2, height + 0.3,
                       f'{pct:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')

    axes[idx].set_title(f'Histograma {parcial}', color='white', fontsize=16, fontweight='bold')
    axes[idx].set_xlabel('Rango', color='white', fontsize=12)
    axes[idx].set_ylabel('Frecuencia', color='white', fontsize=12)
    axes[idx].tick_params(colors='white')  # ticks blancos

plt.tight_layout()
st.pyplot(fig)


st.markdown("## 🥧 Gráficas de Pastel Calificaciones")

# ------------------ Gráficas -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#121212')
for ax in axes:
    ax.set_facecolor('#121212')  # fondo oscuro

tablas_html = []  # Para guardar las dos tablas y mostrarlas después

for idx, parcial in enumerate(['P1', 'P2']):
    calificaciones = calificaciones_dict[parcial]
    if calificaciones.empty:
        axes[idx].set_title(f'{parcial} - Sin datos', color='white')
        axes[idx].axis('off')
        tablas_html.append("")  # Para mantener índice
        continue

    ranges = pd.cut(calificaciones, bins=rango_bins, labels=rango_labels, right=False)
    conteo = ranges.value_counts(sort=False)
    valores = conteo.values
    etiquetas = conteo.index.tolist()
    colores = [rango_colores[label] for label in etiquetas]
    total = valores.sum()
    porcentajes = valores / total * 100

    # Gráfica pastel limpia
    wedges = axes[idx].pie(
        valores,
        labels=None,
        autopct=None,
        startangle=90,
        colors=colores,
        wedgeprops={'edgecolor': '#121212', 'linewidth': 2}
    )
    axes[idx].set_title(f'Pastel {parcial}', color='white', fontsize=16, fontweight='bold')

    # Tabla HTML para guardar
    tabla = "<table style='color:white; font-weight:bold;'>"
    tabla += "<tr><th style='text-align:left; padding-right:15px;'>🎨</th><th style='text-align:left; padding-right:15px;'>Rango</th><th style='text-align:right;'>%</th></tr>"
    for c, r, p in zip(colores, etiquetas, porcentajes):
        tabla += f"<tr>" \
                 f"<td><div style='width:20px; height:20px; background:{c}; border-radius:4px; box-shadow: 0 0 4px {c};'></div></td>" \
                 f"<td style='padding-left:10px;'>{r}</td>" \
                 f"<td style='text-align:right;'>{p:.1f}%</td>" \
                 f"</tr>"
    tabla += "</table>"
    tablas_html.append(tabla)

# Mostrar gráficas
plt.tight_layout()
st.pyplot(fig)

# ------------------ Tablas -------------------
col1, col2 = st.columns(2)
col1.markdown("#### P1")
col1.markdown(tablas_html[0], unsafe_allow_html=True)
col2.markdown("#### P2")
col2.markdown(tablas_html[1], unsafe_allow_html=True)


# ----------- Boxplot ------------------
st.markdown("## 📦 Boxplot Calificaciones")

if not grupo_df[['P1', 'P2']].dropna(how='all').empty:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#121212')
    fig.patch.set_facecolor('#121212')

    # Boxplot detalles
    sns.boxplot(
        data=grupo_df[['P1', 'P2']],
        palette=['#ff073a', '#00ff00'],
        width=0.4,
        linewidth=2.5,
        fliersize=0,
        ax=ax
    )

    # Puntos individuales los de color blanco
    sns.stripplot(
        data=grupo_df[['P1', 'P2']],
        jitter=True,
        dodge=True,
        size=6,
        color='white',
        alpha=0.6,
        ax=ax
    )

    # Fondo y ejes
    ax.set_facecolor('#121212')
    ax.tick_params(colors='white', labelsize=12)
    ax.set_ylabel('Calificación', color='white', fontsize=13)
    ax.set_xlabel('', color='white')

    # Bordes blancos 
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    # Grid discreto en eje Y
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.3)
    ax.set_axisbelow(True)

    st.pyplot(fig)

    # leyenda descriptiva
    with st.expander("📌 ¿Qué muestra este boxplot?"):
        st.markdown("""
        - La **línea central** representa la mediana.
        - El **cuerpo de la caja** abarca del primer al tercer cuartil (Q1 a Q3).
        - Las **líneas externas** (bigotes) muestran el rango típico.
        - Los **puntos blancos** son calificaciones individuales.
        """)
else:
    st.info("📉 No hay datos suficientes para mostrar el boxplot.")
