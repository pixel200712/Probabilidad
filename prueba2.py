import pandas as pd          # Manejo y análisis de datos en estructuras tipo tabla (DataFrames)
import matplotlib.pyplot as plt  # Creación de gráficos estáticos (barras, pasteles, boxplots, etc.)
import seaborn as sns        # Visualización estadística avanzada y gráfica más estética sobre matplotlib
import numpy as np           # Cálculos numéricos y operaciones con arreglos, estadística básica
import streamlit as st       # Framework para crear aplicaciones web interactivas fácilmente
from scipy import stats      # Funciones estadísticas avanzadas como moda, pruebas, etc.
from fpdf import FPDF        # Generar documentos PDF desde Python, agregar texto e imágenes
import tempfile              # Crear archivos temporales para guardar imágenes o datos temporales
import os                    # Manejo de sistema de archivos (eliminar archivos temporales, rutas, etc.)
import re                    # Expresiones regulares para manipular y limpiar texto (ej. quitar emojis)
from PIL import Image        # Biblioteca Pillow para abrir y manejar imágenes (dimensiones, formatos)

# Cargar archivo Excel
df = pd.read_excel("C:/xampp/htdocs/phyton/Archivos/Calificaciones 1 y 2 parcial Plantel Xonacatlán.xlsx")
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

# Encabezado personalizado con estilo moderno
st.markdown(f"""
<style>
.encabezado-box {{
    background: linear-gradient(90deg, #1f1f1f, #2c2c2c);
    border-left: 5px solid #00ffd5;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 25px;
}}
.encabezado-box h4 {{
    color: #00ffd5;
    margin: 0;
    font-size: 20px;
}}

</style>

<div class="encabezado-box">
    <h4>🎓 Carrera: <span style='color:white'>{carrera_seleccionada}</span></h4>
    <h4>📘 Asignatura: <span style='color:white'>{asignatura_seleccionada}</span></h4>
    <h4>👥 Grupo: <span style='color:white'>{grupo_seleccionado}</span> | 🗓️ Semestre: <span style='color:white'>{semestre_seleccionado}</span></h4>
</div>
""", unsafe_allow_html=True)

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

    # Cálculos principales
    media = calificaciones.mean()
    mediana = calificaciones.median()
    moda = stats.mode(calificaciones, nan_policy='omit', keepdims=True)[0][0]
    varianza = calificaciones.var()
    q1 = calificaciones.quantile(0.25)
    q2 = calificaciones.quantile(0.50)
    q3 = calificaciones.quantile(0.75)

    # Cálculos adicionales por que yo lo digo jaja xd 
    maximo = calificaciones.max()
    minimo = calificaciones.min()
    rango = maximo - minimo
    total = calificaciones.count()

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
            <tr><td>Total de Alumnos</td><td>{total}</td></tr>
            <tr><td>Media</td><td>{media:.2f}</td></tr>
            <tr><td>Mediana</td><td>{mediana:.2f}</td></tr>
            <tr><td>Moda</td><td>{moda:.2f}</td></tr>
            <tr><td>Varianza</td><td>{varianza:.2f}</td></tr>
            <tr><td>Rango</td><td>{rango:.2f}</td></tr>
            <tr><td>Q1 (25%)</td><td>{q1:.2f}</td></tr>
            <tr><td>Q2 (50%)</td><td>{q2:.2f}</td></tr>
            <tr><td>Q3 (75%)</td><td>{q3:.2f}</td></tr>
        </tbody>
    </table>
    """

    # Mostrar la tabla en su columna correspondiente
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

# Guardar las variables para el PDF
colores_pie1 = [rango_colores[label] for label in pd.cut(calificaciones_dict['P1'], bins=rango_bins, labels=rango_labels, right=False).value_counts(sort=False).index.tolist()]
etiquetas_pie1 = rango_labels
valores1 = pd.cut(calificaciones_dict['P1'], bins=rango_bins, labels=rango_labels, right=False).value_counts(sort=False).values
porcentajes_pie1 = valores1 / valores1.sum() * 100

colores_pie2 = [rango_colores[label] for label in pd.cut(calificaciones_dict['P2'], bins=rango_bins, labels=rango_labels, right=False).value_counts(sort=False).index.tolist()]
etiquetas_pie2 = rango_labels
valores2 = pd.cut(calificaciones_dict['P2'], bins=rango_bins, labels=rango_labels, right=False).value_counts(sort=False).values
porcentajes_pie2 = valores2 / valores2.sum() * 100


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
        
    with st.expander("✨ Desarrollado por el equipo 603"):
            
        st.markdown("""
        <style>
        .footer-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #ccc;
            font-size: 14px;
            margin-top: 30px;
            padding: 10px;
            animation: fadeIn 1s ease-in-out;
        }

        .footer-line {
            border: none;
            height: 1px;
            width: 80%;
            background: linear-gradient(to right, #00ffff, transparent);
            margin-bottom: 15px;
            opacity: 0.5;
        }

        .footer-list {
            list-style: none;
            padding: 0;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .footer-list li {
            position: relative;
            padding-left: 25px;
            color: #00ffff;
            font-weight: 600;
            text-shadow: 0 0 6px #00ffffaa;
            transition: all 0.3s ease;
        }

        .footer-list li::before {
            content: '💠';
            position: absolute;
            left: 0;
            color: #00ffff;
            font-size: 16px;
            animation: pulse 2s infinite;
        }

        .footer-list li:hover {
            color: #ffffff;
            text-shadow: 0 0 10px #ffffff;
            transform: translateX(5px);
        }

        .footer-links a {
            color: #00ffff;
            margin: 0 10px;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .footer-links a:hover {
            color: #ffffff;
            text-shadow: 0 0 8px #ffffff;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0% { opacity: 0.5; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
            100% { opacity: 0.5; transform: scale(1); }
        }
        </style>

        <hr class='footer-line'>
        <div class='footer-container'>
            <ul class='footer-list'>
                <li>Axel Morales</li>
                <li>Itzel Taneli Hernandez Salinas</li>
                <li>Thalia Ramos Garcia</li>
                <li>Brizza Lizeht Gomez Gracia</li>
                <li>Enrique Morales del Rio</li>
            </ul>
            <div style='margin-top:15px;' class='footer-links'>
                🌐 <a href='https://github.com/equipo603' target='_blank'>GitHub</a> |
                💼 <a href='https://linkedin.com/in/equipo603' target='_blank'>LinkedIn</a> |
                🖼️ <a href='https://portafolio603.com' target='_blank'>Portafolio</a>
            </div>
            <div style='margin-top:10px;'>💻 Proyecto 2 Analisis de Calificaciones - 2025</div>
        </div>
        """, unsafe_allow_html=True)

            
else:
    st.info("📉 No hay datos suficientes para mostrar el boxplot.")
# Función para quitar emojis (¡clave para evitar errores!)
def quitar_emojis(texto):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticonos
        u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # transporte y mapas
        u"\U0001F1E0-\U0001F1FF"  # banderas
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', texto)

#--------------Creación del PDF sin errores de emoji----------------
def generar_pdf(calificaciones_dict, carrera, grupo, asignatura, semestre,
                colores_pies=None, etiquetas_pies=None, porcentajes_pies=None):

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    
    def poner_fondo_negro():
        pdf.set_fill_color(0, 0, 0)
        pdf.rect(0, 0, 210, 297, 'F')

    def quitar_emojis(texto):
        return texto.encode('ascii', 'ignore').decode('ascii')

    # Primera página - encabezado y estadísticas
    pdf.add_page()
    poner_fondo_negro()
    pdf.set_text_color(0, 255, 213)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, quitar_emojis("Reporte de Calificaciones"), ln=True, align='C')

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    encabezado = f"Carrera: {carrera} | Asignatura: {asignatura} | Grupo: {grupo} | Semestre: {semestre}"
    pdf.cell(0, 10, quitar_emojis(encabezado), ln=True)

    for parcial, calificaciones in calificaciones_dict.items():
        if calificaciones.empty:
            continue
        pdf.set_text_color(0, 255, 213)
        pdf.set_font("Arial", 'B', 12)
        pdf.ln(8)
        pdf.cell(0, 10, quitar_emojis(f"Estadísticas de {parcial}"), ln=True)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", '', 11)

        moda_resultado = stats.mode(calificaciones, nan_policy='omit', keepdims=True)
        moda_valor = moda_resultado.mode[0] if len(moda_resultado.mode) > 0 else np.nan

        pdf.cell(0, 8, f"Media: {calificaciones.mean():.2f}", ln=True)
        pdf.cell(0, 8, f"Mediana: {calificaciones.median():.2f}", ln=True)
        pdf.cell(0, 8, f"Moda: {moda_valor:.2f}", ln=True)
        pdf.cell(0, 8, f"Varianza: {calificaciones.var():.2f}", ln=True)
        pdf.cell(0, 8, f"Rango: {calificaciones.max() - calificaciones.min():.2f}", ln=True)

    # Guardar gráficas como imágenes temporales
    img_paths = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(temp_file.name, bbox_inches='tight')
        img_paths.append(temp_file.name)

    # Intentar poner las primeras 2 gráficas juntas en una sola página, verticalmente
    if len(img_paths) >= 2:
        pdf.add_page()
        poner_fondo_negro()
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Gráficas 1 y 2", ln=True, align='C')

        max_width = 180  # casi el ancho total con margen
        max_height = 120  # la mitad aprox. de la página menos márgenes

        y_positions = [25, 25 + max_height + 10]  # arriba y abajo con separación de 10mm
        x_position = 15  # margen lateral fijo

        for i in range(2):
            img_path = img_paths[i]
            im = Image.open(img_path)
            width_px, height_px = im.size
            dpi = im.info.get('dpi', (72, 72))[0]

            width_mm = (width_px / dpi) * 25.4
            height_mm = (height_px / dpi) * 25.4

            scale = min(max_width / width_mm, max_height / height_mm, 1)

            final_width = width_mm * scale
            final_height = height_mm * scale

            pdf.image(img_path, x=x_position, y=y_positions[i], w=final_width, h=final_height)

    # Ahora la gráfica 3 (boxplot) en página nueva
    if len(img_paths) >= 3:
        pdf.add_page()
        poner_fondo_negro()
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Gráfica 3 - Boxplot", ln=True, align='C')

        img_path = img_paths[2]
        im = Image.open(img_path)
        width_px, height_px = im.size
        dpi = im.info.get('dpi', (72, 72))[0]

        width_mm = (width_px / dpi) * 25.4
        height_mm = (height_px / dpi) * 25.4

        # Escalar para que quepa casi toda la página con margen
        max_width = 180
        max_height = 250

        scale = min(max_width / width_mm, max_height / height_mm, 1)

        final_width = width_mm * scale
        final_height = height_mm * scale

        pdf.image(img_path, x=15, y=25, w=final_width, h=final_height)
        
        # Luego de las gráficas, pon leyendas si existen
        if colores_pies and etiquetas_pies and porcentajes_pies:
            pdf.set_xy(10, y_positions[1] + max_height + 5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Leyendas:", ln=True)
            pdf.set_font("Arial", '', 11)
            poner_fondo_negro()

            y_leyenda = pdf.get_y()
            ancho_cuadro = 8
            alto_cuadro = 8

            # Leyendas para las 2 gráficas
            for i in range(2):
                colores = list(colores_pies[i])
                etiquetas = list(etiquetas_pies[i])
                porcentajes = list(porcentajes_pies[i])

                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"Datos Grafica {i + 1}", ln=True)
                pdf.set_font("Arial", '', 11)

                for idx, (c, etiqueta, porcentaje) in enumerate(zip(colores, etiquetas, porcentajes)):
                    x = 10
                    y = pdf.get_y() + 2

                    r, g, b = tuple(int(c.strip('#')[j:j+2], 16) for j in (0, 2, 4))
                    pdf.set_fill_color(r, g, b)
                    pdf.rect(x, y, ancho_cuadro, alto_cuadro, 'F')
                    pdf.set_xy(x + ancho_cuadro + 2, y - 2)
                    pdf.set_text_color(255, 255, 255)
                    pdf.cell(60, 10, etiqueta, ln=0)
                    pdf.cell(20, 10, f"{porcentaje:.1f}%", ln=1)

                pdf.ln(5)
    else:
        # Si hay menos de 2 gráficas o no caben juntas, cada una en página individual
        for i, img in enumerate(img_paths):
            pdf.add_page()
            poner_fondo_negro()
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, quitar_emojis(f"Grafica {i + 1}"), ln=True, align='C')
            pdf.image(img, x=10, y=25, w=190)

    # Guardar y limpiar imágenes temporales
    pdf_path = os.path.join(tempfile.gettempdir(), "reporte_calificaciones.pdf")
    pdf.output(pdf_path)

    for img in img_paths:
        try:
            os.remove(img)
        except PermissionError:
            pass

    return pdf_path
# ------------------ Extraer datos pastel para PDF -------------------
colores1, etiquetas1, porcentajes1 = [], [], []
colores2, etiquetas2, porcentajes2 = [], [], []

for idx, parcial in enumerate(['P1', 'P2']):
    calificaciones = calificaciones_dict[parcial]
    if calificaciones.empty:
        continue

    ranges = pd.cut(calificaciones, bins=rango_bins, labels=rango_labels, right=False)
    conteo = ranges.value_counts(sort=False)
    valores = conteo.values
    etiquetas = conteo.index.tolist()
    colores = [rango_colores[label] for label in etiquetas]
    total = valores.sum()
    porcentajes = valores / total * 100

    if parcial == 'P1':
        colores1, etiquetas1, porcentajes1 = colores, etiquetas, porcentajes
    else:
        colores2, etiquetas2, porcentajes2 = colores, etiquetas, porcentajes

# ------------ Botón que se encarga de generar y descargar el PDF -----------------
if st.button("📥 Generar reporte PDF"):
    pdf_file = generar_pdf(
        calificaciones_dict=calificaciones_dict,
        carrera=carrera_seleccionada,
        grupo=grupo_seleccionado,
        asignatura=asignatura_seleccionada,
        semestre=semestre_seleccionado,
        colores_pies=[colores1, colores2],         # ✅ Correcto
        etiquetas_pies=[etiquetas1, etiquetas2],   # ✅ Correcto
        porcentajes_pies=[porcentajes1, porcentajes2]  # ✅ Correcto
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Descargar PDF",
            data=f,
            file_name="Reporte_Calificaciones.pdf",
            mime="application/pdf"
        )
