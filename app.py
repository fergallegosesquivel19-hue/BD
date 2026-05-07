"""
==============================================================================
  HOMOLOGADOR DE BASES DE DATOS — Streamlit App
  Autor: Claude (Anthropic) — Ingeniería de Software & Datos
  Descripción: Herramienta para homologar, agrupar y estandarizar bases de
               datos mediante búsqueda difusa (fuzzy matching) exhaustiva.
==============================================================================
"""

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Homologador de Bases de Datos",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Fuente y fondo general */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Header principal */
  .main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    border-left: 5px solid #00d4aa;
  }
  .main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    font-size: 0.95rem;
    opacity: 0.8;
    margin: 0;
  }

  /* Tarjetas de pasos */
  .step-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #00d4aa;
  }
  .step-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #00b894;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
  }
  .step-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1a202c;
    margin: 0;
  }
  .step-desc {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.3rem;
  }

  /* Badge de modo */
  .mode-badge {
    display: inline-block;
    background: #0f2027;
    color: #00d4aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }

  /* Separador decorativo */
  .divider {
    height: 2px;
    background: linear-gradient(to right, #00d4aa33, transparent);
    border: none;
    margin: 1.5rem 0;
  }

  /* Tabla de resultados */
  .result-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #2c5364;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  /* Botón principal personalizado via Streamlit no es directo,
     pero podemos mejorar el contenedor */
  .stButton > button {
    background: linear-gradient(135deg, #0f2027, #2c5364) !important;
    color: #00d4aa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: 1px solid #00d4aa !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    letter-spacing: 0.5px !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #00d4aa !important;
    color: #0f2027 !important;
  }

  /* Download button */
  .stDownloadButton > button {
    background: #00d4aa !important;
    color: #0f2027 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
  }

  /* Métricas */
  [data-testid="metric-container"] {
    background: #f0fdf9;
    border: 1px solid #b2f5ea;
    border-radius: 10px;
    padding: 0.8rem 1rem;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def leer_archivo(uploaded_file) -> pd.DataFrame | None:
    """Lee un archivo CSV o XLSX y retorna un DataFrame."""
    try:
        nombre = uploaded_file.name.lower()
        if nombre.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        elif nombre.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, dtype=str)
        else:
            st.error(f"Formato no soportado: `{uploaded_file.name}`. Solo se aceptan .csv o .xlsx")
            return None
        df = df.fillna("")
        return df
    except Exception as e:
        st.error(f"❌ Error al leer `{uploaded_file.name}`: {e}")
        return None


def normalizar_texto(texto: str) -> str:
    """Normaliza texto: minúsculas, sin espacios extremos."""
    return str(texto).strip().lower()


def mejor_coincidencia(valor: str, opciones: list[str], umbral: int = 75):
    """
    Busca la mejor coincidencia de `valor` dentro de `opciones`.
    Usa token_set_ratio para manejar palabras en distinto orden.
    Retorna (mejor_match, score) o (None, 0) si no supera el umbral.
    """
    if not valor or not opciones:
        return None, 0

    valor_norm = normalizar_texto(valor)
    opciones_norm = [normalizar_texto(o) for o in opciones]

    resultado = process.extractOne(
        valor_norm,
        opciones_norm,
        scorer=fuzz.token_set_ratio,
        score_cutoff=umbral,
    )
    if resultado:
        match_norm, score, idx = resultado
        return opciones[idx], score
    return None, 0


def convertir_a_excel(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a bytes Excel en memoria."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultado")
    return buffer.getvalue()


def mostrar_paso(numero: str, titulo: str, descripcion: str = ""):
    """Renderiza una tarjeta de paso visual."""
    desc_html = f'<p class="step-desc">{descripcion}</p>' if descripcion else ""
    st.markdown(f"""
    <div class="step-card">
      <div class="step-label">PASO {numero}</div>
      <p class="step-title">{titulo}</p>
      {desc_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔗 Homologador de Bases de Datos</h1>
  <p>Herramienta para homologar, agrupar y estandarizar registros mediante búsqueda difusa exhaustiva</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCCIONES (EXPANDIBLES)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖  Instrucciones de uso — lee esto antes de comenzar", expanded=False):
    st.markdown("""
    ### ¿Qué hace esta herramienta?
    Permite **cruzar, homologar y estandarizar** registros entre dos bases de datos (o dentro de una misma),
    incluso cuando los textos tienen variaciones ortográficas, abreviaciones, mayúsculas diferentes o palabras en distinto orden.

    ---

    ### Modo A — Cruce de bases de datos
    Úsalo cuando tienes **dos archivos** y quieres traer información del primero al segundo.

    | Paso | Acción |
    |------|--------|
    | 1 | Sube el **Libro Base** (el catálogo o referencia correcta) en CSV o XLSX |
    | 2 | Sube uno o más **Libros a Homologar** |
    | 3 | Selecciona la **columna llave** del Libro Base (ej. "Nombre_Oficial") |
    | 4 | Selecciona la **columna objetivo** en los libros a homologar (lo que se va a comparar) |
    | 5 | Elige qué **columnas adicionales** del Libro Base deseas agregar al resultado |
    | 6 | Ajusta el **umbral de similitud** (75–100; más alto = más estricto) |
    | 7 | Ejecuta y descarga el resultado |

    ---

    ### Modo B — Agrupación interna
    Úsalo cuando tienes **un solo archivo** y quieres detectar duplicados o variantes del mismo registro.

    | Paso | Acción |
    |------|--------|
    | 1 | Sube tu base de datos |
    | 2 | Selecciona la columna con los valores a analizar |
    | 3 | Ajusta el umbral de similitud |
    | 4 | Ejecuta y descarga el resultado con grupos asignados |

    ---

    ### Consejos clave
    - 📌 **Umbral recomendado**: 80 para nombres propios; 70 para descripciones largas.
    - 🔤 La herramienta **ignora mayúsculas/minúsculas** y el orden de palabras automáticamente.
    - ⚠️ Si una columna es numérica, se convierte a texto de forma automática.
    - 📊 El resultado siempre muestra los primeros 10 registros para validación antes de descargar.
    """)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — SELECCIÓN DE MODO
# ─────────────────────────────────────────────────────────────────────────────
mostrar_paso("1", "Selecciona el modo de trabajo")

col_modo1, col_modo2 = st.columns([1, 2])
with col_modo1:
    modo = st.radio(
        "Modo de operación:",
        options=["🔄  Modo A — Cruce de bases", "🔍  Modo B — Agrupación interna"],
        index=0,
        help="Modo A: Cruza dos archivos. Modo B: Detecta similitudes dentro de uno solo.",
    )

modo_id = "A" if "Modo A" in modo else "B"
st.markdown(f'<span class="mode-badge">MODO {modo_id} ACTIVO</span>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — CARGA DE ARCHIVOS
# ─────────────────────────────────────────────────────────────────────────────
mostrar_paso("2", "Carga tus archivos", "Formatos aceptados: .csv  |  .xlsx  |  .xls")

df_base = None
dfs_homologar = []      # Lista de (nombre, DataFrame)

if modo_id == "A":
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📚 Libro Base** *(catálogo de referencia)*")
        archivo_base = st.file_uploader(
            "Sube el Libro Base",
            type=["csv", "xlsx", "xls"],
            key="base_uploader",
            label_visibility="collapsed",
        )
        if archivo_base:
            df_base = leer_archivo(archivo_base)
            if df_base is not None:
                st.success(f"✅ Cargado: `{archivo_base.name}` — {len(df_base):,} filas × {len(df_base.columns)} cols")

    with col_b:
        st.markdown("**📂 Libro(s) a Homologar** *(carga múltiple permitida)*")
        archivos_hom = st.file_uploader(
            "Sube los libros a homologar",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="hom_uploader",
            label_visibility="collapsed",
        )
        for f in archivos_hom:
            df_tmp = leer_archivo(f)
            if df_tmp is not None:
                dfs_homologar.append((f.name, df_tmp))
                st.success(f"✅ `{f.name}` — {len(df_tmp):,} filas × {len(df_tmp.columns)} cols")

    archivos_listos = (df_base is not None) and (len(dfs_homologar) > 0)

else:  # Modo B
    archivo_unico = st.file_uploader(
        "Sube tu base de datos",
        type=["csv", "xlsx", "xls"],
        key="unico_uploader",
    )
    if archivo_unico:
        df_base = leer_archivo(archivo_unico)
        if df_base is not None:
            st.success(f"✅ Cargado: `{archivo_unico.name}` — {len(df_base):,} filas × {len(df_base.columns)} cols")

    archivos_listos = (df_base is not None)

if not archivos_listos:
    st.info("⬆️  Sube los archivos requeridos para continuar al siguiente paso.")
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — SELECCIÓN DE COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────
mostrar_paso("3", "Configura las columnas y parámetros")

col_llave_base = None
col_objetivo_hom = None
cols_adicionales = []
col_analisis_b = None

if modo_id == "A":
    cols_base = list(df_base.columns)
    # Unión de todas las columnas de los libros a homologar
    cols_hom_all = []
    for _, df_h in dfs_homologar:
        for c in df_h.columns:
            if c not in cols_hom_all:
                cols_hom_all.append(c)

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        col_llave_base = st.selectbox(
            "🔑 Columna llave del Libro Base",
            options=[""] + cols_base,
            help="La columna que contiene los valores de referencia correctos (ej. nombre oficial, clave, código).",
        )

    with col_s2:
        col_objetivo_hom = st.selectbox(
            "🎯 Columna objetivo en Libro(s) a Homologar",
            options=[""] + cols_hom_all,
            help="La columna cuyos valores se compararán contra el Libro Base.",
        )

    with col_s3:
        cols_adicionales = st.multiselect(
            "➕ Columnas adicionales a traer del Libro Base",
            options=[c for c in cols_base if c != col_llave_base],
            help="Columnas extra del Libro Base que se agregarán al resultado (ej. ID, categoría, etc.).",
        )

else:  # Modo B
    cols_base = list(df_base.columns)
    col_s1, _ = st.columns([1, 2])
    with col_s1:
        col_analisis_b = st.selectbox(
            "🔍 Columna para análisis de similitudes",
            options=[""] + cols_base,
            help="Se buscarán variantes y duplicados en esta columna.",
        )

# Umbral de similitud (compartido)
st.markdown("---")
col_u1, col_u2 = st.columns([1, 2])
with col_u1:
    umbral = st.slider(
        "⚙️  Umbral de similitud (%)",
        min_value=50,
        max_value=100,
        value=80,
        step=1,
        help="Porcentaje mínimo de similitud para considerar una coincidencia. Mayor = más estricto.",
    )
with col_u2:
    st.markdown(f"""
    <div style='padding:0.8rem 1rem; background:#f0fdf9; border-radius:8px; border:1px solid #b2f5ea; margin-top:1.5rem'>
      <span style='font-family:IBM Plex Mono,monospace; font-size:0.8rem; color:#065f46; font-weight:600'>
        UMBRAL ACTIVO: {umbral}%
      </span><br>
      <span style='font-size:0.8rem; color:#64748b'>
        {"🟢 Estricto — solo coincidencias muy cercanas" if umbral >= 90 else
         "🟡 Balanceado — buen equilibrio precisión/recall" if umbral >= 75 else
         "🟠 Permisivo — captura más variantes, puede generar falsos positivos"}
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — VALIDACIONES Y EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────
mostrar_paso("4", "Ejecuta el proceso")

# Validación de columnas seleccionadas
if modo_id == "A":
    if not col_llave_base or col_llave_base == "":
        st.error("❌ Debes seleccionar la **Columna Llave del Libro Base**.")
        st.stop()
    if not col_objetivo_hom or col_objetivo_hom == "":
        st.error("❌ Debes seleccionar la **Columna Objetivo** en los libros a homologar.")
        st.stop()
else:
    if not col_analisis_b or col_analisis_b == "":
        st.error("❌ Debes seleccionar la **Columna de Análisis**.")
        st.stop()

ejecutar = st.button("🚀  Ejecutar Homologación / Agrupación", use_container_width=False)

if not ejecutar:
    st.info("⬆️  Configura los parámetros y presiona **Ejecutar** cuando estés listo.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — PROCESO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
mostrar_paso("5", "Resultados", "Validación visual y descarga del archivo procesado")

resultado_df = None

# ═══════════════════════════════════════════════════════════════════════════
# MODO A — CRUCE DE BASES
# ═══════════════════════════════════════════════════════════════════════════
if modo_id == "A":
    with st.spinner("⏳ Procesando homologación... Esto puede tomar unos segundos."):

        # Verificar y unificar tipos de la columna llave → texto
        if df_base[col_llave_base].dtype != object:
            st.warning(
                f"⚠️ La columna `{col_llave_base}` del Libro Base es numérica. "
                "Se convierte automáticamente a texto para la comparación."
            )
            df_base[col_llave_base] = df_base[col_llave_base].astype(str)

        valores_base = df_base[col_llave_base].fillna("").tolist()
        cols_traer = [col_llave_base] + [c for c in cols_adicionales if c in df_base.columns]

        resultados = []

        for nombre_archivo, df_hom in dfs_homologar:
            # Verificar que la columna objetivo exista en este archivo
            if col_objetivo_hom not in df_hom.columns:
                st.warning(
                    f"⚠️ El archivo `{nombre_archivo}` no tiene la columna `{col_objetivo_hom}`. "
                    "Se omitirá este archivo."
                )
                continue

            # Verificar tipos de la columna objetivo
            if df_hom[col_objetivo_hom].dtype != object:
                st.warning(
                    f"⚠️ La columna `{col_objetivo_hom}` en `{nombre_archivo}` es numérica. "
                    "Se convierte automáticamente a texto."
                )
                df_hom[col_objetivo_hom] = df_hom[col_objetivo_hom].astype(str)

            df_hom[col_objetivo_hom] = df_hom[col_objetivo_hom].fillna("")

            filas_resultado = []
            barra = st.progress(0, text=f"Procesando `{nombre_archivo}`...")

            total = len(df_hom)
            for i, row in df_hom.iterrows():
                valor_buscar = str(row[col_objetivo_hom])
                match, score = mejor_coincidencia(valor_buscar, valores_base, umbral)

                fila = row.to_dict()
                fila["__archivo_origen__"] = nombre_archivo

                if match:
                    fila["HOMOLOGADO_VALOR_BASE"] = match
                    fila["HOMOLOGADO_SCORE"] = score
                    # Traer columnas adicionales del libro base
                    fila_base = df_base[
                        df_base[col_llave_base].apply(normalizar_texto) == normalizar_texto(match)
                    ]
                    if not fila_base.empty:
                        for col_ad in cols_adicionales:
                            if col_ad in fila_base.columns:
                                fila[f"BASE_{col_ad}"] = fila_base.iloc[0][col_ad]
                else:
                    fila["HOMOLOGADO_VALOR_BASE"] = "⚠️ SIN COINCIDENCIA"
                    fila["HOMOLOGADO_SCORE"] = 0
                    for col_ad in cols_adicionales:
                        fila[f"BASE_{col_ad}"] = ""

                filas_resultado.append(fila)
                barra.progress(min(int((i + 1) / total * 100), 100),
                               text=f"Procesando `{nombre_archivo}` — {i+1}/{total}")

            barra.empty()
            if filas_resultado:
                resultados.append(pd.DataFrame(filas_resultado))

        if resultados:
            resultado_df = pd.concat(resultados, ignore_index=True)
        else:
            st.error("❌ No se pudo generar ningún resultado. Revisa los archivos y columnas seleccionadas.")
            st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# MODO B — AGRUPACIÓN INTERNA
# ═══════════════════════════════════════════════════════════════════════════
else:
    with st.spinner("⏳ Analizando similitudes internas..."):

        if df_base[col_analisis_b].dtype != object:
            st.warning(
                f"⚠️ La columna `{col_analisis_b}` es numérica. "
                "Se convierte automáticamente a texto."
            )
            df_base[col_analisis_b] = df_base[col_analisis_b].astype(str)

        df_base[col_analisis_b] = df_base[col_analisis_b].fillna("")
        valores = df_base[col_analisis_b].tolist()

        # Asignar grupos por similitud
        grupo_id = {}     # valor_norm → grupo
        grupos = {}       # grupo → representante
        contador_grupo = 0
        asignaciones = []  # (grupo_num, representante, score)

        barra_b = st.progress(0, text="Agrupando registros similares...")

        for i, valor in enumerate(valores):
            val_norm = normalizar_texto(valor)

            if val_norm in grupo_id:
                # Ya fue asignado
                g = grupo_id[val_norm]
                asignaciones.append((g, grupos[g], 100))
            else:
                # Buscar si hay un grupo existente con alta similitud
                representantes = list(grupos.values())
                match_rep, score = mejor_coincidencia(valor, representantes, umbral) if representantes else (None, 0)

                if match_rep:
                    # Buscar a qué grupo pertenece el representante
                    g_encontrado = None
                    for g_num, rep in grupos.items():
                        if normalizar_texto(rep) == normalizar_texto(match_rep):
                            g_encontrado = g_num
                            break
                    if g_encontrado is not None:
                        grupo_id[val_norm] = g_encontrado
                        asignaciones.append((g_encontrado, grupos[g_encontrado], score))
                    else:
                        # Crear nuevo grupo
                        contador_grupo += 1
                        grupo_id[val_norm] = contador_grupo
                        grupos[contador_grupo] = valor
                        asignaciones.append((contador_grupo, valor, 100))
                else:
                    # Nuevo grupo
                    contador_grupo += 1
                    grupo_id[val_norm] = contador_grupo
                    grupos[contador_grupo] = valor
                    asignaciones.append((contador_grupo, valor, 100))

            barra_b.progress(
                min(int((i + 1) / len(valores) * 100), 100),
                text=f"Agrupando — {i+1}/{len(valores)} registros"
            )

        barra_b.empty()

        df_resultado_b = df_base.copy()
        df_resultado_b["GRUPO_ID"] = [a[0] for a in asignaciones]
        df_resultado_b["GRUPO_REPRESENTANTE"] = [a[1] for a in asignaciones]
        df_resultado_b["SIMILITUD_SCORE"] = [a[2] for a in asignaciones]

        # Ordenar por grupo para facilitar revisión
        df_resultado_b = df_resultado_b.sort_values("GRUPO_ID").reset_index(drop=True)
        resultado_df = df_resultado_b


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if resultado_df is not None and not resultado_df.empty:
    st.success("✅ ¡Proceso completado exitosamente!")

    # Métricas resumen
    if modo_id == "A":
        total_reg = len(resultado_df)
        con_match = len(resultado_df[resultado_df["HOMOLOGADO_SCORE"] > 0])
        sin_match = total_reg - con_match
        score_prom = resultado_df[resultado_df["HOMOLOGADO_SCORE"] > 0]["HOMOLOGADO_SCORE"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Total registros", f"{total_reg:,}")
        m2.metric("✅ Con coincidencia", f"{con_match:,}", f"{con_match/total_reg*100:.1f}%")
        m3.metric("⚠️ Sin coincidencia", f"{sin_match:,}", f"-{sin_match/total_reg*100:.1f}%" if sin_match else "0%")
        m4.metric("🎯 Score promedio", f"{score_prom:.1f}%" if not pd.isna(score_prom) else "—")

    else:
        total_reg = len(resultado_df)
        num_grupos = resultado_df["GRUPO_ID"].nunique()
        singletons = resultado_df.groupby("GRUPO_ID").size()
        duplicados = int((singletons > 1).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("📊 Total registros", f"{total_reg:,}")
        m2.metric("🗂️ Grupos únicos", f"{num_grupos:,}")
        m3.metric("🔁 Grupos con variantes", f"{duplicados:,}")

    # Vista previa
    st.markdown('<p class="result-header">👁  Vista previa — primeros 10 registros</p>', unsafe_allow_html=True)
    st.dataframe(
        resultado_df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    # Botón de descarga
    excel_bytes = convertir_a_excel(resultado_df)
    nombre_descarga = (
        "homologacion_resultado.xlsx"
        if modo_id == "A"
        else "agrupacion_resultado.xlsx"
    )

    st.markdown("---")
    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            label="⬇️  Descargar resultado en Excel",
            data=excel_bytes,
            file_name=nombre_descarga,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_dl2:
        st.markdown(
            f"<span style='font-size:0.82rem; color:#64748b; line-height:2.5'>"
            f"📄 `{nombre_descarga}` — {len(resultado_df):,} registros — todas las columnas incluidas"
            f"</span>",
            unsafe_allow_html=True,
        )

else:
    st.warning("⚠️ El resultado está vacío. Revisa la configuración e intenta nuevamente.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:0.75rem; color:#94a3b8; font-family:IBM Plex Mono,monospace'>"
    "Homologador de Bases de Datos · Desarrollado con Streamlit + RapidFuzz + Pandas"
    "</p>",
    unsafe_allow_html=True,
)
