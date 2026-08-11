"""
=============================================================================
 DIAGRAMA DE CARACTERÍSTICA TRACTIVA 
 Autor: Pablo | Ingeniería en Sistemas Automotrices
=============================================================================
"""

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA 
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diagrama de Característica Tractiva",
    page_icon=None,
    layout="wide",
)

ROJO_DUCATI = "#C80E1B"
FONDO = "#121214"
PALETA_MODELO = {
    1: ["#E62626", "#F27319", "#F2C119", "#D94D8C", "#BF1A59", "#F2994D",
        "#CC3333", "#E68026"],
    2: ["#3388E6", "#40BFA6", "#7359E6", "#26BFD9", "#598C59", "#4D73BF",
        "#8CBFE6", "#33A68C"],
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {FONDO}; }}
    h1, h2, h3 {{ color: {ROJO_DUCATI} !important; }}
    [data-testid="stCaptionContainer"] * {{ color: #FFFFFF !important; }}
    [data-testid="stWidgetLabel"] p {{ color: #FFFFFF !important; }}
    [data-testid="stWidgetLabel"] * {{ color: #FFFFFF !important; }}
    [data-testid="stMarkdownContainer"] p {{ color: #FFFFFF !important; }}
    [data-testid="stMarkdownContainer"] * {{ color: #FFFFFF !important; }}
    .stApp, .stApp p, .stApp span, .stApp label {{ color: #FFFFFF; }}
    [data-testid="stAlertContainer"] p {{ color: #FFFFFF !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: #FFFFFF !important; }}
    [data-testid="stTooltipIcon"] {{ color: #FFFFFF !important; }}
    .stTabs [data-baseweb="tab"] p {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] {{ background-color: #1B1B1D; }}
    [data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] svg {{ fill: #FFFFFF !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Diagrama de Característica Tractiva")
st.caption("Fuerza de tracción por marcha vs. velocidad — herramienta de análisis de desempeño dinámico")


def encontrar_cruce(v1, f1, v2, f2):
    """
    Encuentra el punto (velocidad, fuerza) donde se cruzan dos curvas
    fuerza-velocidad de marchas consecutivas. Ese cruce es el punto
    óptimo teórico de cambio de marcha: antes de ahí conviene ir en la
    marcha 'inferior' (más fuerza), después conviene la 'superior'.
    Regresa None si las curvas no se traslapan o no se cruzan.
    """
    lo = max(v1.min(), v2.min())
    hi = min(v1.max(), v2.max())
    if lo >= hi:
        return None

    v_comun = np.linspace(lo, hi, 500)
    f1i = np.interp(v_comun, v1, f1)
    f2i = np.interp(v_comun, v2, f2)
    diff = f1i - f2i
    signo = np.sign(diff)
    cambios = np.where(np.diff(signo) != 0)[0]
    if len(cambios) == 0:
        return None

    k = cambios[0]
    v_a, v_b = v_comun[k], v_comun[k + 1]
    d_a, d_b = diff[k], diff[k + 1]
    if d_b == d_a:
        return None
    v_cruce = v_a - d_a * (v_b - v_a) / (d_b - d_a)
    f_cruce = np.interp(v_cruce, v1, f1)
    return v_cruce, f_cruce


DATOS_EJEMPLO = {
    1: dict(nombre="Ducati Monster", Rin=17, ancho=180, perfil=0.55,
            relaciones=[2.466, 1.842, 1.500, 1.286, 1.150, 1.043],
            RD=2.687, eta=0.92,
            curva=pd.DataFrame({
                "RPM": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 9750, 10500, 11000],
                "Par_Nm": [88, 102, 115, 123, 128, 130, 126, 118, 104, 92],
            }),
            masa=190, Cd=0.60, Af=0.42, Crr=0.015),
    2: dict(nombre="Ducati Panigale V4", Rin=17, ancho=200, perfil=0.55,
            relaciones=[2.533, 1.882, 1.500, 1.286, 1.130, 1.043],
            RD=2.688, eta=0.92,
            curva=pd.DataFrame({
                "RPM": [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000],
                "Par_Nm": [95, 112, 124, 132, 138, 142, 138, 126, 108, 88],
            }),
            masa=200, Cd=0.50, Af=0.40, Crr=0.014),
}

# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------
st.sidebar.header("Configuración")

if "config_version" not in st.session_state:
    st.session_state["config_version"] = 0
if "cfg_cargada" not in st.session_state:
    st.session_state["cfg_cargada"] = None

archivo_subido = st.sidebar.file_uploader(
    "Cargar configuración guardada (JSON)", type="json",
    help="Sube un archivo exportado antes con el botón 'Descargar configuración' "
         "para no tener que volver a llenar el formulario.",
)
if archivo_subido is not None and st.session_state.get("_ultimo_archivo") != archivo_subido.name:
    st.session_state["cfg_cargada"] = json.load(archivo_subido)
    st.session_state["config_version"] += 1
    st.session_state["_ultimo_archivo"] = archivo_subido.name
    # Si el archivo ya trae guardado cuántos modelos eran, nos saltamos la
    # pantalla de la pregunta inicial.
    st.session_state["n_modelos_confirmado"] = st.session_state["cfg_cargada"].get("n_modelos", 2)
    st.rerun()

cfg_cargada = st.session_state["cfg_cargada"]
sufijo = f"_v{st.session_state['config_version']}"

if cfg_cargada:
    st.sidebar.success("Configuración cargada")

incluir_resistencia_default = cfg_cargada.get("incluir_resistencia", True) if cfg_cargada else True
incluir_resistencia = st.sidebar.checkbox(
    "Incluir resistencia al avance (rodadura + aero)",
    value=incluir_resistencia_default, key=f"incluir_resistencia{sufijo}",
)

st.sidebar.divider()
st.sidebar.caption(
    "Tip: usa [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer) "
    "(gratis) para extraer con precisión los puntos rpm/par de la gráfica "
    "oficial del fabricante."
)

if st.session_state.get("n_modelos_confirmado") is not None:
    st.sidebar.divider()
    if st.sidebar.button("Cambiar número de modelos", use_container_width=True):
        st.session_state["n_modelos_confirmado"] = None
        st.rerun()

# ---------------------------------------------------------------------------
# PANTALLA INICIAL: PREGUNTA 
# ---------------------------------------------------------------------------
if "n_modelos_confirmado" not in st.session_state:
    st.session_state["n_modelos_confirmado"] = None

if st.session_state["n_modelos_confirmado"] is None:
    st.subheader("¿Cuántos modelos quieres evaluar?")
    n_modelos_elegido = st.radio(
        "Cantidad de modelos", [1, 2], horizontal=True,
        label_visibility="collapsed", key="eleccion_n_modelos",
    )
    if st.button("Continuar", type="primary"):
        st.session_state["n_modelos_confirmado"] = n_modelos_elegido
        st.rerun()
    st.stop()

n_modelos = st.session_state["n_modelos_confirmado"]

# ---------------------------------------------------------------------------
# FORMULARIOS POR MODELO 
# ---------------------------------------------------------------------------
tabs = st.tabs([f"Modelo {i + 1}" for i in range(n_modelos)])
modelos = []

for idx, tab in enumerate(tabs, start=1):
    with tab:
        if cfg_cargada and idx <= len(cfg_cargada.get("modelos", [])):
            ejemplo = cfg_cargada["modelos"][idx - 1]
            curva_default = pd.DataFrame(ejemplo["curva"])
        else:
            ejemplo = DATOS_EJEMPLO[idx]
            curva_default = ejemplo["curva"]

        nombre = st.text_input("Nombre del modelo", value=ejemplo["nombre"], key=f"nombre_{idx}{sufijo}")

        st.subheader("Rueda / neumático")
        c1, c2, c3 = st.columns(3)
        Rin = c1.number_input("Medida del aro (in)", value=float(ejemplo["Rin"]), key=f"rin_{idx}{sufijo}")
        ancho = c2.number_input("Ancho de la llanta (mm)", value=float(ejemplo["ancho"]), key=f"ancho_{idx}{sufijo}")
        perfil = c3.number_input("Perfil (ej. 0.55)", value=float(ejemplo["perfil"]), step=0.01, key=f"perfil_{idx}{sufijo}")

        st.subheader("Transmisión")
        relaciones_valor = (
            ejemplo["relaciones"] if isinstance(ejemplo["relaciones"], str)
            else ", ".join(str(r) for r in ejemplo["relaciones"])
        )
        relaciones_txt = st.text_input(
            "Relaciones de marcha, separadas por coma",
            value=relaciones_valor, key=f"relaciones_{idx}{sufijo}",
        )
        c1, c2 = st.columns(2)
        RD = c1.number_input("Relación del diferencial", value=float(ejemplo["RD"]), key=f"rd_{idx}{sufijo}")
        eta = c2.number_input("Eficiencia de transmisión (0-1)", value=float(ejemplo["eta"]),
                               min_value=0.0, max_value=1.0, step=0.01, key=f"eta_{idx}{sufijo}")

        st.subheader("Curva de motor (rpm vs par)")
        curva = st.data_editor(
            curva_default, num_rows="dynamic", key=f"curva_{idx}{sufijo}",
            column_config={
                "RPM": st.column_config.NumberColumn(min_value=0),
                "Par_Nm": st.column_config.NumberColumn(min_value=0),
            },
        )

        st.subheader("Datos del vehículo")
        c1, c2, c3, c4 = st.columns(4)
        masa = c1.number_input("Masa moto + piloto (kg)", value=float(ejemplo["masa"]), key=f"masa_{idx}{sufijo}")
        Cd = Af = Crr = None
        if incluir_resistencia:
            Cd = c2.number_input("Coef. aerodinámico Cd", value=float(ejemplo.get("Cd") or 0.55), key=f"cd_{idx}{sufijo}")
            Af = c3.number_input("Área frontal (m²)", value=float(ejemplo.get("Af") or 0.42), key=f"af_{idx}{sufijo}")
            Crr = c4.number_input("Coef. de rodadura Crr", value=float(ejemplo.get("Crr") or 0.015), step=0.001,
                                   format="%.3f", key=f"crr_{idx}{sufijo}")

        relaciones = np.array([float(x.strip()) for x in relaciones_txt.split(",") if x.strip()])
        r_metal = (Rin / 2) * 0.0254
        altura_flanco = (ancho / 1000) * perfil
        radio_rueda = r_metal + altura_flanco

        curva_valida = curva.dropna()
        rpm_motor = curva_valida["RPM"].to_numpy(dtype=float)
        par_motor = curva_valida["Par_Nm"].to_numpy(dtype=float)
        orden = np.argsort(rpm_motor)
        rpm_motor, par_motor = rpm_motor[orden], par_motor[orden]

        modelos.append(dict(
            nombre=nombre or f"Modelo {idx}",
            Rin=Rin, ancho=ancho, perfil=perfil, radio_rueda=radio_rueda,
            relaciones=relaciones, relaciones_txt=relaciones_txt,
            RD=RD, eta=eta,
            rpm_motor=rpm_motor, par_motor=par_motor,
            curva_records=curva_valida[["RPM", "Par_Nm"]].to_dict("records"),
            masa=masa, Cd=Cd, Af=Af, Crr=Crr,
        ))

# ---------------------------------------------------------------------------
# GUARDAR CONFIGURACIÓN ACTUAL 
# ---------------------------------------------------------------------------
config_export = {
    "n_modelos": n_modelos,
    "incluir_resistencia": incluir_resistencia,
    "modelos": [
        {
            "nombre": m["nombre"], "Rin": m["Rin"], "ancho": m["ancho"], "perfil": m["perfil"],
            "relaciones": m["relaciones_txt"], "RD": m["RD"], "eta": m["eta"],
            "curva": m["curva_records"], "masa": m["masa"],
            "Cd": m["Cd"], "Af": m["Af"], "Crr": m["Crr"],
        }
        for m in modelos
    ],
}
st.sidebar.divider()
st.sidebar.download_button(
    "Descargar configuración (JSON)",
    data=json.dumps(config_export, indent=2, ensure_ascii=False),
    file_name=f"config_tractiva_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json",
    use_container_width=True,
    help="Guarda todos los datos capturados para volver a cargarlos después sin llenar el formulario de nuevo.",
)

# ---------------------------------------------------------------------------
# BOTÓN
# ---------------------------------------------------------------------------
generar = st.button("Generar diagrama", type="primary", use_container_width=True)

if generar:
    valido = all(len(m["rpm_motor"]) >= 3 and len(m["relaciones"]) >= 1 for m in modelos)
    if not valido:
        st.error("Cada modelo necesita al menos 3 puntos de curva de motor y 1 marcha.")
        st.stop()

    for mod in modelos:
        if len(np.unique(mod["rpm_motor"])) != len(mod["rpm_motor"]):
            st.error(
                f"En '{mod['nombre']}' hay valores de RPM repetidos en la tabla de curva de motor. "
                "Cada fila debe tener un RPM distinto — revisa y corrige antes de generar el diagrama."
            )
            st.stop()

    # -----------------------------------------------------------------
    # CÁLCULO POR MARCHA, POR MODELO
    # -----------------------------------------------------------------
    n_modelos_calc = len(modelos)
    vel_kmh = [None] * n_modelos_calc
    F_N = [None] * n_modelos_calc
    filas = []

    for mi, mod in enumerate(modelos):
        rpm_fino = np.linspace(mod["rpm_motor"].min(), mod["rpm_motor"].max(), 400)
        interpolador = PchipInterpolator(mod["rpm_motor"], mod["par_motor"])
        par_fino = interpolador(rpm_fino)

        p = (mod["radio_rueda"] * 2 * np.pi) * 0.06
        n_marchas = len(mod["relaciones"])
        vel_kmh[mi] = [None] * n_marchas
        F_N[mi] = [None] * n_marchas

        for i, R in enumerate(mod["relaciones"]):
            rpm_rueda = rpm_fino / (R * mod["RD"])
            par_rueda = par_fino * R * mod["RD"] * mod["eta"]
            v = p * rpm_rueda
            F = par_rueda / mod["radio_rueda"]

            vel_kmh[mi][i] = v
            F_N[mi][i] = F

            for rpm_m, rpm_r, par_r, f_val, v_val in zip(rpm_fino, rpm_rueda, par_rueda, F, v):
                filas.append((mi + 1, mod["nombre"], i + 1, rpm_m, rpm_r, par_r, f_val, v_val))

    T = pd.DataFrame(filas, columns=[
        "Modelo", "NombreModelo", "Marcha", "RPM_motor", "RPM_rueda",
        "Par_rueda_Nm", "Fuerza_N", "Velocidad_kmh",
    ])

    # -----------------------------------------------------------------
    # RESISTENCIA AL AVANCE
    # -----------------------------------------------------------------
    v_r_kmh = [None] * n_modelos_calc
    F_res = [None] * n_modelos_calc
    if incluir_resistencia:
        rho_aire, g = 1.225, 9.81
        for mi, mod in enumerate(modelos):
            v_max_kmh = max(v.max() for v in vel_kmh[mi]) * 1.05
            v_kmh_m = np.linspace(0, v_max_kmh, 400)
            v_ms_m = v_kmh_m / 3.6
            F_rod = mod["Crr"] * mod["masa"] * g * np.ones_like(v_ms_m)
            F_aero = 0.5 * rho_aire * mod["Cd"] * mod["Af"] * v_ms_m ** 2
            v_r_kmh[mi] = v_kmh_m
            F_res[mi] = F_rod + F_aero

    # -----------------------------------------------------------------
    # PUNTOS ÓPTIMOS DE CAMBIO DE MARCHA
    # -----------------------------------------------------------------
    puntos_cambio = []
    for mi, mod in enumerate(modelos):
        n_marchas = len(mod["relaciones"])
        for i in range(n_marchas - 1):
            cruce = encontrar_cruce(vel_kmh[mi][i], F_N[mi][i], vel_kmh[mi][i + 1], F_N[mi][i + 1])
            if cruce is not None:
                v_c, f_c = cruce
                puntos_cambio.append((mod["nombre"], i + 1, i + 2, v_c, f_c / 1000))

    # -----------------------------------------------------------------
    # GRÁFICA INTERACTIVA DE TRACCIÓN 
    # -----------------------------------------------------------------
    fig = go.Figure()
    estilo_linea = {1: "solid", 2: "dash"}

    for mi, mod in enumerate(modelos):
        m = mi + 1
        paleta = PALETA_MODELO[m]
        for i in range(len(mod["relaciones"])):
            c = paleta[i % len(paleta)]
            fig.add_trace(go.Scatter(
                x=vel_kmh[mi][i], y=F_N[mi][i] / 1000,
                mode="lines", name=f"{mod['nombre']} - {i + 1}ª marcha",
                line=dict(color=c, width=3, dash=estilo_linea[m]),
                hovertemplate="Vel: %{x:.1f} km/h<br>Fuerza: %{y:.2f} kN<extra>%{fullData.name}</extra>",
            ))
        if incluir_resistencia:
            fig.add_trace(go.Scatter(
                x=v_r_kmh[mi], y=F_res[mi] / 1000,
                mode="lines", name=f"Resistencia - {mod['nombre']}",
                line=dict(color="rgba(220,220,220,0.6)", width=2, dash="dot"),
            ))

    if puntos_cambio:
        fig.add_trace(go.Scatter(
            x=[p[3] for p in puntos_cambio], y=[p[4] for p in puntos_cambio],
            mode="markers", name="Puntos de cambio óptimo",
            marker=dict(color="white", size=10, symbol="diamond", line=dict(color="black", width=1)),
            hovertemplate="Cambio %{text}<br>Vel: %{x:.1f} km/h<br>Fuerza: %{y:.2f} kN<extra></extra>",
            text=[f"{p[1]}ª→{p[2]}ª ({p[0]})" for p in puntos_cambio],
        ))

    if n_modelos_calc == 1:
        titulo = f"DIAGRAMA DE CARACTERÍSTICA TRACTIVA — {modelos[0]['nombre']}"
    else:
        titulo = f"DIAGRAMA DE CARACTERÍSTICA TRACTIVA — {modelos[0]['nombre']} vs {modelos[1]['nombre']}"

    fig.update_layout(
        title=dict(text=titulo, font=dict(color=ROJO_DUCATI, size=22)),
        paper_bgcolor=FONDO, plot_bgcolor=FONDO,
        font=dict(color="#EBEBEB"),
        xaxis=dict(title="Velocidad [km/h]", gridcolor="#3A3A3C", zeroline=False),
        yaxis=dict(title="Fuerza de tracción [kN]", gridcolor="#3A3A3C", zeroline=False),
        legend=dict(bgcolor="#1E1E20", bordercolor="#3A3A3C", borderwidth=1),
        height=650,
        margin=dict(t=80),
    )

    st.plotly_chart(fig, use_container_width=True)

    if puntos_cambio:
        st.subheader("Puntos óptimos de cambio de marcha")
        st.caption("Velocidad donde la marcha siguiente ya entrega más fuerza de tracción que la actual.")
        df_cambios = pd.DataFrame(
            puntos_cambio,
            columns=["Modelo", "De marcha", "A marcha", "Velocidad (km/h)", "Fuerza (kN)"],
        )
        st.dataframe(df_cambios, use_container_width=True)

    # -----------------------------------------------------------------
    # GRÁFICA DE ACELERACIÓN vs VELOCIDAD
    # -----------------------------------------------------------------
    st.subheader("Aceleración vs velocidad")
    if not incluir_resistencia:
        st.caption("Sin resistencia al avance activada: es la aceleración bruta (F_tracción / masa), sin restar arrastre.")

    fig_acc = go.Figure()
    for mi, mod in enumerate(modelos):
        m = mi + 1
        paleta = PALETA_MODELO[m]
        for i in range(len(mod["relaciones"])):
            c = paleta[i % len(paleta)]
            v = vel_kmh[mi][i]
            F = F_N[mi][i]
            if incluir_resistencia:
                F_res_interp = np.interp(v, v_r_kmh[mi], F_res[mi])
            else:
                F_res_interp = 0.0
            a = (F - F_res_interp) / mod["masa"]
            fig_acc.add_trace(go.Scatter(
                x=v, y=a, mode="lines", name=f"{mod['nombre']} - {i + 1}ª marcha",
                line=dict(color=c, width=3, dash=estilo_linea[m]),
                hovertemplate="Vel: %{x:.1f} km/h<br>Aceleración: %{y:.2f} m/s²<extra>%{fullData.name}</extra>",
            ))

    fig_acc.update_layout(
        title=dict(text="ACELERACIÓN vs VELOCIDAD", font=dict(color=ROJO_DUCATI, size=20)),
        paper_bgcolor=FONDO, plot_bgcolor=FONDO,
        font=dict(color="#EBEBEB"),
        xaxis=dict(title="Velocidad [km/h]", gridcolor="#3A3A3C", zeroline=False),
        yaxis=dict(title="Aceleración [m/s²]", gridcolor="#3A3A3C", zeroline=False),
        legend=dict(bgcolor="#1E1E20", bordercolor="#3A3A3C", borderwidth=1),
        height=550,
        margin=dict(t=80),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # -----------------------------------------------------------------
    # TABLA + DESCARGAS
    # -----------------------------------------------------------------
    with st.expander("Ver tabla de resultados"):
        st.dataframe(T, use_container_width=True)

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")

    excel_buffer = io.BytesIO()
    T.to_excel(excel_buffer, index=False)

    try:
        png_bytes = fig.to_image(format="png", scale=3)  # requiere 'kaleido'
        png_bytes_acc = fig_acc.to_image(format="png", scale=3)
    except Exception:
        png_bytes = None
        png_bytes_acc = None
        st.warning(
            "No se pudieron generar las imágenes PNG en este servidor "
            "(problema con la librería de exportación). La tabla en Excel "
            "y las gráficas interactivas siguen funcionando normalmente; "
            "puedes hacer clic derecho sobre una gráfica y 'Guardar imagen "
            "como' desde el navegador si necesitas el PNG."
        )

    c1, c2, c3 = st.columns(3)
    c1.download_button("Descargar tabla (Excel)", data=excel_buffer.getvalue(),
                        file_name=f"Resultados_tractiva_{fecha}.xlsx", use_container_width=True)
    if png_bytes is not None:
        c2.download_button("Descargar tracción (PNG)", data=png_bytes,
                            file_name=f"Curva_traccion_{fecha}.png", use_container_width=True)
    if png_bytes_acc is not None:
        c3.download_button("Descargar aceleración (PNG)", data=png_bytes_acc,
                            file_name=f"Curva_aceleracion_{fecha}.png", use_container_width=True)
else:
    st.info("Llena los datos de cada modelo arriba y presiona **Generar diagrama**.")