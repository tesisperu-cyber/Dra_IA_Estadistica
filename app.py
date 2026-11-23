import os
import re
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_TITLE = "Dra. Estadística IA"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_URL = os.getenv("APP_URL", "http://localhost:8501")

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")

SYSTEM_PROMPT = """
Eres Dra. Estadística IA, una profesora de estadística cercana, paciente, rigurosa y didáctica.
Tu objetivo es que el estudiante comprenda, no que memorice mecánicamente.

PROTOCOLO PEDAGÓGICO:
1. Enseña progresivamente: explicación breve y luego práctica.
2. Explica la lógica y el significado antes de presentar fórmulas.
3. Usa ejemplos cotidianos y situaciones concretas; ayuda al estudiante a imaginar el problema.
4. Aclara las notaciones y convenciones cuando puedan variar.
5. Resuelve los ejercicios por etapas y explica por qué se realiza cada paso.
6. Anticipa errores frecuentes y explica cómo evitarlos.
7. Distingue la posición de un cuartil/percentil del valor que ocupa esa posición.
8. En varianza y desviación estándar, distingue población y muestra, denominadores y unidades.
9. Resume periódicamente lo aprendido y propone ejercicios autónomos.
10. Mantén un tono empático, motivador y respetuoso; adapta la profundidad al nivel del estudiante.
11. No inventes datos, resultados, fuentes ni citas. Si falta información, dilo.
12. Si un resultado depende de una convención, explica cuál se está usando.

ESTILO:
- Empieza por una respuesta directa y clara.
- Evita presentar demasiadas fórmulas de golpe.
- Usa LaTeX cuando ayude, pero explica los símbolos con palabras.
- Si el estudiante comete un error, corrige con respeto y muestra el razonamiento.
- No afirmes haber ejecutado Python: solo la calculadora de la aplicación ejecuta código real.
"""

# RAG mínimo, modular y ampliable. Son notas didácticas internas, no referencias bibliográficas.
KNOWLEDGE_BASE = [
    {
        "id": "fundamentos",
        "topic": "Conceptos y fundamentos de estadística",
        "text": "La estadística es un conjunto de métodos para recolectar, organizar, resumir, analizar e interpretar datos. La población es el conjunto completo de unidades de interés; una muestra es una parte de esa población. Un parámetro describe a la población y un estadístico describe una muestra. Las variables pueden ser cualitativas o cuantitativas; las cuantitativas pueden ser discretas o continuas."
    },
    {
        "id": "descriptiva",
        "topic": "Estadística descriptiva",
        "text": "La estadística descriptiva organiza y resume los datos observados mediante tablas, gráficos y medidas numéricas. No permite por sí sola generalizar a una población más allá de los datos observados. Antes de analizar, conviene revisar unidades, valores faltantes, errores de digitación y tipo de variable."
    },
    {
        "id": "tendencia_dispersion",
        "topic": "Medidas de tendencia central y dispersión",
        "text": "La media aritmética es la suma de los valores dividida entre el número de observaciones. La mediana es el valor central de los datos ordenados. La moda es el valor o los valores más frecuentes. El rango es máximo menos mínimo. La varianza resume la dispersión cuadrática respecto de la media y la desviación estándar es su raíz cuadrada. Para población se divide entre N; para la varianza muestral insesgada se divide entre n-1."
    },
    {
        "id": "cuantiles",
        "topic": "Cuartiles, deciles y percentiles",
        "text": "Los cuantiles dividen una distribución ordenada en partes. Los cuartiles corresponden a cuatro partes, los deciles a diez y los percentiles a cien. Es importante distinguir la posición calculada del valor del dato o cuantíl. Existen métodos de cálculo diferentes, especialmente cuando la posición no es entera; siempre se debe indicar el método o convención utilizada."
    },
    {
        "id": "ejercicios",
        "topic": "Ejercicios resueltos y procedimientos",
        "text": "Procedimiento general: 1) identifica qué se pregunta; 2) reconoce la variable y sus unidades; 3) ordena o limpia los datos si es necesario; 4) elige la medida y el método; 5) sustituye los valores; 6) calcula; 7) interpreta el resultado en el contexto; 8) comprueba que la respuesta tenga sentido. No confundas una posición con el valor que ocupa esa posición."
    },
]

def retrieve_context(question: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Recuperación léxica simple. Puede sustituirse por Chroma u otro vector store."""
    tokens = set(re.findall(r"[a-záéíóúñ0-9]+", question.lower()))
    scored = []
    for doc in KNOWLEDGE_BASE:
        doc_tokens = set(re.findall(r"[a-záéíóúñ0-9]+", (doc["topic"] + " " + doc["text"]).lower()))
        score = len(tokens & doc_tokens)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

@st.cache_data(ttl=600, show_spinner=False)
def get_free_chat_models() -> List[Dict[str, Any]]:
    """Descubre modelos gratuitos de texto desde el catálogo público de OpenRouter."""
    response = requests.get(OPENROUTER_MODELS_URL, timeout=25)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    free = []
    for model in models:
        pricing = model.get("pricing") or {}
        try:
            prompt_price = float(pricing.get("prompt", "inf"))
            completion_price = float(pricing.get("completion", "inf"))
        except (TypeError, ValueError):
            continue
        architecture = model.get("architecture") or {}
        modalities = architecture.get("modality") or ""
        modality_text = str(modalities).lower()
        # El filtro de precio es estricto. Se priorizan modelos de texto.
        is_text = not modality_text or "text" in modality_text
        if prompt_price == 0 and completion_price == 0 and is_text:
            free.append({
                "id": model.get("id", ""),
                "name": model.get("name") or model.get("id", ""),
                "context_length": model.get("context_length"),
            })
    return sorted([m for m in free if m["id"]], key=lambda m: m["name"].lower())

def chat_with_fallback(api_key: str, messages: List[Dict[str, str]],
                       selected_model: Optional[str], model_ids: List[str]) -> Tuple[str, str, List[str]]:
    """Intenta el modelo elegido y luego los demás modelos gratuitos del catálogo."""
    candidates = []
    if selected_model:
        candidates.append(selected_model)
    candidates.extend([m for m in model_ids if m != selected_model])
    errors = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": APP_TITLE,
    }
    for model_id in candidates:
        body = {"model": model_id, "messages": messages, "temperature": 0.4}
        try:
            resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=body, timeout=90)
            if resp.ok:
                data = resp.json()
                answer = data["choices"][0]["message"].get("content", "")
                if isinstance(answer, list):
                    answer = "\n".join(str(part.get("text", "")) for part in answer if isinstance(part, dict))
                if answer:
                    return str(answer), model_id, errors
                errors.append(f"{model_id}: respuesta vacía")
            else:
                errors.append(f"{model_id}: HTTP {resp.status_code} - {resp.text[:180]}")
        except requests.RequestException as exc:
            errors.append(f"{model_id}: {exc}")
        except (KeyError, IndexError, ValueError) as exc:
            errors.append(f"{model_id}: respuesta no reconocida ({exc})")
    raise RuntimeError("No se pudo obtener respuesta de los modelos gratuitos probados. " +
                       " | ".join(errors[:5]))

def parse_numbers(raw: str) -> List[float]:
    parts = re.split(r"[,;\s]+", raw.strip())
    values = []
    for part in parts:
        if not part:
            continue
        # Se espera punto decimal para evitar ambigüedad entre coma decimal y separador.
        values.append(float(part))
    if not values:
        raise ValueError("Introduce al menos un número.")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Los datos deben ser números finitos.")
    return values

def calculate_descriptive(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    unique, counts = np.unique(arr, return_counts=True)
    max_count = int(counts.max())
    modes = unique[counts == max_count].tolist() if max_count > 1 else []
    return {
        "n": n,
        "media": float(np.mean(arr)),
        "mediana": float(np.median(arr)),
        "moda": modes,
        "minimo": float(np.min(arr)),
        "maximo": float(np.max(arr)),
        "rango": float(np.ptp(arr)),
        "var_poblacional": float(np.var(arr, ddof=0)),
        "desv_poblacional": float(np.std(arr, ddof=0)),
        "var_muestral": float(np.var(arr, ddof=1)) if n > 1 else None,
        "desv_muestral": float(np.std(arr, ddof=1)) if n > 1 else None,
        "q1_q2_q3": np.quantile(arr, [0.25, 0.50, 0.75], method="linear").tolist(),
    }

def format_number(value):
    if value is None:
        return "No disponible (se requieren al menos 2 datos)"
    return f"{value:.6g}"

# ---------------- UI ----------------
st.title("📊 Dra. Estadística IA")
st.caption("Aprende estadística paso a paso. Comprende la lógica, practica y comprueba los cálculos con Python.")

with st.sidebar:
    st.header("Configuración")
    env_key = st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "Clave API de OpenRouter",
        value=env_key or os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="La clave se utiliza en esta sesión. No la compartas ni la subas a GitHub."
    )
    st.caption("Los modelos se consultan dinámicamente desde el catálogo de OpenRouter.")
    if st.button("Actualizar catálogo gratuito"):
        get_free_chat_models.clear()
        st.rerun()

    try:
        models = get_free_chat_models()
        model_ids = [m["id"] for m in models]
        if models:
            labels = {m["id"]: f'{m["name"]} — {m["id"]}' for m in models}
            default_model = model_ids[0]
            selected_model = st.selectbox(
                "Modelo inicial",
                options=model_ids,
                index=0,
                format_func=lambda x: labels.get(x, x)
            )
            st.success(f"{len(models)} modelos gratuitos de texto detectados.")
        else:
            selected_model, model_ids = None, []
            st.warning("No se detectaron modelos gratuitos compatibles en este momento.")
    except Exception as exc:
        selected_model, model_ids = None, []
        st.error(f"No se pudo consultar el catálogo: {exc}")

    st.divider()
    st.markdown("**Áreas iniciales del RAG**")
    for item in KNOWLEDGE_BASE:
        st.caption("• " + item["topic"])

tab_chat, tab_calc, tab_about = st.tabs(["💬 Aula de estadística", "🧮 Calculadora Python", "ℹ️ Acerca de"])

with tab_chat:
    st.subheader("Conversa con tu profesora")
    st.write("Puedes preguntar por conceptos, fórmulas, ejercicios, interpretación de resultados o errores frecuentes.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Por ejemplo: explícame la diferencia entre varianza poblacional y muestral...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        retrieved = retrieve_context(prompt)
        context_text = "\n\n".join(
            f"[{doc['topic']}]\n{doc['text']}" for doc in retrieved
        ) or "No se recuperó una nota específica del RAG. Responde con conocimiento general y declara límites cuando corresponda."
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCONTEXTO RECUPERADO:\n" + context_text}]
        messages.extend(st.session_state.messages[-12:])

        with st.chat_message("assistant"):
            if not api_key:
                answer = "Para conversar con un modelo de IA, introduce tu clave de OpenRouter en el panel lateral. Mientras tanto, puedes utilizar la calculadora Python."
                st.info(answer)
            elif not model_ids:
                answer = "No hay modelos gratuitos de chat disponibles en el catálogo consultado. Actualiza el catálogo o inténtalo más tarde."
                st.warning(answer)
            else:
                with st.spinner("Dra. Estadística IA está preparando la explicación..."):
                    try:
                        answer, used_model, errors = chat_with_fallback(
                            api_key, messages, selected_model, model_ids
                        )
                        st.caption(f"Modelo utilizado: {used_model}")
                        if errors:
                            st.caption(f"Se activó fallback tras {len(errors)} intento(s) fallido(s).")
                        st.markdown(answer)
                    except Exception as exc:
                        answer = f"No fue posible obtener una respuesta. Detalle: {exc}"
                        st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_calc:
    st.subheader("Calculadora de estadística descriptiva")
    st.write("Introduce datos separados por espacios, comas o punto y coma. Utiliza punto para los decimales.")
    raw_data = st.text_area("Datos", "8, 10, 10, 12, 15, 17, 18", height=90)
    if st.button("Calcular medidas", type="primary"):
        try:
            values = parse_numbers(raw_data)
            result = calculate_descriptive(values)
            c1, c2, c3 = st.columns(3)
            c1.metric("Número de datos (n)", result["n"])
            c2.metric("Media", format_number(result["media"]))
            c3.metric("Mediana", format_number(result["mediana"]))
            c1.metric("Mínimo", format_number(result["minimo"]))
            c2.metric("Máximo", format_number(result["maximo"]))
            c3.metric("Rango", format_number(result["rango"]))
            st.markdown("**Dispersión**")
            dispersion = pd.DataFrame({
                "Medida": ["Varianza poblacional", "Desviación estándar poblacional",
                           "Varianza muestral (n−1)", "Desviación estándar muestral"],
                "Resultado": [format_number(result["var_poblacional"]),
                              format_number(result["desv_poblacional"]),
                              format_number(result["var_muestral"]),
                              format_number(result["desv_muestral"])]
            })
            st.dataframe(dispersion, hide_index=True, use_container_width=True)
            st.markdown("**Cuartiles (método lineal de NumPy)**")
            q1, q2, q3 = result["q1_q2_q3"]
            st.write(f"Q1 = {q1:.6g} · Q2 = {q2:.6g} · Q3 = {q3:.6g}")
            if result["moda"]:
                st.write("Moda(s): " + ", ".join(format_number(x) for x in result["moda"]))
            else:
                st.write("Moda: no hay una moda única (ningún valor se repite más que los demás).")
            st.info(
                "Interpretación: la varianza poblacional divide entre N; la varianza muestral "
                "divide entre n−1. Los cuartiles se calcularon con el método lineal de NumPy; "
                "otras convenciones pueden dar resultados distintos."
            )
        except Exception as exc:
            st.error(f"No se pudieron calcular los datos: {exc}")

with tab_about:
    st.subheader("Una profesora que enseña y calcula")
    st.markdown("""
    **Arquitectura de la versión 1.0**
    - System Prompt: identidad, pedagogía y reglas de rigor.
    - RAG modular: notas iniciales recuperadas por coincidencia léxica.
    - OpenRouter: catálogo dinámico y fallback entre modelos gratuitos compatibles.
    - Python: cálculos estadísticos reales en la pestaña Calculadora.

    **Límites actuales**
    - La base RAG es introductoria y aún no contiene documentos académicos externos verificados.
    - El chat no ejecuta código arbitrario; los cálculos reproducibles se realizan en la calculadora.
    - No se conserva el historial al cerrar la sesión.
    """)
    st.caption("Dra. Estadística IA · Prototipo funcional v1.0")
