# Dra. Estadística IA — versión 1.0

Profesora virtual de estadística con enseñanza progresiva, recuperación de conocimientos básicos (RAG inicial), modelos gratuitos de OpenRouter y cálculos estadísticos ejecutados con Python.

## Qué incluye esta versión

- **Enseñanza pedagógica:** teoría breve, comprensión antes de fórmulas, ejemplos cotidianos, resolución por etapas, prevención de errores y retroalimentación.
- **OpenRouter:** consulta dinámica del catálogo de modelos gratuitos y selección de modelos compatibles con chat.
- **Fallback:** si un modelo falla, la aplicación intenta con otros modelos gratuitos disponibles.
- **Motor estadístico Python:** medidas descriptivas para datos numéricos: media, mediana, moda, mínimo, máximo, rango, varianza y desviación estándar poblacional y muestral.
- **RAG inicial modular:** cinco áreas de conocimiento básico:
  1. Conceptos y fundamentos.
  2. Estadística descriptiva.
  3. Tendencia central y dispersión.
  4. Cuartiles, deciles y percentiles.
  5. Ejercicios y procedimientos.
- **Notebook de Google Colab:** versión de demostración para ejecutar la conversación, el RAG básico y los cálculos desde Colab.

## Estructura del repositorio

```text
dra-estadistica-ia/
├── app.py
├── requirements.txt
├── README.md
└── Dra_Estadistica_IA_Colab.ipynb
```

## 1. Ejecutar localmente en Windows

Se recomienda Python 3.10 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abrirá normalmente en `http://localhost:8501`.

## 2. Configurar OpenRouter

1. Crea o utiliza una clave API en [OpenRouter](https://openrouter.ai/).
2. Abre la aplicación y pega la clave en el panel lateral.
3. La clave se mantiene en la sesión de Streamlit; no se escribe en el repositorio.
4. La aplicación consulta el catálogo público de modelos y filtra los modelos de texto con precio de entrada y salida igual a cero.
5. Si un modelo falla, intenta automáticamente otros modelos gratuitos compatibles.

**Importante:** la disponibilidad, los límites y las condiciones de los modelos gratuitos dependen de OpenRouter y pueden cambiar. No se garantiza que todos los modelos gratuitos del catálogo sean modelos de chat. La aplicación filtra los que parecen compatibles con conversación de texto y maneja los errores de disponibilidad.

Para Streamlit Community Cloud, también puedes guardar la clave en `Settings → Secrets` con este formato:

```toml
OPENROUTER_API_KEY = "tu_clave"
```

No publiques la clave API en GitHub, notebooks compartidos ni capturas de pantalla.

## 3. Desplegar en Streamlit Community Cloud

1. Sube estos archivos a un repositorio de GitHub.
2. Entra en [Streamlit Community Cloud](https://share.streamlit.io/).
3. Conecta GitHub y selecciona el repositorio.
4. Indica `app.py` como archivo principal.
5. En la configuración de Secrets, agrega `OPENROUTER_API_KEY`.
6. Despliega la aplicación.

## 4. Usar el notebook en Google Colab

1. Sube `Dra_Estadistica_IA_Colab.ipynb` a Google Drive o abre Colab.
2. En Colab selecciona **Archivo → Subir notebook**.
3. Ejecuta las celdas en orden.
4. Cuando se solicite, introduce la clave de OpenRouter mediante el cuadro seguro de entrada.

El notebook es una versión de trabajo para Colab. No necesita que la clave quede escrita dentro del archivo.

## 5. Alcance y límites de la versión 1.0

- El RAG incluido es una base didáctica inicial y pequeña, no una biblioteca exhaustiva ni una revisión de literatura.
- Los contenidos base están redactados para esta aplicación. Aún no se han cargado documentos externos con metadatos, páginas o citas bibliográficas verificadas.
- La calculadora ejecuta cálculos reales en Python. La respuesta conversacional del modelo, por sí sola, no ejecuta código.
- La selección de cuartiles usa el método lineal de NumPy. Otros métodos pueden producir resultados diferentes; el método debe indicarse al comparar resultados.
- La moda puede ser múltiple. Si todos los valores aparecen una sola vez, la aplicación informa que no hay moda única.
- Esta versión no incorpora todavía autenticación de estudiantes, historial persistente, carga de archivos, gráficos avanzados ni agentes múltiples.

## 6. Criterios de uso responsable

Dra. Estadística IA debe explicar los supuestos, distinguir población de muestra, mostrar unidades y advertir cuando existan convenciones alternativas. El estudiante debe poder revisar los datos, el método y los resultados.

## Licencia

Añade aquí la licencia que decidas utilizar antes de distribuir públicamente el proyecto.
