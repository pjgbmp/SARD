import streamlit as st
import feedparser
from urllib.parse import quote

from scraper import (
    fetch_article_content,
    fetch_twitter_posts,
)

from sentiment import (
    hybrid_sentiment,
    analyze_items,
    aggregate_by_source,
    aggregate_global
)

# ---------------- CONFIGURACIÓN ----------------

st.set_page_config(
    page_title="Sentimiento Económico RD",
    layout="wide"
)

st.title("Análisis de Sentimiento – República Dominicana")
st.subheader("Medición de opinión pública y mediática, Prototipo.")

st.markdown("""
Esta aplicación analiza noticias y opinión pública utilizando un modelo híbrido  
(**VADER + RoBERTa modelos multilingües**).
""")


# ---------------- SIDEBAR ----------------

st.sidebar.header("Fuentes de información")

use_news = st.sidebar.checkbox("📰 Noticias", value=True)
use_twitter = st.sidebar.checkbox("🐦 Twitter / X", value=False)
# use_reddit = st.sidebar.checkbox("Reddit", value=False)  # futuro

st.sidebar.markdown("---")
st.sidebar.info(
    "Puedes activar o desactivar las fuentes para comparar "
    "opinión mediática vs opinión pública. "
    "El modelo esta ambientado a noticias principalmente de República Dominicana, en su construcción y ajuste."
)

st.sidebar.caption("""
           El modulo de información de opinión pública usa X, el cual suele bloquear la lib utilizada, proximos pasos utilizar lib oficial con token.
              """)

st.sidebar.markdown("---")
st.sidebar.subheader("¿Cómo funciona el análisis de sentimiento?")

with st.sidebar.expander("¿Qué mide esta aplicación?"):
    st.markdown("""
    Esta aplicación estima el sentimiento del discurso público sobre temas económicos
    en la República Dominicana a partir de textos provenientes de:

    - Titulares de noticias
    - Contenido de artículos
    - (En desarrollo) Publicaciones en redes sociales

    El objetivo no es predecir hechos, sino medir el tono emocional dominante
    (positivo, neutral o negativo) con el que se habla de un tema.
    """)

with st.sidebar.expander("Fuentes de información"):
    st.markdown("""
    La app puede analizar distintas fuentes, cada una con características propias:

    **Noticias**
    - Lenguaje más formal
    - Mayor contexto
    - Menor carga emocional explícita

    **Twitter / X (En desarrollo)**
    - Lenguaje corto y emocional
    - Alta volatilidad
    - Puede estar sujeto a bloqueos externos

    Reddit podría agregarse en el futuro de considerarse relevante en el contexto dominicano.

    Cada fuente se analiza **por separado** antes de agregarse a un resultado global.
    """)

with st.sidebar.expander("Modelos de análisis de sentimiento utilizados"):
    st.markdown("""
    La aplicación utiliza **dos enfoques complementarios** de análisis de sentimiento:

    1. **VADER** (reglas lingüísticas)
    2. **RoBERTa** (modelo de lenguaje basado en Deep Learning)

    Ambos modelos analizan el mismo texto, pero desde perspectivas diferentes.
    """)

with st.sidebar.expander("VADER – Análisis basado en reglas (explicable)"):
    st.markdown("""
    **VADER (Valence Aware Dictionary and sEntiment Reasoner)** es un modelo basado en reglas
    y diccionarios lingüísticos.

    ### ¿Cómo funciona?
    - Usa un léxico predefinido de palabras con carga emocional
    - Detecta:
        - Negaciones (*no, nunca*)
        - Intensificadores (*muy, extremadamente*)
        - Signos de exclamación
        - Uso de mayúsculas
    - Calcula una puntuación llamada compound, que va de -1 a +1

    ### ¿Por qué se usa?
    - Es rápido
    - Es altamente interpretable
    - Funciona muy bien con titulares

    ### Limitaciones
    - No entiende contexto largo
    - Puede fallar en frases ambiguas o irónicas
    - No capta bien matices económicos complejos
    """)

with st.sidebar.expander("RoBERTa – Modelo de lenguaje contextual"):
    st.markdown("""
    **RoBERTa** es un modelo de Deep Learning basado en la arquitectura Transformer,
    entrenado con millones de textos reales de redes sociales (entrenado con más de 160 GB de texto).

    ### ¿Cómo funciona?
    - Analiza el texto completo como un todo
    - Cada palabra se interpreta en función del contexto
    - Usa mecanismos de atención para entender relaciones entre palabras
    - Devuelve una clasificación:
        - Positivo
        - Neutral
        - Negativo
      junto con una probabilidad (confianza)

    ### ¿Por qué se usa?
    - Entiende frases complejas
    - Maneja bien ambigüedades
    - Es más robusto en textos largos y reales

    ### Limitaciones
    - Es más lento
    - No es fácilmente explicable palabra por palabra
    - Tiene un límite de longitud de texto
    """)

with st.sidebar.expander("Enfoque híbrido: ¿por qué combinar VADER y RoBERTa?"):
    st.markdown("""
    Ningún modelo es perfecto por sí solo. Por eso se utiliza un enfoque mixto:

    - **VADER** aporta rapidez, sencillez y explicabilidad
    - **RoBERTa** aporta comprensión semántica profunda

    ### Estrategia usada:
    - Si ambos modelos coinciden → alta confianza
    - Si difieren → se prioriza RoBERTa
    - La confianza final combina:
        - Intensidad emocional (VADER)
        - Probabilidad del modelo (RoBERTa)

    Esto reduce falsos positivos y mejora la estabilidad del análisis.
    """)

with st.sidebar.expander("Interpretación de resultados"):
    st.markdown("""
    ### Etiquetas de sentimiento
    - **Positivo**: tono optimista o favorable
    - **Neutral**: informativo, balanceado o mixto
    - **Negativo**: preocupación, crítica o pesimismo

    ### Sentimiento promedio
    El valor agregado va de:
    - **+1** → muy positivo
    - **0** → neutral
    - **-1** → muy negativo

    Este valor **no representa hechos económicos**, sino la percepción y el discurso.
    """)

with st.sidebar.expander("⚠️ Limitaciones del análisis"):
    st.markdown("""
    - El sentimiento no equivale a impacto real
    - Las fuentes pueden tener sesgos
    - Ironía y sarcasmo no siempre se detectan
    - Twitter/X puede no estar disponible
    - El idioma y el contexto cultural influyen

    Este análisis debe interpretarse como una **señal complementaria**, no como indicador absoluto.
    """)

st.sidebar.caption("""
           Patricio Guzmán, Técnico en Política Fiscal, MHE.
              """)

# ---------------- INPUTS ----------------

st.subheader("Temas a analizar")

queries_text = st.text_area(
    "Ingresa uno o varios temas (uno por línea)",
    value=(
        "reforma fiscal OR impuestos) AND República Dominicana\n"
        "economía dominicana\n"
        "inflación RD"
    ),
    height=120
)

#Nota con ejemplos bien construidos
st.caption("""
**Ejemplos de queries bien construidos:**
Usar las conjeturas AND, OR, site: y comillas para frases exactas.
           
- `(reforma fiscal OR impuestos) AND "República Dominicana"`  
- `inflación RD site:listindiario.com`  
- `economía dominicana site:diariolibre.com`
""")

with st.expander("Filtro temporal (Google News)"):
    st.markdown("""
    Puedes limitar el rango temporal usando operadores:
    
    - `when:7d` → últimos 7 días
    - `when:30d` → último mes
    - `after:2024-12-01`
    - `before:2024-12-15`
    
    **Ejemplo:**
    ```
    inflación RD when:14d
    ```
    """)

num_articles = st.slider(
    "Cantidad de noticias POR TEMA",
    min_value=5,
    max_value=50,
    value=10
)

# ---------------- FUNCIONES AUX ----------------

def fetch_news(query, limit):
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={quote(query)}"
        "&hl=es-419"
        "&gl=DO"
        "&ceid=DO:es-419"
    )
    feed = feedparser.parse(rss_url)
    return feed.entries[:limit]

# ---------------- EJECUCIÓN ----------------

if st.button("Analizar sentimiento"):
    with st.spinner("Buscando información y analizando sentimiento..."):

        queries = [q.strip() for q in queries_text.split("\n") if q.strip()]

        if not queries:
            st.warning("⚠️ Debes ingresar al menos un tema.")
        else:
            items = []
            detailed_news = []

            for query in queries:

                # ---------- NOTICIAS ----------
                if use_news:
                    news = fetch_news(query, num_articles)
                    for item in news:
                        content = fetch_article_content(item.link)

                        items.append({
                            "source": "news",
                            "text": item.title + ". " + content,
                            "engagement": 1
                        })

                        sentiment = hybrid_sentiment(item.title, content)

                        detailed_news.append({
                            "query": query,
                            "title": item.title,
                            "link": item.link,
                            "sentiment": sentiment["sentiment"],
                            "confidence": sentiment["confidence"],
                            "vader": sentiment["vader"],
                            "roberta": sentiment["roberta"]
                        })

                # ---------- TWITTER / X ----------
                if use_twitter:
                    items += fetch_twitter_posts(query, limit=30)

            # ---------- ANÁLISIS ----------
            results = analyze_items(items)

            # ---------------- RESULTADOS ----------------

            st.subheader("Sentimiento por Fuente")
            st.caption("Por ahora solo hay una fuente activa funcional.")

            by_source = aggregate_by_source(results)
            cols = st.columns(len(by_source)) if by_source else []

            for col, (src, score) in zip(cols, by_source.items()):
                col.metric(
                    label=src.upper(),
                    value=f"{score:.2f}"
                )

            global_score = aggregate_global(results)

            st.subheader("Sentimiento Global")
            st.metric("Score Global", f"{global_score:.2f}")

            st.markdown("""
            **Interpretación del score:**
            - `+1` → Muy positivo  
            - `0` → Neutral  
            - `-1` → Muy negativo
            """)

            # ---------------- DETALLE NOTICIAS ----------------

            if use_news and detailed_news:
                st.subheader("Detalle por Noticia")

                for r in detailed_news:
                    st.markdown("---")
                    st.markdown(f"### {r['title']}")
                    st.write(f"**Tema:** {r['query']}")
                    st.write(f"**Sentimiento:** {r['sentiment']}")
                    st.write(f"**Confianza:** {r['confidence']}")
                    st.write(f"**VADER:** {r['vader']}")
                    st.write(f"**RoBERTa:** {r['roberta']}")
                    st.markdown(f"[Leer noticia]({r['link']})")

                    if r["vader"] != r["roberta"]:
                        st.warning("⚠️ Análisis divergente entre VADER y RoBERTa.")
