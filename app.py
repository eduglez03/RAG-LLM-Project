import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from ollama_client import generate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG · Ollama", layout="wide", page_icon="🧠")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 4rem; max-width: 1100px; }

.rag-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 1.5rem;
}
.rag-header .icon { font-size: 2rem; line-height: 1; }
.rag-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0f4ff;
    margin: 0;
    letter-spacing: -0.02em;
}
.rag-header .subtitle {
    font-size: 0.8rem;
    color: #4a6080;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}

.stTextInput > div > div > input {
    background-color: #131720 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 8px !important;
    color: #c9d8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
.stTextInput label {
    color: #4a6080 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #131720;
    border: 1px solid #1e2d42;
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #60a5fa;
    margin-bottom: 1rem;
    animation: pulse-border 1.5s ease infinite;
}
@keyframes pulse-border {
    0%, 100% { border-color: #1e2d42; }
    50%       { border-color: #3b82f6; }
}

.answer-card {
    background: #131720;
    border: 1px solid #1e2d42;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin: 1.25rem 0 2rem;
    line-height: 1.75;
    font-size: 0.97rem;
    color: #d1ddf5;
}

/* Tarjeta de "no encontrado" */
.answer-card.not-found {
    border-left-color: #f59e0b;
    color: #92847a;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3b5270;
    margin-bottom: 0.75rem;
    margin-top: 0.25rem;
}

.context-chunk {
    background: #0f1318;
    border: 1px solid #181f2b;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #4a6080;
    line-height: 1.65;
    position: relative;
    overflow: hidden;
}
.context-chunk::before {
    content: attr(data-index);
    position: absolute;
    top: 8px;
    right: 12px;
    font-size: 0.65rem;
    color: #1e2d42;
    font-weight: 600;
}

.rag-divider {
    border: none;
    border-top: 1px solid #1a2030;
    margin: 1.75rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="icon">🧠</div>
    <div>
        <h1>RAG · Ollama</h1>
        <div class="subtitle">retrieval-augmented generation // local inference</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ChromaDB + Embeddings ─────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    chroma = chromadb.PersistentClient(path="vectorstore")
    collection = chroma.get_or_create_collection(
        name="my_docs",
        metadata={"hnsw:space": "cosine"}
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder

# ← Primero cargar recursos
collection, embedder = load_resources()

# ← DESPUÉS definir retrieve (ya tiene acceso a embedder)
def retrieve(query, k=3, threshold=0.7):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "distances"]
    )
    docs      = results["documents"][0]
    distances = results["distances"][0]

    print("=== DEBUG DISTANCIAS (coseno) ===")
    for doc, dist in zip(docs, distances):
        print(f"Distancia: {dist:.4f} | Fragmento: {doc[:80]}...")

    filtered = [doc for doc, dist in zip(docs, distances) if dist < threshold]
    return filtered

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input("", placeholder="Escribe tu pregunta aquí…", label_visibility="collapsed")

# ── Results ───────────────────────────────────────────────────────────────────
if query:
    st.markdown('<div class="status-pill">⟳ &nbsp;Buscando en la base de conocimiento…</div>', unsafe_allow_html=True)
    docs = retrieve(query)

    # ── Sin resultados relevantes ─────────────────────────────────────────────
    if not docs:
        st.markdown('<div class="section-label">▸ Respuesta generada</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="answer-card not-found">'
            '⚠ &nbsp;No encontré información sobre eso en los documentos.'
            '</div>',
            unsafe_allow_html=True
        )
        st.stop()

    # ── Generar respuesta ─────────────────────────────────────────────────────
    context = "\n\n".join(docs)
    st.markdown('<div class="status-pill">◈ &nbsp;Generando respuesta con Ollama…</div>', unsafe_allow_html=True)

    prompt = f"""Eres un asistente que ÚNICAMENTE responde basándose en el contexto proporcionado.
Si la respuesta no se encuentra en el contexto, responde exactamente: "No encontré información sobre eso en los documentos."
No uses conocimiento externo bajo ninguna circunstancia.

Contexto:
{context}

Pregunta: {query}
Respuesta:"""

    answer = generate(prompt)

    # ── Respuesta ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">▸ Respuesta generada</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="rag-divider"/>', unsafe_allow_html=True)

    # ── Fragmentos recuperados ────────────────────────────────────────────────
    st.markdown('<div class="section-label">▸ Fragmentos recuperados</div>', unsafe_allow_html=True)
    for i, d in enumerate(docs, 1):
        st.markdown(
            f'<div class="context-chunk" data-index="#{i}">{d[:300]}…</div>',
            unsafe_allow_html=True
        )