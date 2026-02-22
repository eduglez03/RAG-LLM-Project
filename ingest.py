import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

chroma_client = chromadb.PersistentClient(path="vectorstore")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return [embedder.encode(t).tolist() for t in texts]

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Divide el texto en fragmentos con solapamiento"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # avanza dejando solapamiento
    return chunks

def ingest_docs(folder="docs"):
    try:
        chroma_client.delete_collection("my_docs")
        print("🗑️  Colección anterior eliminada")
    except:
        pass

    # Añadir metadata="hnsw:space" para usar coseno
    collection = chroma_client.get_or_create_collection(
        name="my_docs",
        metadata={"hnsw:space": "cosine"}
    )
    total_chunks = 0

    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        text = ""

        if fname.lower().endswith(".pdf"):
            text = pdf_to_text(path)
        elif fname.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        if text.strip() == "":
            continue

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print(f"📄 {fname} → {len(chunks)} fragmentos")

        for i, chunk in enumerate(chunks):
            if chunk.strip() == "":
                continue
            chunk_id = f"{fname}_chunk_{i}"
            embedding = embed_texts([chunk])
            collection.add(
                documents=[chunk],
                embeddings=embedding,
                ids=[chunk_id]
            )
            total_chunks += 1

    print(f"✅ Ingesta completada — {total_chunks} fragmentos indexados")

if __name__ == "__main__":
    ingest_docs()