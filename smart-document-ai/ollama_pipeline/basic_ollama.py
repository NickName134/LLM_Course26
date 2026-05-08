import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pymupdf4llm
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
DOCUMENTS_DIR = "ollama_pipeline/documents"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_index(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_context(question, chunks, index, top_k=2):
    q_emb = embedding_model.encode([question])
    q_emb = np.array(q_emb).astype("float32")

    _, indices = index.search(q_emb, top_k)

    return "\n\n".join([chunks[i] for i in indices[0]])


def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        return pymupdf4llm.to_markdown(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_documents(documents_dir):
    texts = []

    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)

        if filename.lower().endswith((".txt", ".md", ".pdf")):
            print(f"Caricamento documento: {filename}")
            texts.append(load_document(file_path))

    return "\n\n".join(texts)


def build_prompt(context, question, strategy):
    if strategy == "few-shot":
        return f"""
Rispondi SOLO usando il contesto.
Cita esplicitamente le informazioni presenti nel contesto.

Esempio 1:
Contesto: Il retrieval consiste nel recuperare informazioni rilevanti da una base documentale.
Domanda: Cos'è il retrieval?
Risposta: Il retrieval è il processo di recupero delle informazioni più rilevanti da documenti o dati.

Esempio 2:
Contesto: Un sistema RAG combina retrieval e generazione linguistica.
Domanda: Cosa combina un sistema RAG?
Risposta: Un sistema RAG combina il recupero di informazioni con la generazione di risposte tramite un modello linguistico.

Ora rispondi alla nuova domanda.

Contesto:
{context}

Domanda:
{question}

Risposta:
"""

    return f"""
Usa SOLO il seguente contesto per rispondere.
Se la risposta non è nel contesto, scrivi: "Non lo so".

Contesto:
{context}

Domanda:
{question}

Risposta:
"""


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        return f"Errore nella richiesta a Ollama: {response.text}"

    return response.json()["response"]


if __name__ == "__main__":
    full_text = load_all_documents(DOCUMENTS_DIR)
    chunks = chunk_text(full_text)
    index, _ = build_index(chunks)

    question = input("Domanda: ")
    strategy = input("Strategia (zero-shot/few-shot): ").strip().lower()

    if strategy not in ["zero-shot", "few-shot"]:
        print("Strategia non valida. Uso zero-shot come default.")
        strategy = "zero-shot"

    retrieved_context = retrieve_context(question, chunks, index)

    prompt_rag = build_prompt(retrieved_context, question, strategy)
    answer_rag = ask_ollama(prompt_rag)

    prompt_no_rag = build_prompt("Nessun contesto disponibile.", question, strategy)
    answer_no_rag = ask_ollama(prompt_no_rag)

    print("\n--- CONTESTO RECUPERATO ---")
    print(retrieved_context)

    print("\n--- STRATEGIA ---")
    print(strategy)

    print("\n--- RISPOSTA SENZA RAG ---")
    print(answer_no_rag)

    print("\n--- RISPOSTA CON RAG ---")
    print(answer_rag)