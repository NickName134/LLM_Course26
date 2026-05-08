import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pymupdf4llm
import os
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
DOCUMENTS_DIR = "ollama_pipeline/documents"


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

STOPWORDS = {
    "secondo", "documento", "cos", "cosa", "come", "quando", "quale", "quali",
    "sono", "essere", "viene", "vengono", "dell", "della", "delle", "degli",
    "allo", "alla", "agli", "alle", "con", "per", "nel", "nello", "nella",
    "nelle", "nei", "sul", "sulla", "sulle", "dal", "dallo", "dalla", "dalle",
    "una", "uno", "gli", "che", "del", "dei", "tra", "fra"
}

def chunk_text(text, chunk_size=1200):
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


def normalize_text(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_important_terms(question):
    return [
        term for term in normalize_text(question)
        if len(term) > 3 and term not in STOPWORDS
    ]


def keyword_score(question, chunk):
    question_terms = set(get_important_terms(question))
    chunk_terms = set(normalize_text(chunk))

    if not question_terms:
        return 0.0

    overlap = question_terms.intersection(chunk_terms)
    return len(overlap) / len(question_terms)


def lexical_candidates(question, chunks):
    important_terms = get_important_terms(question)
    candidates = []

    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        matched_terms = [term for term in important_terms if term in chunk_lower]

        if matched_terms:
            exact_boost = 0
            if "ritiro" in matched_terms and "commercio" in matched_terms:
                exact_boost = 5

            candidates.append({
                "chunk_index": index,
                "matched_terms": matched_terms,
                "match_count": len(matched_terms) + exact_boost,
                "chunk": chunk
            })

    return sorted(candidates, key=lambda item: item["match_count"], reverse=True)


def build_index(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_context(question, chunks, index, top_k=5, candidate_k=20):
    q_emb = embedding_model.encode([question])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    semantic_scores, indices = index.search(q_emb, min(candidate_k, len(chunks)))

    ranked_chunks = []
    for rank, chunk_index in enumerate(indices[0]):
        semantic_score = float(semantic_scores[0][rank])
        lexical_score = keyword_score(question, chunks[chunk_index])
        final_score = (0.60 * semantic_score) + (0.40 * lexical_score)

        ranked_chunks.append({
            "chunk_index": chunk_index,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "final_score": final_score,
            "source": "semantic+lexical",
            "chunk": chunks[chunk_index]
        })

    exact_candidates = lexical_candidates(question, chunks)
    for candidate in exact_candidates[:top_k]:
        chunk_index = candidate["chunk_index"]

        if not any(item["chunk_index"] == chunk_index for item in ranked_chunks):
            ranked_chunks.append({
                "chunk_index": chunk_index,
                "semantic_score": 0.0,
                "lexical_score": keyword_score(question, chunks[chunk_index]),
                "final_score": 1.00 + (0.10 * candidate["match_count"]),
                "source": f"exact_terms: {', '.join(candidate['matched_terms'])}",
                "chunk": chunks[chunk_index]
            })

    ranked_chunks = sorted(ranked_chunks, key=lambda item: item["final_score"], reverse=True)
    selected_chunks = ranked_chunks[:top_k]

    retrieved_chunks = []
    for rank, item in enumerate(selected_chunks):
        retrieved_chunks.append(
            f"[Chunk {rank + 1} | final: {item['final_score']:.3f} | semantic: {item['semantic_score']:.3f} | lexical: {item['lexical_score']:.3f} | source: {item['source']}]\n{item['chunk']}"
        )

    return "\n\n".join(retrieved_chunks)


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
Rispondi esclusivamente usando il contesto fornito.
Se nel contesto non trovi una risposta esplicita, rispondi esattamente: "Non lo so".
Non usare conoscenze esterne.
Non inventare informazioni.
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
Rispondi esclusivamente usando il contesto fornito.
Se nel contesto non trovi una risposta esplicita, rispondi esattamente: "Non lo so".
Non usare conoscenze esterne.
Non inventare informazioni.

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