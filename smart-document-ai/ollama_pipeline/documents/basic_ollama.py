import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

with open("ollama_pipeline/documents/sample.txt", "r", encoding="utf-8") as f:
    context = f.read()

question = input("Domanda: ")

prompt = f"""
Usa SOLO il seguente contesto per rispondere.

Contesto:
{context}

Domanda:
{question}

Risposta:
"""

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
)

print(response.json()["response"])