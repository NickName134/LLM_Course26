import os
import streamlit as st
from ollama_pipeline.basic_ollama import (
    load_all_documents,
    chunk_text,
    build_index,
    retrieve_context,
    build_prompt,
    ask_ollama
)

DOCUMENTS_DIR = "ollama_pipeline/documents"

st.set_page_config(page_title="Smart Document AI", page_icon="📄")

st.title("📄 Smart Document AI")
st.caption("Assistente RAG per interrogare documenti PDF, TXT e Markdown con Ollama.")



def list_documents(documents_dir):
    supported_extensions = (".txt", ".md", ".pdf")
    return [
        filename for filename in os.listdir(documents_dir)
        if filename.lower().endswith(supported_extensions)
    ]


# Save uploaded file function
def save_uploaded_file(uploaded_file, documents_dir):
    os.makedirs(documents_dir, exist_ok=True)
    file_path = os.path.join(documents_dir, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def load_all_documents_for_single_file(file_path):
    from ollama_pipeline.basic_ollama import load_document
    return load_document(file_path)


@st.cache_resource
def load_pipeline(selected_documents):
    texts = []

    for document in selected_documents:
        file_path = os.path.join(DOCUMENTS_DIR, document)
        texts.append(load_all_documents_for_single_file(file_path))

    full_text = "\n\n".join(texts)
    chunks = chunk_text(full_text)
    index, _ = build_index(chunks)
    return chunks, index


if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

documents = list_documents(DOCUMENTS_DIR)

with st.sidebar:
    st.header("⚙️ Impostazioni")

    uploaded_file = st.file_uploader(
        "Carica un documento",
        type=["pdf", "txt", "md"]
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_file:
        save_uploaded_file(uploaded_file, DOCUMENTS_DIR)
        st.session_state.last_uploaded_file = uploaded_file.name
        st.cache_resource.clear()
        st.success(f"Documento caricato: {uploaded_file.name}")
        st.rerun()

    strategy = st.selectbox(
        "Strategia di prompting",
        ["zero-shot", "few-shot"]
    )

    documents = list_documents(DOCUMENTS_DIR)

    st.subheader("📚 Documenti caricati")

    selected_documents = []

    if documents:
        for document in documents:
            key = f"doc_select_{document}"

            # default: selezionato
            if key not in st.session_state:
                st.session_state[key] = True

            checked = st.checkbox(document, key=key)

            if checked:
                selected_documents.append(document)
    else:
        st.warning("Nessun documento trovato nella cartella documents.")

    if st.button("🔄 Reset chat"):
        st.session_state.messages = []
        st.rerun()

if not documents:
    st.stop()

if not selected_documents:
    st.info("Seleziona almeno un documento dalla sidebar per iniziare.")
    st.stop()

chunks, index = load_pipeline(tuple(selected_documents))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Fai una domanda sui documenti...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Recupero il contesto e genero la risposta..."):
        context = retrieve_context(question, chunks, index)
        prompt = build_prompt(context, question, strategy)
        answer = ask_ollama(prompt)

    with st.chat_message("assistant"):
        st.write(answer)

        with st.expander("📎 Contesto recuperato"):
            st.write(context)

        with st.expander("📚 Documenti usati"):
            for document in selected_documents:
                st.write(f"- {document}")

    st.session_state.messages.append({"role": "assistant", "content": answer})