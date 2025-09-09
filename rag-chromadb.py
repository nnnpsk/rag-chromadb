import os
import re
import json
import torch
import ollama
import PyPDF2
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---------- INIT ----------
client = OpenAI(base_url="http://localhost:11434/v1", api_key="llama3")

# Setup ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="vault",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
        # model_name="nomic-embed-text-v1.5"
    ),
)

# ---------- UTILS ----------
# def normalize_and_chunk(text, max_len=1000):
#     text = re.sub(r'\s+', ' ', text).strip()
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     chunks, current = [], ""
#     for sentence in sentences:
#         if len(current) + len(sentence) + 1 < max_len:
#             current += sentence + " "
#         else:
#             chunks.append(current.strip())
#             current = sentence + " "
#     if current:
#         chunks.append(current.strip())
#     return chunks

def normalize_and_chunk(text, max_len=512, overlap=50):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_len:
            current_chunk += sentence + " "
        else:
            # Add current chunk
            chunks.append(current_chunk.strip())
            # Start next chunk with overlap from previous
            if overlap > 0:
                # Take the last `overlap` characters from current chunk
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + " " + sentence + " "
            else:
                current_chunk = sentence + " "

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_chunks(chunks):
    existing = set(collection.get()['documents'])
    new_chunks = [chunk for chunk in chunks if chunk not in existing]
    for chunk in new_chunks:
        collection.add(documents=[chunk], ids=[f"{hash(chunk)}"])
    return len(new_chunks)

def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    chunks = normalize_and_chunk(text)


##################
    for i, chunk in enumerate(chunks):
        if "Trump" in chunk:
          print(f"Found target string in chunk {i}:\n{chunk[:200]}...")

###############
    return process_chunks(chunks)

def process_txt(file):
    text = file.read().decode("utf-8")
    chunks = normalize_and_chunk(text)
    return process_chunks(chunks)

# ---------- RAG FUNCTIONS ----------
def get_relevant_context(query, top_k=5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

def ollama_chat(user_input, system_message, relevant_context, ollama_model):
    if relevant_context:
        user_input += "\n\nRelevant Context:\n" + "\n".join(relevant_context)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=256,
        n=1,
        temperature=0.1,
    )
    return response.choices[0].message.content

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Local RAG",page_icon="💬",layout="wide")
st.title("RAG + ChromaDB + Ollama = 💬")

# Initialize session state
if 'upload_msg' not in st.session_state:
    st.session_state.upload_msg = ""

if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# Sidebar file upload
st.sidebar.header("Upload Documents")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
uploaded_txt = st.sidebar.file_uploader("Upload Text", type=["txt"])

# ---------- File Upload Handler ----------
def handle_upload(file, filetype):
    if file is None:
        return
    # Process only if a new file is uploaded
    if file.name != st.session_state.last_uploaded_filename:
        if filetype == "pdf":
            count = process_pdf(file)
        elif filetype == "txt":
            count = process_txt(file)
        else:
            return

        st.session_state.upload_msg = f"✅ {count} new chunks added"
        st.session_state.last_uploaded_filename = file.name

# Process each file upload
handle_upload(uploaded_pdf, "pdf")
handle_upload(uploaded_txt, "txt")

# Display upload message if any
if st.session_state.upload_msg:
    st.sidebar.success(st.session_state.upload_msg)
    st.session_state.upload_msg = ""  # Clear message after showing

# ---------- Chat Interface ----------
st.subheader("Question about Your Document")
user_query = st.text_input("Enter your question:", key="input_box")

if (st.button("Ask") or st.session_state.input_box):
    if user_query.strip():
        with st.spinner("🤔 Thinking..."):
            context = get_relevant_context(user_query)
            print("Context Pulled from Documents: \n\n" + str(context))








            system_msg = (
                "You are a helpful assistant. ONLY use the context provided in the 'Relevant Context' section "
                "to answer the user's question. If the answer is not present in the context, respond with: "
                "'Sorry, I couldn't find relevant information in the provided documents.' "
                "Do NOT use external knowledge. Do NOT make assumptions. Stay strictly within the supplied context."
            )
            answer = ollama_chat(user_query, system_msg, context, "llama3")
        st.success("### Response:")
        st.write(answer)
    else:
        st.warning("Enter your question again:")
