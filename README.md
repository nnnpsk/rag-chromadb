# RAG + ChromaDB + Ollama

### Setup Instructions
1. Install Ollama from https://ollama.com/download
2. ollama pull llama3.2 (ollama cp llama3.2 llama3 && ollama serve)
3. git clone https://github.com/nnnpsk/rag-chromadb.git
4. pip install -r requirements.txt
5. streamlit run rag.py

6. Upload a PDF/TXT files --> Split into chunks --> Embeddings stored in local chromadb
7. Ask a question about the document uploaded --> retrieves most relavant chunks --> passes to LLM --> generates answer --> shown in streamlit UI.

![UI Screenshot](https://raw.githubusercontent.com/nnnpsk/rag-chromadb/main/ui.png)
   

   
