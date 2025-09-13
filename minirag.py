import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# --------------------------
# Load models
# --------------------------
@st.cache_resource
def load_models():
    # Embedding model
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Small local LLM for generation (tiny model to run on CPU)
    gen_model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
    generator = pipeline("text-generation", model=gen_model, tokenizer=tokenizer)
    
    return emb_model, generator

emb_model, generator = load_models()

# --------------------------
# Build FAISS vector store
# --------------------------
@st.cache_resource
def build_faiss(docs):
    embeddings = emb_model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Example knowledge base (you can replace with your own docs)
DOCUMENTS = [
    "The sun rises in the east and sets in the west.",
    "Streamlit is an open-source app framework for Machine Learning and Data Science.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "RAG stands for Retrieval-Augmented Generation, combining search with text generation.",
]

index, doc_embeddings = build_faiss(DOCUMENTS)

# --------------------------
# RAG pipeline
# --------------------------
def rag_pipeline(query, k=2):
    # Encode query
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    # Search top-k docs
    D, I = index.search(q_emb, k)
    retrieved_docs = [DOCUMENTS[i] for i in I[0]]
    
    # Create context
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based on the following documents:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Generate
    response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# --------------------------
# Streamlit UI
# --------------------------
st.title("📚 Mini RAG App")
st.write("A simple Retrieval-Augmented Generation (RAG) demo with FAISS + Hugging Face models.")

query = st.text_input("Ask a question:")

if st.button("🔍 Search & Generate") and query:
    answer = rag_pipeline(query)
    st.subheader("📝 Answer")
    st.write(answer)
