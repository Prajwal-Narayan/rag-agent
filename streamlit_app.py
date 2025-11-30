import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="Advanced RAG Agent", page_icon="🤖", layout="wide")

st.title("Advanced Agentic RAG")
st.markdown("### Powered by LangChain, FastAPI, and Cohere")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Ingestion
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Ingesting... (Chunking -> Embedding -> Indexing)"):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/ingest", files=files)
                if response.status_code == 200:
                    st.success(f"✅ Ingested! processed {response.json()['chunks_processed']} chunks.")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    st.divider()
    st.markdown("**System Status:**")
    st.markdown("- Router: ✅ Active")
    st.markdown("- Multi-Query: ✅ Active")
    st.markdown("- Reranker: ✅ Active")
    st.markdown("- Self-Correction: ✅ Active")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking... (Routing -> Searching -> Grading)"):
                payload = {"message": prompt}
                response = requests.post(f"{API_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    # Display Answer
                    message_placeholder.markdown(answer)
                    
                    # Display Sources in an Expander
                    if sources and sources[0] != "General Conversation (No Search Performed)":
                        with st.expander("📚 Verified Sources (Cohere Reranked)"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**Source {idx+1}:**\n{source[:300]}...")
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error: {response.text}")
                    
        except Exception as e:
            st.error(f"Connection Error: Is the FastAPI server running? \n {e}")