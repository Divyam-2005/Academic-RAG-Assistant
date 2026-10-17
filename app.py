import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="PDF Academic Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Academic RAG Assistant")
st.caption("Upload technical textbooks or research papers and chat with them using local embeddings & Groq LLM.")

load_dotenv()

# --- 2. Sidebar: Dynamic User Settings & Uploads ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # User-provided API Key (Pre-fills with .env if present, but allows custom input)
    groq_api_key = st.text_input(
        "Groq API Key:",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get your free API key at https://console.groq.com"
    )
    
    # Model Selector
    selected_model = st.selectbox(
        "Select Model:",
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.1-8b-instant"]
    )
    
    # Temperature Slider
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values (0.0–0.2) are ideal for factual & academic Q&A."
    )
    
    st.divider()
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    process_button = st.button("Build / Update Vector Store", type="primary")

# --- 3. Cache Embedding Model ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()

# --- 4. Process PDFs & Build Vector Store ---
if process_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing PDFs and generating vector embeddings..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_name"] = uploaded_file.name
                    all_docs.extend(docs)
                finally:
                    os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(all_docs)

            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.sidebar.success(f"Processed {len(all_docs)} pages into {len(chunks)} chunks!")

# --- 5. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Input your Groq API key and upload study materials on the left to start."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    if "vectorstore" not in st.session_state:
        st.error("Please upload and process PDF documents first.")
    elif not groq_api_key:
        st.error("Please provide a valid Groq API key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sources and generating answer..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=selected_model,
                    temperature=temperature
                )

                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                retrieved_docs = retriever.invoke(user_query)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert academic research assistant.\nUse the provided context to answer the question with mathematical precision.\nIf context is missing, state clearly what is known.\n\nContext:\n{context}"),
                    ("human", "{question}")
                ])

                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": format_docs(retrieved_docs),
                    "question": user_query
                })

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if retrieved_docs:
                    with st.expander("📌 Source Metadata Citations"):
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get("source_name", "Uploaded PDF")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**[{i+1}] {source} (Page {page})**")
                            st.caption(f'"{doc.page_content[:250]}..."')