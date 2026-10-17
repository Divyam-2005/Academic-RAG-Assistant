import os
import re
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

def fix_latex_delimiters(text: str) -> str:
    # 1. Convert standard LaTeX block math \[ ... \] to $$ ... $$
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n$$\1$$\n', text, flags=re.DOTALL)
    
    # 2. Convert standard LaTeX inline math \( ... \) to $ ... $
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)
    
    # 3. Fix literal bracketed math [ \command ... ] to $$ \command ... $$
    text = re.sub(r'(?<!\w)\[\s*(\\[a-zA-Z]+.*?)\s*\]', r'\n$$\1$$\n', text, flags=re.DOTALL)
    
    # 4. Fix literal parenthesized math ( \command ... ) to $ \command ... $
    text = re.sub(r'(?<!\w)\(\s*(\\[a-zA-Z]+.*?)\s*\)', r'$\1$', text)
    
    # 5. Fix isolated raw LaTeX lines and table cells missing delimiters
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        
        # Wrap standalone lines starting with raw LaTeX commands in $$ ... $$
        if re.match(r'^\\(mathcal|mathbb|mathbf|mathrm|cfrac|frac|sum|prod|int|gamma|alpha|beta|delta|theta|lambda|sigma)\b', stripped) and '$' not in line:
            line = f"$${stripped}$$"
            
        # Fix table cells containing LaTeX commands with missing or broken $ delimiters
        elif '|' in line:
            parts = line.split('|')
            new_parts = []
            for part in parts:
                p_strip = part.strip()
                if re.search(r'\\[a-zA-Z]+', p_strip) and (p_strip.count('$') % 2 != 0 or '$' not in p_strip):
                    clean_cell = p_strip.replace('$', '')
                    p_strip = f"${clean_cell}$"
                new_parts.append(p_strip)
            line = ' | '.join(new_parts)
            
        fixed_lines.append(line)
        
    return '\n'.join(fixed_lines)

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
    ("system", (
        "You are an expert academic research assistant.\n"
        "Use the provided context to answer the question with mathematical precision.\n\n"
        "STRICT LATEX FORMATTING RULES:\n"
        "1. ALL mathematical expressions, variables, sets, and equations MUST be wrapped in math delimiters.\n"
        "2. Wrap inline math in single dollar signs $ ... $ (e.g., $C > 0$).\n"
        "3. Wrap standalone block equations in double dollar signs $$ ... $$ on their own lines.\n"
        "4. NEVER output raw LaTeX commands (like \\mathcal, \\mathbb, \\mathbf, \\gamma, \\frac) outside $ or $$.\n"
        "5. Inside Markdown tables, wrap the ENTIRE formula in $ ... $.\n"
        "6. DO NOT use \\( ... \\) or \\[ ... \\]. ONLY use $ and $$.\n\n"
        "If context is missing, state clearly what is known.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}")
])

                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": format_docs(retrieved_docs),
                    "question": user_query
                })

                answer = fix_latex_delimiters(answer)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if retrieved_docs:
                    with st.expander("📌 Source Metadata Citations"):
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get("source_name", "Uploaded PDF")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**[{i+1}] {source} (Page {page})**")
                            st.caption(f'"{doc.page_content[:250]}..."')