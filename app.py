import os
import re
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Academic RAG Assistant", layout="wide")
st.title("📚 Academic RAG Assistant")
st.caption("Upload technical textbooks or research papers and chat with them using local embeddings & Groq LLM.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key:", type="password", help="Get your API key from https://console.groq.com/")
    
    selected_model = st.selectbox(
        "Select Model:",
        [
            "openai/gpt-oss-120b",
            "qwen-2.5-32b",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]
    )
    
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    build_button = st.button("Build / Update Vector Store", type="primary")

# --- Initialize Embeddings ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# --- Robust LaTeX Sanitizer Pipeline ---
def fix_latex_delimiters(text: str) -> str:
    if not text:
        return text

    # Step 1: Fix escaped ampersands inside alignment environments (\& -> &)
    text = re.sub(
        r'\\begin\{([a-zA-Z]+)\}(.*?)\\end\{\1\}',
        lambda m: f"\\begin{{{m.group(1)}}}{m.group(2).replace(r'\&', '&')}\\end{{{m.group(1)}}}",
        text,
        flags=re.DOTALL
    )

    # Step 2: Standardize bracket delimiters [ ] and \( \)
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n$$\1$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)

    # Step 3: Clean and wrap multi-line environments strictly in $$...$$
    def wrap_environment(match):
        env_str = match.group(0)
        clean_env = re.sub(r'\$+', '', env_str).strip()
        return f"\n$$\n{clean_env}\n$$\n"

    text = re.sub(r'\\begin\{[a-zA-Z]+\}.*?\\end\{[a-zA-Z]+\}', wrap_environment, text, flags=re.DOTALL)

    # Step 4: Protect existing valid math blocks using tokens
    placeholders = []
    def mask_math(match):
        placeholders.append(match.group(0))
        return f"___MATH_TOKEN_{len(placeholders)-1}___"

    text = re.sub(r'\$\$.*?\$\$', mask_math, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', mask_math, text)

    # Step 5: Process unmasked lines containing raw LaTeX commands
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or '___MATH_TOKEN_' in line or stripped.startswith('```'):
            fixed_lines.append(line)
            continue

        if re.search(r'\\[a-zA-Z]+', stripped):
            if re.match(r'^[\-\*\d\.]+\s+', stripped):
                if ':' in line:
                    prefix, eq = line.split(':', 1)
                    line = f"{prefix}: ${eq.strip()}$"
                else:
                    line = f"${line}$"
            elif any(kw in stripped for kw in ['=', r'\min', r'\max', r'\sum', r'\prod', r'\int', r'\frac']):
                line = f"$${stripped}$$"
            else:
                line = f"${stripped}$"

        fixed_lines.append(line)

    text = '\n'.join(fixed_lines)

    # Step 6: Restore protected math tokens
    for i, token in enumerate(placeholders):
        text = text.replace(f"___MATH_TOKEN_{i}___", token)

    # Step 7: Final cleanup of duplicate delimiters
    text = re.sub(r'\$\$\s*\$\$', '', text)
    return text

# --- Vector Store Processing ---
if build_button:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing documents and generating embeddings..."):
            all_docs = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                try:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_name"] = file.name
                    all_docs.extend(docs)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(all_docs)

            vectorstore = FAISS.from_documents(split_docs, embeddings)
            st.session_state.vectorstore = vectorstore
            st.sidebar.success(f"Processed {len(all_docs)} pages into {len(split_docs)} chunks!")

# --- Chat Interface ---
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
                        "1. ALL mathematical expressions MUST be wrapped in math delimiters.\n"
                        "2. Wrap inline math in single dollar signs $...$.\n"
                        "3. Wrap standalone block equations in double dollar signs $$...$$ on new lines.\n"
                        "4. For multi-line aligned equations, wrap the ENTIRE \\begin{{aligned}} ... \\end{{aligned}} block in $$ ... $$. DO NOT use \\& inside aligned blocks; use &.\n"
                        "5. NEVER output raw LaTeX commands outside $ or $$.\n\n"
                        "Context:\n{context}"
                    )),
                    ("human", "{question}")
                ])

                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": format_docs(retrieved_docs),
                    "question": user_query
                })

                clean_answer = fix_latex_delimiters(answer)

                st.write(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

                if retrieved_docs:
                    with st.expander("📌 Source Metadata Citations"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            source = doc.metadata.get("source_name", "Unknown Document")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**[{i}] {source}** — *Page {page}*")
                            st.caption(f'"{doc.page_content[:200]}..."')