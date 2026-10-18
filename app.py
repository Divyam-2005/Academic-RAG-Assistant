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

# --- Sidebar: Configuration & Document Management ---
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

# --- Initialize Embeddings (Cached for Performance) ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# --- Helper Function: Tokenized LaTeX Sanitizer ---
def fix_latex_delimiters(text: str) -> str:
    if not text:
        return text

    # 1. Standardize standard LaTeX block and inline math delimiters
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\n$$\1$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$\1$', text, flags=re.DOTALL)

    # 2. Fix multi-line environments (\begin{aligned} ... \end{aligned})
    def sanitize_env(match):
        env_content = match.group(0)
        clean_content = env_content.replace('$$', '').strip()
        return f"\n$$\n{clean_content}\n$$\n"

    text = re.sub(r'\\begin\{[a-zA-Z]+\}.*?\\end\{[a-zA-Z]+\}', sanitize_env, text, flags=re.DOTALL)

    # 3. Clean empty/nested dollar signs
    text = re.sub(r'\$\$\s*\$\$', '', text)

    # 4. Tokenize and protect already-valid math blocks ($$...$$ and $...$)
    math_placeholders = []
    def protect_math(match):
        math_placeholders.append(match.group(0))
        return f"___MATH_PLACEHOLDER_{len(math_placeholders)-1}___"

    # Protect block math first, then inline math
    text = re.sub(r'\$\$.*?\$\$', protect_math, text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', protect_math, text)

    # 5. Process remaining unwrapped text lines
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines, placeholders, or code blocks
        if not stripped or '___MATH_PLACEHOLDER_' in line or stripped.startswith('```'):
            fixed_lines.append(line)
            continue

        # If line contains raw LaTeX commands that were missed by the LLM
        if re.search(r'\\[a-zA-Z]+', stripped):
            if stripped.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.', '6.')):
                if ':' in line:
                    prefix, eq = line.split(':', 1)
                    line = f"{prefix}: ${eq.strip()}$"
                else:
                    line = f"${line}$"
            elif '=' in stripped or r'\min' in stripped or r'\sum' in stripped or r'\max' in stripped or r'\frac' in stripped:
                line = f"$${stripped}$$"
            else:
                line = f"${stripped}$"

        fixed_lines.append(line)

    text = '\n'.join(fixed_lines)

    # 6. Restore protected math blocks
    for i, block in enumerate(math_placeholders):
        text = text.replace(f"___MATH_PLACEHOLDER_{i}___", block)

    # Clean any accidental double-wrapping
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
                        "1. ALL mathematical expressions, variables, sets, and equations MUST be wrapped in math delimiters.\n"
                        "2. Wrap inline math in single dollar signs $...$ (e.g. $w \\in \\mathbb{{R}}^d$).\n"
                        "3. Wrap standalone block equations in double dollar signs $$...$$ on new lines.\n"
                        "4. For multi-line aligned equations, wrap the ENTIRE \\begin{{aligned}} ... \\end{{aligned}} block in a single pair of $$ ... $$. DO NOT place $$ inside aligned environments.\n"
                        "5. NEVER output raw LaTeX commands (like \\mathcal, \\mathbb, \\mathbf, \\gamma, \\frac) outside $ or $$.\n\n"
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

                # Sanitize LaTeX formatting for Streamlit
                clean_answer = fix_latex_delimiters(answer)

                st.write(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

                # Display Citations
                if retrieved_docs:
                    with st.expander("📌 Source Metadata Citations"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            source = doc.metadata.get("source_name", "Unknown Document")
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**[{i}] {source}** — *Page {page}*")
                            st.caption(f'"{doc.page_content[:200]}..."')