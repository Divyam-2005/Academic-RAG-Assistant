# 📚 Academic RAG Assistant: Chat with Technical PDFs

An interactive Retrieval-Augmented Generation (RAG) web application designed to help students, researchers, and engineers query dense academic textbooks, research papers, and study materials. 

Built with a focus on processing complex theoretical and mathematical concepts, this tool dynamically ingests PDFs, indexes them using local open-source embeddings, and retrieves mathematically precise answers utilizing Groq's high-speed inference models.

## ✨ Key Features

* **Dynamic Document Processing:** Upload multiple technical PDFs on the fly. The pipeline uses `PyMuPDFLoader` and recursive character splitting to preserve the structural integrity of dense text and equations.
* **Zero-Cost Local Embeddings:** Utilizes Hugging Face's `all-MiniLM-L6-v2` (`sentence-transformers`) running entirely on your local CPU to build the vector index, eliminating external embedding API costs and ensuring document privacy during the indexing phase.
* **High-Speed Inference (BYOK):** Features a "Bring Your Own Key" interface for Groq, allowing users to select between top-tier open-source models (like `openai/gpt-oss-120b` or `qwen/qwen3.8-27b`) for sub-second generation speeds.
* **Source Metadata Citations:** Ensures absolute transparency and mitigates hallucination by citing the exact source filename and page number for every retrieved context chunk.
* **Tunable Parameters:** Interactive sidebar to adjust model temperature for strict factual retrieval vs. creative synthesis.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain (LCEL)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Hugging Face (`sentence-transformers`)
* **LLM API:** Groq
* **Document Parsing:** PyMuPDF

## 🚀 How to Run Locally

### 1. Clone the Repository
`git clone https://github.com/Divyam-2005/Academic-RAG-Assistant.git`
`cd Academic-RAG-Assistant`

### 2. Install Dependencies
`pip install -r requirements.txt`

### 3. Launch the Application
`python -m streamlit run app.py`

The application will automatically open in your default web browser at `http://localhost:8501`.

## 💡 Usage Guide

1. **Enter API Key:** Obtain a free API key from the [Groq Console](https://console.groq.com/) and paste it into the sidebar.
2. **Upload Documents:** Drag and drop your `.pdf` textbooks, lecture notes, or research papers into the sidebar uploader.
3. **Build Vector Store:** Click the "Build / Update Vector Store" button. The app will parse the documents, chunk the text, and generate local embeddings.
4. **Start Chatting:** Ask complex, multi-layered questions about the uploaded material. Click the "📌 Source Metadata Citations" expander under the AI's response to verify the exact page references.

## 🧠 Future Roadmap

* Implement hybrid search (BM25 + FAISS) for better exact-keyword retrieval on specific mathematical theorems.
* Integrate layout-aware parsing for better handling of multi-column research papers and embedded tables.