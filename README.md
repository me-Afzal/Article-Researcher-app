# 🧠 Articles Research Tool – A RAG App for Smarter Content Exploration

Welcome to the **Articles Research Tool**, a Retrieval-Augmented Generation (RAG) powered app that helps you explore and understand articles with the power of Large Language Models — so you don’t have to read every article line by line.

## 🚀 Live Demo

👉 Try it here: [https://article-researcher-app-cbngx5mmmxxdtjzfewngtn.streamlit.app](https://article-researcher-app-cbngx5mmmxxdtjzfewngtn.streamlit.app)

---

## 📌 Features

✅ Paste one or more article URLs (same topic or different topics)  
✅ Automatically scrapes and processes content from the web  
✅ Splits and embeds the text using `Hugging Face` embeddings  
✅ Stores vectors in a FAISS Vector Database  
✅ Uses Retrieval-Augmented Generation (RAG) to answer questions  
✅ Provides source references for transparency  
✅ Clean and interactive Streamlit-based UI

---

## 🛠️ Tech Stack

- **[LangChain](https://www.langchain.com/)** – for chaining the retrieval and generation process  
- **[Hugging Face Embeddings](https://huggingface.co/)** – for high-quality semantic vector embeddings  
- **[FAISS](https://github.com/facebookresearch/faiss)** – for efficient vector similarity search  
- **[Together API](https://docs.together.ai/docs/inference)** – to query powerful open-source LLMs  
- **[Streamlit](https://streamlit.io/)** – for building the frontend interface quickly and beautifully

---

## 💡 How It Works

1. **Collect**: User provides article URLs
2. **Scrape**: App extracts content using `UnstructuredURLLoader`
3. **Chunk**: Text is split into manageable pieces
4. **Embed**: Each chunk is converted into embeddings using Hugging Face
5. **Store**: Vectors are stored in a local FAISS DB
6. **Ask**: User types in a question
7. **Retrieve + Generate**: Top relevant chunks are retrieved and sent to the LLM via Together API
8. **Answer**: Response is generated and shown with optional source links

---

## 🧑‍💻 Ideal For

- 📚 Researchers
- 🧑‍🎓 Students
- 🗞️ Journalists
- ✍️ Content Creators
- Anyone diving deep into topic-based article research

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/articles-research-tool.git
cd articles-research-tool
pip install -r requirements.txt
streamlit run app.py
