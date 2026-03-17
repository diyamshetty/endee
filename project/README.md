# ArXiv-RAG 🔍

A local, fully offline Retrieval-Augmented Generation (RAG) system for exploring recent ArXiv machine learning papers. Built with the **Endee Vector Database** to demonstrate a practical semantic search and RAG AI workflow.

---

## Project Overview

**Problem Statement:** Keeping up with the flood of machine learning research papers on ArXiv is impossible. Furthermore, Large Language Models have a knowledge cutoff and often hallucinate when asked about very recent academic developments.

**The Solution:** ArXiv-RAG is an open-source research assistant. It fetches the latest 2,000 papers on Large Language Models, generates high-dimensional vector embeddings of their abstracts, and stores them in an ultra-fast local **Endee Vector Database**. When a user asks a question, the system queries Endee for the most semantically relevant papers and feeds them to a local LLaMA 3.1 model to synthesize a grounded, accurate answer without hallucinations.

## System Design & Technical Approach

The system follows a standard offline RAG architecture:
1. **Data Ingestion:** The `arxiv` Python API fetches the 2,000 latest papers matching the query "Large Language Models".
2. **Embedding:** The `all-MiniLM-L6-v2` model from `sentence-transformers` generates 384-dimensional dense vectors for each paper's abstract.
3. **Storage & Retrieval:** The **Endee Vector Database** stores the vectors and their metadata. It uses Cosine Similarity and HNSW to perform lightning-fast Approximate Nearest Neighbor (ANN) searches.
4. **Generation:** An open-source **LLaMA 3.1 8B** model runs locally via **Ollama**. It receives a structured prompt containing the user's question and the context retrieved from Endee, synthesizing a final answer via HTTP streaming.
5. **UI:** A fully reactive, dark-themed **Streamlit** front-end.

*(For a more detailed technical breakdown, see [`TECHNICAL_EXPLAINER.md`](./TECHNICAL_EXPLAINER.md) included in this repo).*

---

## How Endee Vector Database is Used

Endee serves as the core retrieval engine for the RAG pipeline. The specific usage in `vector_store.py` highlights several enterprise-grade features:

- **Local Docker Deployment:** The Endee server runs locally via a lightweight Docker container, ensuring data privacy and zero API costs.
- **Optimized Index Creation:** The index is created using `space_type="cosine"` for semantic similarity and `precision=Precision.INT8`. Endee natively quantizes the floating-point vectors to 8-bit integers, drastically reducing memory overhead while maintaining high recall.
- **Metadata Filtering & Storage:** During the `upsert` phase, each vector is paired with a metadata payload containing the paper's title, authors, published date, abstract, and URL. 
- **Sub-millisecond Querying:** When a user poses a question, the encoded query vector is passed to `index.query(top_k=3)`. Endee searches across the 2,000 vectors and immediately returns the top documents, unpacking the metadata payload directly into the application context for the LLM.

---

## Setup and Execution Instructions

### Prerequisites

1. **Python 3.11+** installed
2. **Docker Desktop** installed and running
3. **Ollama** installed on your machine

### 1. Start the Endee Server
Endee provides the vector database backend. Start it using Docker Compose:
```bash
docker compose up -d
```
You can verify it is running by navigating to [http://localhost:8080/dashboard](http://localhost:8080/dashboard) in your browser.

### 2. Pull the LLaMA Model
Ensure Ollama is running, then pull the required LLaMA 3.1 model:
```bash
ollama pull llama3.1:8b-instruct
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
Start the Streamlit UI:
```bash
streamlit run app.py
```

### Usage Workflow
1. Open `http://localhost:8501` in your browser.
2. In the left sidebar, click **"⚡ Build Index"**.
   - *This will fetch the papers from ArXiv, embed them, and index them into Endee. This process takes 5-10 minutes on the first run, but data is persisted to the Endee Docker volume for instant startup in the future.*
3. Once the status shows **READY**, type a question into the main search bar (e.g., *"What are the latest techniques for mitigating LLM hallucinations?"*).
4. Click **"Search & Ask →"** to view the retrieved papers and the LLM's synthesized response.
