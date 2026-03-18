# Architecture Deep Dive: Building a RAG System with Endee Vector Database

Welcome, engineering students! This document is an in-depth, step-by-step guide to understanding the **ArXiv-RAG** project we just built. We'll demystify Retrieval-Augmented Generation (RAG), explain how Vector Databases like **Endee** work under the hood, and walk through the exact code we used to build it.

---

## 1. The Core Problem: Why Do We Need RAG?

Large Language Models (LLMs) like ChatGPT, Claude, and LLaMA are incredibly smart. However, they have three major weaknesses:

1. **Information Cutoff**: They only know what they were trained on. A model trained in 2023 knows nothing about a research paper published last week.
2. **Hallucinations**: When an LLM doesn't know the answer, it tends to confidently make things up (hallucinate).
3. **No Private Data**: By default, LLMs cannot see your company's private documents or your local files.

### The Solution: Retrieval-Augmented Generation (RAG)

Instead of relying on the LLM's internal memory, we give the model an "open book" test. 

1. **Retrieve**: When a user asks a question, we rapidly search an external database for the most relevant documents (e.g., academic papers).
2. **Augment**: We take those retrieved documents and paste them into the LLM's prompt. 
   *(Example: "Use these three papers to answer the question: [Papers] User Question: ...")*
3. **Generate**: The LLM reads the papers provided in the prompt and generates a factual, grounded answer.

To make the "Retrieve" step blazingly fast and semantically accurate, we use a **Vector Database**.

---

## 2. Demystifying Vector Embedded and Databases

Traditional databases (like SQL) search for exact keyword matches. If you search for "automobile", it won't find documents that only say "car".

**Vector Databases** use *semantic search*. They understand the *meaning* of words.

### How Embeddings Work
We use a small AI model (like `sentence-transformers`) to convert text into a high-dimensional mathematical array called a **Vector Embedding**.
For example, a vector might have 384 dimensions: `[0.12, -0.44, 0.88, ...]`.

In high-dimensional space, sentences with similar *meanings* end up close together geometrically. 
* The vector for "Car" is physically close to the vector for "Automobile".
* The vector for "Apple" (the fruit) is far from "Apple" (the company) depending on the context.

### The Role of Endee Vector Database
Searching for the "closest" vectors among millions of documents requires immense computational power. **Endee** is a high-performance vector database specifically engineered for this task. 
It uses algorithms like **HNSW (Hierarchical Navigable Small World)** to approximate the nearest neighbors in milliseconds, rather than checking every single vector one by one.

---

## 3. How ArXiv-RAG Works (Step-by-Step)

Our system consists of 5 main components. Let's look at how they interact.

### Phase 1: Ingestion (`ingest.py`)
We use the `arxiv` Python API to download the metadata and abstracts of the 2,000 most recent research papers containing the keyword "Large Language Models". We save this to a local JSON file to avoid re-downloading.

### Phase 2: Embedding (`embedder.py`)
We take the abstract of every paper and feed it through an open-source embedding model called `all-MiniLM-L6-v2`. 
```python
# From embedder.py
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)
```
This turns our 2,000 text abstracts into 2,000 arrays of floating-point numbers (each 384 dimensions long). Normalizing the embeddings allows Endee to use **Cosine Similarity** to compare them extremely efficiently.

### Phase 3: The Vector Database (`vector_store.py`)
This is where **Endee** shines. We connect to a local Endee server and create an "Index" (like a table in SQL).

```python
# Creating the index in Endee
client.create_index(
    name="arxiv_papers",
    dimension=384,          # Because our embeddings are 384 numbers long
    space_type="cosine",    # We use cosine similarity to measure distance
    precision=Precision.INT8, # Quantizes vectors to 8-bit integers to save memory!
    M=16,
    ef_con=200,
)
```

Look at `precision=Precision.INT8`. This is a powerful feature. Instead of storing massive 32-bit floats, Endee compresses the vectors. This dramatically reduces memory usage and speeds up search times with almost zero loss in accuracy.

Next, we **upsert** (insert or update) our papers into Endee. We don't just insert the vector; we also include the "payload" or metadata (the title, abstract, and URL):
```python
# We insert standard dictionaries. The SDK handles serialization.
records.append({
    "id": paper["id"],
    "vector": embedded_vector,
    "meta": {
        "title": paper["title"],
        "abstract": paper["abstract"],
        "url": paper["url"]
    }
})
index.upsert(records)
```

### Phase 4: Querying (Also `vector_store.py`)
When a user types a query (e.g., *"How to reduce AI hallucinations?"*):
1. We convert their text query into a 384-dimensional vector using the exact same embedding model.
2. We ask Endee to find the 3 vectors in its database that are closest to the query vector.

```python
results = index.query(
    vector=query_vector,
    top_k=3, # Return top 3 matches
)
```
Because Endee uses the HNSW algorithm, it traverses a hierarchical graph to find these matches in a fraction of a millisecond.

### Phase 5: The LLM (`llm.py`)
Now that Endee has retrieved the top 3 most relevant academic papers, we build a prompt. We combine standard instructions, the retrieved papers, and the user's question, and send it to a local LLaMA 3.1 model running via Ollama.

```text
You are an AI research assistant. Use ONLY the provided paper abstracts to answer...

[Retrieved Paper 1 Context...]
[Retrieved Paper 2 Context...]

User Question: How to reduce AI hallucinations?
```
The LLM reads the papers and synthesizes a final, highly accurate answer based *only* on the retrieved academic literature.

---

## 4. Key Takeaways for Engineers

1. **Separation of Concerns:** The embedding model (Sentence Transformers), the Storage/Search Engine (Endee), and the Generator (LLaMA) are completely decoupled. You can hot-swap any of them as better models release.
2. **Speed vs Memory:** Endee's `INT8` precision quantization shows how systems programming concepts (data representation) directly impact AI application scalability.
3. **Data is King:** An LLM is only as good as the context you provide it. A highly tuned vector database ensures the LLM gets the most factual, relevant context possible.

By combining the retrieval power of the Endee Vector Database with the generative capabilities of LLaMA, you have built an enterprise-grade, offline-first AI research assistant!
