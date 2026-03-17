"""
llm.py — Local LLM integration via Ollama API.
Sends RAG-augmented prompts to LLaMA 3.1 8B Instruct running locally.
"""
import json
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b-instruct"
TIMEOUT = 120   # seconds


RAG_PROMPT_TEMPLATE = """\
You are a helpful AI research assistant specializing in machine learning and AI.
Use ONLY the provided paper abstracts to answer the question. \
If the abstracts do not contain sufficient information, say so.

=== Retrieved ArXiv Paper Abstracts ===
{context}
=== End of Abstracts ===

User Question: {question}

Answer (based solely on the provided abstracts):"""


def format_context(retrieved_papers: list[dict]) -> str:
    """
    Format retrieved papers into a numbered context string for the prompt.
    """
    parts = []
    for i, paper in enumerate(retrieved_papers, 1):
        authors = paper.get("authors", "Unknown")
        published = paper.get("published", "")
        title = paper.get("title", "Unknown Title")
        abstract = paper.get("abstract", "")
        score = paper.get("score", 0.0)

        parts.append(
            f"[{i}] \"{title}\"\n"
            f"    Authors  : {authors}\n"
            f"    Published: {published}  |  Similarity Score: {score:.3f}\n"
            f"    Abstract : {abstract[:800]}{'...' if len(abstract) > 800 else ''}"
        )
    return "\n\n".join(parts)


def generate_answer(question: str, retrieved_papers: list[dict]) -> str:
    """
    Generate an answer using LLaMA 3.1 via Ollama, given a question and
    retrieved paper context.

    Args:
        question         : The user's natural-language question.
        retrieved_papers : List of result dicts from vector_store.search().

    Returns:
        The model's full response as a string.
    """
    context = format_context(retrieved_papers)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
        },
    }

    response_text = ""
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response_text += chunk.get("response", "")
                if chunk.get("done", False):
                    break
    except requests.exceptions.ConnectionError:
        return (
            "❌ **Could not connect to Ollama.**\n\n"
            f"Make sure Ollama is running at `{OLLAMA_URL.replace('/api/generate', '')}` "
            f"and the model `{MODEL_NAME}` is pulled:\n"
            f"```\nollama pull {MODEL_NAME}\n```"
        )
    except requests.exceptions.Timeout:
        return (
            "❌ **Ollama request timed out.**\n\n"
            "The model may be loading or your hardware is too slow. Please try again."
        )
    except requests.exceptions.HTTPError as e:
        return f"❌ **Ollama HTTP error:** {e}"

    return response_text.strip() or "⚠️ The model returned an empty response."


if __name__ == "__main__":
    print("=== LLM self-test ===")
    print(f"Sending test prompt to {MODEL_NAME} via Ollama ...")

    dummy_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "published": "2017-06-12",
            "abstract": (
                "We propose a new simple network architecture, the Transformer, based solely on "
                "attention mechanisms, dispensing with recurrence and convolutions entirely."
            ),
            "score": 0.95,
        }
    ]

    start = time.time()
    answer = generate_answer(
        "What is the Transformer architecture?",
        dummy_papers,
    )
    elapsed = time.time() - start

    print(f"\nAnswer (took {elapsed:.1f}s):\n{answer}")
    print("\nSelf-test complete.")
