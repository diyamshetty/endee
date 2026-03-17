"""
ingest.py — Fetch 2,000 recent ArXiv papers on "Large Language Models"
and save them to papers.json for caching.
"""
import json
import os
import time
import arxiv

PAPERS_FILE = os.path.join(os.path.dirname(__file__), "papers.json")
QUERY = "Large Language Models"
MAX_RESULTS = 2000


def fetch_papers(max_results: int = MAX_RESULTS, force: bool = False) -> list[dict]:
    """
    Fetch papers from ArXiv and cache to papers.json.
    Returns a list of paper dicts with keys:
      id, title, authors, abstract, published, url
    """
    if not force and os.path.exists(PAPERS_FILE):
        print(f"[ingest] Loading cached papers from {PAPERS_FILE} ...")
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"[ingest] Loaded {len(papers)} papers from cache.")
        return papers

    print(f"[ingest] Fetching up to {max_results} papers from ArXiv for query: '{QUERY}' ...")
    search = arxiv.Search(
        query=QUERY,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    for i, result in enumerate(client.results(search)):
        paper = {
            "id": result.entry_id.split("/")[-1],   # e.g. "2401.12345v1"
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors[:5]],  # first 5 authors
            "abstract": result.summary.strip().replace("\n", " "),
            "published": result.published.strftime("%Y-%m-%d") if result.published else "",
            "url": result.entry_id,
        }
        papers.append(paper)

        if (i + 1) % 100 == 0:
            print(f"[ingest]   Fetched {i + 1} papers so far ...")

    print(f"[ingest] Done. Total papers fetched: {len(papers)}")
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"[ingest] Papers saved to {PAPERS_FILE}")
    return papers


if __name__ == "__main__":
    start = time.time()
    papers = fetch_papers(force=False)
    elapsed = time.time() - start
    print(f"\nSample paper:")
    if papers:
        p = papers[0]
        print(f"  ID      : {p['id']}")
        print(f"  Title   : {p['title']}")
        print(f"  Authors : {', '.join(p['authors'])}")
        print(f"  Date    : {p['published']}")
        print(f"  Abstract: {p['abstract'][:200]}...")
    print(f"\nTotal time: {elapsed:.1f}s")
