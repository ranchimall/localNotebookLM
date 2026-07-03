"""
Query your Keep notes RAG index.

Usage:
    python query.py "what was that recipe with lentils I saved?"
    python query.py "gift ideas for mom" --top-k 8 --no-llm
"""

import argparse
import os

import chromadb
from chromadb.utils import embedding_functions


def retrieve(collection, query, n_results=5, label_filter=None):
    where = {"labels": {"$eq": label_filter}} if label_filter else None
    return collection.query(query_texts=[query], n_results=n_results, where=where)


def build_context(results, show_full=False):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    blocks = []
    for doc, meta, dist in zip(docs, metas, dists):
        similarity = 1 - dist
        header = f"[{meta.get('title', '(untitled)')}] (similarity={similarity:.3f}, source={meta.get('source_file', '?')})"
        # Use the full original note if requested/available, otherwise
        # fall back to just the matched chunk.
        text = meta.get("full_text", doc) if show_full else doc
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def answer_with_claude(question, context):
    """Optional generation step. Returns None if the anthropic package or
    API key isn't available, so query.py still works retrieval-only."""
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are answering a question using only the person's own Google Keep notes below. "
        "If the notes don't contain the answer, say so plainly rather than guessing.\n\n"
        f"NOTES:\n{context}\n\nQUESTION: {question}"
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def main():
    parser = argparse.ArgumentParser(description="Query your Keep notes RAG index")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--collection", default="keep_notes")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--label", default=None, help="Filter to notes with this exact label string")
    parser.add_argument("--no-llm", action="store_true", help="Only show retrieved notes, skip LLM answer")
    parser.add_argument("--full", action="store_true", help="Show the full original note instead of just the matched chunk")
    args = parser.parse_args()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model)
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(name=args.collection, embedding_function=embed_fn)

    results = retrieve(collection, args.query, n_results=args.top_k, label_filter=args.label)
    context = build_context(results, show_full=args.full)

    print("=== Retrieved notes ===\n")
    print(context)
    print("\n========================\n")

    if args.no_llm:
        return

    answer = answer_with_claude(args.query, context)
    if answer:
        print("=== Answer ===\n")
        print(answer)
    else:
        print("(Set ANTHROPIC_API_KEY and `pip install anthropic` to get a generated "
              "answer here. Showing retrieved notes only for now.)")


if __name__ == "__main__":
    main()
