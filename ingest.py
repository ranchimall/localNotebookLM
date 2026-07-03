"""
Ingest Google Keep Takeout JSON notes into a local, persistent ChromaDB
collection, using a sentence-transformers model for embeddings.

Usage:
    python ingest.py --input-dir "/path/to/Takeout/Keep"
"""

import argparse
import glob
import hashlib
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document

import pytesseract
from PIL import Image
# Only needed on Windows if tesseract.exe isn't on PATH — point this at
# your actual install location.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def parse_keep_note(path):
    """Turn one Keep Takeout JSON file into a dict with text + metadata,
    or return None if the note should be skipped (trashed / empty)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("isTrashed"):
        return None

    title = (data.get("title") or "").strip()
    body_parts = []

    if data.get("textContent"):
        body_parts.append(data["textContent"])
    elif data.get("listContent"):
        # Checklist-style note
        for item in data["listContent"]:
            text = item.get("text", "")
            checked = item.get("isChecked", False)
            prefix = "[x]" if checked else "[ ]"
            if text:
                body_parts.append(f"{prefix} {text}")

    body = "\n".join(body_parts).strip()
    if not title and not body:
        return None

    full_text = f"{title}\n\n{body}".strip() if title else body

    labels = [l.get("name", "") for l in data.get("labels", []) if l.get("name")]
    note_id = hashlib.md5(path.encode()).hexdigest()[:12]

    metadata = {
        "title": title or "(untitled)",
        "labels": ", ".join(labels),
        "pinned": bool(data.get("isPinned", False)),
        "archived": bool(data.get("isArchived", False)),
        "color": data.get("color", ""),
        "created_usec": data.get("createdTimestampUsec", 0),
        "source_file": os.path.basename(path),
    }

    return {"id": note_id, "text": full_text, "metadata": metadata}

def parse_pdf(path):
    """Turn one PDF file into a dict with text + metadata, mirroring the
    shape parse_keep_note() returns, or return None if it has no
    extractable text."""
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"  ! Could not open {path}: {e}")
        return None

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    body = "\n\n".join(pages).strip()
    if not body:
        return None

    title = os.path.splitext(os.path.basename(path))[0]
    full_text = f"{title}\n\n{body}".strip()
    note_id = hashlib.md5(path.encode()).hexdigest()[:12]

    metadata = {
        "title": title,
        "labels": "",
        "pinned": False,
        "archived": False,
        "color": "",
        "created_usec": 0,
        "source_file": os.path.basename(path),
        "page_count": len(reader.pages),
    }

    return {"id": note_id, "text": full_text, "metadata": metadata}  

def parse_docx(path):
    """Turn one Word .docx file into a dict with text + metadata, mirroring
    the shape parse_keep_note() returns, or return None if it has no
    extractable text."""
    try:
        doc = Document(path)
    except Exception as e:
        print(f"  ! Could not open {path}: {e}")
        return None

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    body = "\n\n".join(paragraphs).strip()
    if not body:
        return None

    title = os.path.splitext(os.path.basename(path))[0]
    full_text = f"{title}\n\n{body}".strip()
    note_id = hashlib.md5(path.encode()).hexdigest()[:12]

    metadata = {
        "title": title,
        "labels": "",
        "pinned": False,
        "archived": False,
        "color": "",
        "created_usec": 0,
        "source_file": os.path.basename(path),
    }

    return {"id": note_id, "text": full_text, "metadata": metadata}


def parse_txt(path):
    """Turn one plain-text .txt file into a dict with text + metadata,
    mirroring the shape parse_keep_note() returns, or return None if it's
    empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            body = f.read().strip()
    except UnicodeDecodeError as e:
        print(f"  ! Could not read {path}: {e}")
        return None

    if not body:
        return None

    title = os.path.splitext(os.path.basename(path))[0]
    full_text = f"{title}\n\n{body}".strip()
    note_id = hashlib.md5(path.encode()).hexdigest()[:12]

    metadata = {
        "title": title,
        "labels": "",
        "pinned": False,
        "archived": False,
        "color": "",
        "created_usec": 0,
        "source_file": os.path.basename(path),
    }

    return {"id": note_id, "text": full_text, "metadata": metadata}      

def parse_image(path):
    """Run OCR on an image file and turn any extracted text into a dict
    with text + metadata, mirroring parse_keep_note(). Returns None if no
    text was found (e.g. a photo with no writing in it)."""
    try:
        img = Image.open(path)
        body = pytesseract.image_to_string(img).strip()
    except Exception as e:
        print(f"  ! Could not OCR {path}: {e}")
        return None

    if not body:
        return None

    title = os.path.splitext(os.path.basename(path))[0]
    full_text = f"{title}\n\n{body}".strip()
    note_id = hashlib.md5(path.encode()).hexdigest()[:12]

    metadata = {
        "title": title,
        "labels": "",
        "pinned": False,
        "archived": False,
        "color": "",
        "created_usec": 0,
        "source_file": os.path.basename(path),
    }

    return {"id": note_id, "text": full_text, "metadata": metadata}    


def chunk_text(text, max_chars=800, overlap=100):
    """Most Keep notes are short and won't need splitting, but long ones
    (e.g. pasted articles) get chunked with a bit of overlap."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest Google Keep Takeout JSON notes into ChromaDB")
    parser.add_argument("--input-dir", required=True, help="Folder with Keep .json files (Takeout/Keep)")
    parser.add_argument("--db-path", default="./chroma_db", help="Where to persist the Chroma database")
    parser.add_argument("--collection", default="keep_notes", help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--max-chars", type=int, default=800, help="Max characters per chunk")
    parser.add_argument("--include-archived", action="store_true", help="Include archived notes")
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    pdf_files = glob.glob(os.path.join(args.input_dir, "*.pdf"))
    docx_files = glob.glob(os.path.join(args.input_dir, "*.docx"))
    txt_files = glob.glob(os.path.join(args.input_dir, "*.txt"))

    image_files = (
        glob.glob(os.path.join(args.input_dir, "*.png"))
        + glob.glob(os.path.join(args.input_dir, "*.jpg"))
        + glob.glob(os.path.join(args.input_dir, "*.jpeg"))
    )
    print(f"Found {len(json_files)} JSON, {len(pdf_files)} PDF, {len(docx_files)} DOCX, "
          f"{len(txt_files)} TXT, and {len(image_files)} image files in {args.input_dir}")

    notes = []
    skipped = 0
    for path in json_files:
        try:
            note = parse_keep_note(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  ! Skipping {path}: {e}")
            skipped += 1
            continue
        if note is None:
            skipped += 1
            continue
        if note["metadata"]["archived"] and not args.include_archived:
            skipped += 1
            continue
        notes.append(note)

    for path in pdf_files:
        note = parse_pdf(path)
        if note is None:
            skipped += 1
            continue
        notes.append(note)

    for path in docx_files:
        note = parse_docx(path)
        if note is None:
            skipped += 1
            continue
        notes.append(note)

    for path in txt_files:
        note = parse_txt(path)
        if note is None:
            skipped += 1
            continue
        notes.append(note)    

    for path in image_files:
        note = parse_image(path)
        if note is None:
            skipped += 1
            continue
        notes.append(note)    

    print(f"Parsed {len(notes)} usable notes ({skipped} skipped: trashed/empty/archived)")

    if not notes:
        print("Nothing to ingest. Exiting.")
        return

    ids, documents, metadatas = [], [], []
    for note in notes:
        chunks = chunk_text(note["text"], max_chars=args.max_chars)
        for i, chunk in enumerate(chunks):
            ids.append(f"{note['id']}_{i}")
            documents.append(chunk)
            meta = dict(note["metadata"])
            meta["chunk_index"] = i
            meta["chunk_count"] = len(chunks)
            # Keep the full original note text too, so a search hit on one
            # chunk can still show the whole note, not just that fragment.
            meta["full_text"] = note["text"]
            metadatas.append(meta)

    print(f"Created {len(documents)} chunks. Loading embedding model '{args.model}' "
          f"(first run downloads it, ~80MB for MiniLM)...")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.model)

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection, embedding_function=embed_fn)

    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Upserted {min(end, len(documents))}/{len(documents)}")

    print(f"\nDone. Collection '{args.collection}' now has {collection.count()} chunks "
          f"stored at {args.db_path}")


if __name__ == "__main__":
    main()
