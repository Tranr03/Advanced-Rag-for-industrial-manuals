#!/usr/bin/env python3
"""
inspect_chunks.py — Query ChromaDB for all chunks of a specific manual.

Usage:
    python inspect_chunks.py
    python inspect_chunks.py --source "some_other_manual_stem"
    python inspect_chunks.py --type text
    python inspect_chunks.py --type image_description
    python inspect_chunks.py --save          # also writes results to data/inspect_output.json
"""

import argparse
import json
import chromadb

from config import VECTOR_DB_PATH

DEFAULT_SOURCE = "B-85444EN-1_03_02_Robodrill_Operator_Manual"


def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB chunks for a manual")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="PDF stem (filename without .pdf)")
    parser.add_argument("--type", dest="chunk_type", default=None,
                        choices=["text", "image_description", "vision_skipped"],
                        help="Filter by chunk type (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to data/inspect_output.json")
    args = parser.parse_args()

    client     = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    col_names  = [c.name for c in client.list_collections()]
    if not col_names:
        print("No collections found in ChromaDB.")
        return

    # Use the first (and usually only) collection
    col = client.get_collection(col_names[0])
    print(f"Collection: '{col_names[0]}'  —  {col.count()} total documents\n")

    # Build where filter
    if args.chunk_type:
        where = {"$and": [
            {"source": {"$eq": args.source}},
            {"type":   {"$eq": args.chunk_type}},
        ]}
    else:
        where = {"source": {"$eq": args.source}}

    results = col.get(
        where=where,
        include=["documents", "metadatas"],
    )

    ids       = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"]

    if not ids:
        print(f"No chunks found for source='{args.source}'"
              + (f", type='{args.chunk_type}'" if args.chunk_type else "") + ".")
        return

    # Group by type for the summary
    by_type: dict[str, int] = {}
    for m in metadatas:
        t = m.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Source : {args.source}")
    print(f"Found  : {len(ids)} chunks")
    for t, n in sorted(by_type.items()):
        print(f"         {n:>4}  {t}")
    print()

    # Sort by page then chunk id for readable output
    rows = sorted(
        zip(ids, documents, metadatas),
        key=lambda x: (x[2].get("page", 0), x[0])
    )

    for chunk_id, doc, meta in rows:
        chunk_type = meta.get("type", "unknown")
        page       = meta.get("page", "?")

        if chunk_type == "vision_skipped":
            print(f"[p.{page}]  {chunk_id}  →  VISION SKIPPED")
            continue

        label = "IMG" if chunk_type == "image_description" else "TXT"
        print(f"[p.{page}] [{label}]  {chunk_id}")

        if chunk_type == "image_description" and meta.get("image_path"):
            print(f"         image: {meta['image_path']}")

        # Print preview (first 400 chars)
        preview = doc.replace("\n", " ").strip()
        print(f"         {preview[:400]}" + ("..." if len(preview) > 400 else ""))
        print()

    if args.save:
        out_path = VECTOR_DB_PATH.parent / "inspect_output.json"
        payload  = [
            {"id": cid, "document": doc, "metadata": meta}
            for cid, doc, meta in rows
        ]
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(payload)} chunks to {out_path}")


if __name__ == "__main__":
    main()
