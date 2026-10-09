#!/usr/bin/env python3
"""
build_glossary.py — Offline glossary builder for the RAG pipeline.

Two stages with a HUMAN REVIEW GATE in between:

  Stage 1 (extract):  manual text  ->  candidate_terms.txt   [you review this]
  Stage 2 (aliases):  reviewed terms -> glossary.json        [you spot-check this]

Run stage 1, open candidate_terms.txt, delete the junk / fix the terms,
then run stage 2. Both stages are resumable: re-running skips work already done.

This is OFFLINE batch tooling — slow is fine. Use a bigger/better model here
than you use at runtime if you have one; the output is reviewed by a human
before it ever reaches a user, so the runtime reliability concerns don't apply.

Usage:
    python build_glossary.py extract
    # ... review data/candidate_terms.txt by hand ...
    python build_glossary.py aliases
"""

import argparse
import json
import os
import sys
import requests

from config import OLLAMA_URL, OLLAMA_MODEL, DATA_DIR, MARKDOWN_FOLDER

TERMS_FILE    = DATA_DIR / "candidate_terms.txt"
GLOSSARY_FILE = DATA_DIR / "glossary.json"


def _ask_llm(prompt: str, num_ctx: int = 2048, temperature: float = 0.4) -> str:
    """Single blocking call to Ollama. think=True is fine here — it's offline."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "think":   True,
            "options": {"num_ctx": num_ctx, "temperature": temperature, "num_gpu": 99},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _read_manual_text(manuals_dir: str) -> list[str]:
    """
    Load plain-text chunks from a directory of .txt/.md files.
    Defaults to data/markdown_temp/ which already contains extracted manual text.
    """
    chunks = []
    for root, _, files in os.walk(manuals_dir):
        for fn in files:
            if fn.lower().endswith((".txt", ".md")):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    text = f.read()
                # ~1500-char windows so a small model can attend to the whole thing
                for i in range(0, len(text), 1500):
                    piece = text[i:i + 1500].strip()
                    if len(piece) > 100:
                        chunks.append(piece)
    return chunks


# --------------------------------------------------------------------------- #
# STAGE 1: extract candidate technical terms                                  #
# --------------------------------------------------------------------------- #
def extract_terms(manuals_dir: str) -> None:
    chunks = _read_manual_text(manuals_dir)
    if not chunks:
        sys.exit(f"No .txt/.md text found under {manuals_dir!r}. Export your manuals first.")

    print(f"Extracting terms from {len(chunks)} text windows...")
    found: set[str] = set()

    for n, chunk in enumerate(chunks, 1):
        prompt = (
            "Extract the technical terms and named procedures, components, faults, "
            "or maintenance actions from the following machine-manual text. "
            "Return ONLY a list, one term per line, no numbering, no commentary. "
            "Use the exact wording from the text. Skip generic words.\n\n"
            f"TEXT:\n{chunk}\n"
        )
        try:
            out = _ask_llm(prompt)
        except Exception as e:
            print(f"  [chunk {n}] skipped: {e}")
            continue

        for line in out.splitlines():
            term = line.strip(" -*\t").strip()
            # keep multi-word-ish terms, drop one-letter noise and over-long lines
            if 3 <= len(term) <= 80 and not term.lower().startswith(("here", "the term")):
                found.add(term)
        print(f"  [chunk {n}/{len(chunks)}] running total: {len(found)} terms", end="\r")

    terms = sorted(found, key=str.lower)
    with open(TERMS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(terms) + "\n")

    print(f"\n\nWrote {len(terms)} candidate terms to {TERMS_FILE}")
    print(f"==> REVIEW IT BY HAND: delete junk, fix wording, then run: "
          f"python build_glossary.py aliases")


# --------------------------------------------------------------------------- #
# STAGE 2: generate colloquial aliases for the reviewed terms                 #
# --------------------------------------------------------------------------- #
def _existing_canonicals() -> set[str]:
    """For resumability: don't regenerate aliases we already have."""
    if not os.path.exists(GLOSSARY_FILE):
        return set()
    try:
        with open(GLOSSARY_FILE, encoding="utf-8") as f:
            return {e["canonical"] for e in json.load(f)}
    except Exception:
        return set()


def generate_aliases() -> None:
    if not os.path.exists(TERMS_FILE):
        sys.exit(f"{TERMS_FILE} not found. Run the 'extract' stage first.")

    with open(TERMS_FILE, encoding="utf-8") as f:
        terms = [t.strip() for t in f if t.strip()]

    glossary = []
    if os.path.exists(GLOSSARY_FILE):
        with open(GLOSSARY_FILE, encoding="utf-8") as f:
            glossary = json.load(f)
    done = _existing_canonicals()

    print(f"Generating aliases for {len(terms)} terms ({len(done)} already done)...")

    for n, term in enumerate(terms, 1):
        if term in done:
            continue

        prompt = (
            f'A machine manual uses the technical term: "{term}".\n'
            "List 5-8 everyday, non-technical ways a machine operator might "
            "describe this same thing or problem in plain language — the kind "
            "of phrasing someone would type into a search box. "
            "One per line, no numbering, no explanations. "
            "If the term is a pure code or part number, just repeat it once.\n"
        )
        try:
            out = _ask_llm(prompt, num_ctx=1024, temperature=0.6)
        except Exception as e:
            print(f"  [{n}] {term!r}: FAILED ({e})")
            continue

        aliases = []
        for line in out.splitlines():
            a = line.strip(" -*\t\"'").strip()
            if 2 <= len(a) <= 60 and a.lower() != term.lower():
                aliases.append(a)
        aliases = list(dict.fromkeys(aliases))  # dedup, keep order

        if aliases:
            glossary.append({"canonical": term, "aliases": aliases})
            # write after EVERY term so a crash never loses progress
            with open(GLOSSARY_FILE, "w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
        print(f"  [{n}/{len(terms)}] {term!r}: {len(aliases)} aliases")

    print(f"\nDone. {len(glossary)} entries in {GLOSSARY_FILE}")
    print("==> SPOT-CHECK IT: open the file and delete any nonsense aliases.")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline glossary builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Stage 1: pull candidate terms from manuals")
    pe.add_argument(
        "--manuals",
        default=str(MARKDOWN_FOLDER),
        help=f"Directory of .txt/.md manual text (default: {MARKDOWN_FOLDER})",
    )

    sub.add_parser("aliases", help="Stage 2: generate colloquial aliases for reviewed terms")

    args = p.parse_args()
    if args.cmd == "extract":
        extract_terms(args.manuals)
    elif args.cmd == "aliases":
        generate_aliases()


if __name__ == "__main__":
    main()
