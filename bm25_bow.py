#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm25_bow.py
===========
Method A: build a BM25-weighted bag-of-words (document-term matrix) over a corpus
and export it as JSON so a retriever can read it and perform query rewriting /
query expansion.

Design notes
------------
* Each (term, document) cell gets the BM25 term weight:

      w(t,d) = IDF(t) * [ f(t,d) * (k1 + 1) ]
                        ----------------------------------------
                        f(t,d) + k1 * (1 - b + b * |d| / avgdl)

* IDF uses the Lucene / non-negative variant:

      IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

  This is ALWAYS >= 0. The classic Robertson-Sparck-Jones IDF can go negative
  for terms appearing in >~50% of documents, which corrupts BoW feature vectors
  (a documented failure mode). We avoid it by default.

* Output is SPARSE: per document we store only the terms actually present
  (optionally the top-K by weight), never the full dense matrix.

Zero third-party dependencies (standard library only) for portability.

Usage
-----
    python bm25_bow.py
    python bm25_bow.py --input ./some_other_dir --output bow.json --top-k 25

Input formats
-------------
* A directory : every *.txt and *.md file becomes one document (filename stem = id)
* .jsonl      : one JSON object per line, e.g. {"id": "doc1", "text": "...", "title": "..."}
* .json       : a list of such objects
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR, MARKDOWN_FOLDER

# --------------------------------------------------------------------------- #
# Stopwords (compact, extend or override with --stopwords FILE)
# --------------------------------------------------------------------------- #
STOPWORDS_EN = frozenset("""
a an and are as at be but by for from had has have he her his i in into is it
its of on or she that the their them then there these they this to was were
what when where which who will with would you your our we us not no can could
""".split())

STOPWORDS_EL = frozenset("""
και η ή αλλά αν ως προς στη στην στον στους στις σε από για με μετά που παρά
ο η το οι τα του της των τον την ένα μια μία έναν είναι ήταν να θα δεν μη μην
που πώς ποιος ποια ποιο αυτός αυτή αυτό αυτά αυτές αυτοί μας σας τους έχει
πώς εντώ κατά επί πως
""".split())

# Token = sequence of Unicode letters/digits (keeps "bm25", "iso9001").
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_corpus(path: Optional[str], id_field: str, text_field: str,
                title_field: str) -> List[Dict]:
    """Return a list of {'id', 'text', 'title'} dicts."""
    if path is None:
        path = str(MARKDOWN_FOLDER)

    if os.path.isdir(path):
        docs = []
        for name in sorted(os.listdir(path)):
            if name.lower().endswith((".txt", ".md")):
                full = os.path.join(path, name)
                with open(full, encoding="utf-8") as fh:
                    docs.append({"id": os.path.splitext(name)[0],
                                 "text": fh.read(), "title": ""})
        return docs

    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:  # .json (a list)
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        if isinstance(rows, dict):
            rows = [rows]

    docs = []
    for i, row in enumerate(rows):
        docs.append({
            "id": str(row.get(id_field, i)),
            "text": str(row.get(text_field, "")),
            "title": str(row.get(title_field, "")),
        })
    return docs


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def tokenize(text: str, stopwords: frozenset, min_len: int,
             drop_numeric: bool) -> List[str]:
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if len(tok) < min_len:
            continue
        if drop_numeric and tok.isdigit():
            continue
        if tok in stopwords:
            continue
        tokens.append(tok)
    return tokens


# --------------------------------------------------------------------------- #
# BM25 statistics
# --------------------------------------------------------------------------- #
def bm25_idf(df: int, n_docs: int) -> float:
    """Non-negative (Lucene) IDF."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_weight(tf: int, idf: float, doc_len: int, avgdl: float,
                k1: float, b: float) -> float:
    denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
    return idf * (tf * (k1 + 1.0)) / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Core: build the BM25-weighted bag of words
# --------------------------------------------------------------------------- #
def build_bow(docs: List[Dict], stopwords: frozenset, k1: float, b: float,
              min_df: int, min_len: int, drop_numeric: bool,
              top_k: int) -> Dict:
    # 1. tokenize + per-doc term frequencies
    tokenized: List[Tuple[str, str, Counter]] = []  # (id, title, tf_counter)
    df_counter: Counter = Counter()   # document frequency (in how many docs)
    cf_counter: Counter = Counter()   # collection frequency (total occurrences)
    for d in docs:
        toks = tokenize(d["text"], stopwords, min_len, drop_numeric)
        tf = Counter(toks)
        tokenized.append((d["id"], d.get("title", ""), tf))
        df_counter.update(tf.keys())
        cf_counter.update(tf)

    n_docs = len(tokenized)
    if n_docs == 0:
        raise ValueError("Empty corpus after preprocessing.")

    # 2. df threshold -> vocabulary
    vocab_df = {t: df for t, df in df_counter.items() if df >= min_df}
    if not vocab_df:
        raise ValueError(
            f"Vocabulary empty with min_df={min_df}; lower --min-df.")

    # 3. avgdl over kept terms only (consistent with the matrix we build)
    doc_lengths = []
    for _id, _title, tf in tokenized:
        doc_lengths.append(sum(c for t, c in tf.items() if t in vocab_df))
    avgdl = sum(doc_lengths) / n_docs if n_docs else 0.0
    avgdl = avgdl or 1.0

    # 4. idf per vocab term
    idf = {t: bm25_idf(df, n_docs) for t, df in vocab_df.items()}

    # 5. weighted BoW per document (sparse)
    documents = {}
    for (doc_id, title, tf), dl in zip(tokenized, doc_lengths):
        row = []
        for term, count in tf.items():
            if term not in vocab_df:
                continue
            w = bm25_weight(count, idf[term], dl, avgdl, k1, b)
            row.append({"term": term, "weight": round(w, 6), "tf": count})
        row.sort(key=lambda x: x["weight"], reverse=True)
        if top_k and top_k > 0:
            row = row[:top_k]
        documents[doc_id] = {
            "length": dl,
            "title": title,
            "terms": row,
        }

    # 6. corpus-level expansion vocabulary: most discriminative terms
    #    (rank by idf; df>=min_df already guarantees they are not hapax noise)
    corpus_top = sorted(
        ({"term": t, "idf": round(idf[t], 6), "df": vocab_df[t]}
         for t in vocab_df),
        key=lambda x: x["idf"], reverse=True,
    )

    return {
        "meta": {
            "num_documents": n_docs,
            "vocabulary_size": len(vocab_df),
            "avgdl": round(avgdl, 4),
            "bm25_params": {"k1": k1, "b": b, "idf_variant": "lucene_nonneg"},
            "min_df": min_df,
            "top_k_per_doc": top_k,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "vocabulary": {t: {"df": vocab_df[t], "cf": cf_counter[t],
                           "idf": round(idf[t], 6)}
                       for t in vocab_df},
        "documents": documents,
        "corpus_top_terms": corpus_top,
    }


# --------------------------------------------------------------------------- #
# Example consumer: how the retriever would use the JSON for query rewrite
# --------------------------------------------------------------------------- #
def rewrite_query(query: str, bow: Dict, n_expansions: int = 5) -> str:
    """
    Minimal demonstration of query expansion driven by the BM25 BoW.

    Strategy: score every document by summing the BoW weights of the terms it
    shares with the query, take the best-matching document, and append its
    highest-weighted (non-query) terms as expansion terms. This is the classic
    pseudo-relevance-feedback idea, but the term weights come from the
    BM25-weighted bag of words instead of raw TF.
    """
    q_terms = set(TOKEN_RE.findall(query.lower()))

    best_id, best_score = None, -1.0
    for doc_id, doc in bow["documents"].items():
        score = sum(t["weight"] for t in doc["terms"] if t["term"] in q_terms)
        if score > best_score:
            best_id, best_score = doc_id, score

    if best_id is None or best_score <= 0:
        return query  # no lexical anchor; leave the query untouched

    expansions = [t["term"] for t in bow["documents"][best_id]["terms"]
                  if t["term"] not in q_terms][:n_expansions]
    return query + " " + " ".join(expansions) if expansions else query


# --------------------------------------------------------------------------- #
# Visualisation (matplotlib imported lazily so the core stays dependency-free)
# --------------------------------------------------------------------------- #
def make_plots(bow: Dict, out_dir: str, top_n: int = 25) -> List[str]:
    """Write three figures: raw frequency, BM25 weight, frequency distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed -> skipping "
              "(pip install matplotlib)")
        return []

    os.makedirs(out_dir, exist_ok=True)
    vocab = bow["vocabulary"]
    paths: List[str] = []

    def _barh(labels, values, xlabel, title, fname, color):
        labels, values = labels[::-1], values[::-1]
        h = max(3.0, 0.32 * len(labels))
        fig, ax = plt.subplots(figsize=(8, h))
        ax.barh(range(len(labels)), values, color=color)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.margins(y=0.01)
        fig.tight_layout()
        p = os.path.join(out_dir, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # 1) top-N by raw collection frequency
    top_cf = sorted(vocab.items(), key=lambda kv: kv[1]["cf"],
                    reverse=True)[:top_n]
    _barh([t for t, _ in top_cf], [v["cf"] for _, v in top_cf],
          "Collection frequency (raw counts)",
          f"Top {len(top_cf)} terms by raw frequency",
          "01_top_terms_frequency.png", "#4C72B0")

    # 2) top-N by aggregated BM25 weight (mass = sum of weights across docs)
    bm25_mass: Counter = Counter()
    for doc in bow["documents"].values():
        for t in doc["terms"]:
            bm25_mass[t["term"]] += t["weight"]
    top_bm = bm25_mass.most_common(top_n)
    _barh([t for t, _ in top_bm], [round(w, 3) for _, w in top_bm],
          "Aggregated BM25 weight (sum over documents)",
          f"Top {len(top_bm)} terms by BM25 weight",
          "02_top_terms_bm25.png", "#C44E52")

    # 3) frequency distribution histogram (the long tail that motivates IDF)
    cf_values = [v["cf"] for v in vocab.values()]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cf_values, bins=min(40, max(5, len(set(cf_values)))),
            color="#55A868", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Collection frequency of a term")
    ax.set_ylabel("Number of terms (log scale)")
    ax.set_title("Term frequency distribution (long tail)")
    fig.tight_layout()
    p = os.path.join(out_dir, "03_frequency_distribution.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    print(f"[plot] wrote {len(paths)} figure(s) to {out_dir}/")
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_stopwords(lang: str, extra_path: Optional[str]) -> frozenset:
    base = set()
    if lang in ("en", "both"):
        base |= STOPWORDS_EN
    if lang in ("el", "both"):
        base |= STOPWORDS_EL
    if extra_path:
        with open(extra_path, encoding="utf-8") as fh:
            base |= {w.strip().lower() for w in fh if w.strip()}
    return frozenset(base)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(MARKDOWN_FOLDER),
                   help=f"corpus.jsonl | corpus.json | directory of .txt/.md "
                        f"(default: {MARKDOWN_FOLDER})")
    p.add_argument("--output", default=str(DATA_DIR / "bow.json"),
                   help=f"output JSON path (default: {DATA_DIR / 'bow.json'})")
    p.add_argument("--id-field", default="id")
    p.add_argument("--text-field", default="text")
    p.add_argument("--title-field", default="title")
    p.add_argument("--lang", choices=["en", "el", "both", "none"], default="en",
                   help="built-in stopword set to use")
    p.add_argument("--stopwords", help="extra stopword file (one word per line)")
    p.add_argument("--k1", type=float, default=1.5, help="BM25 k1 (TF saturation)")
    p.add_argument("--b", type=float, default=0.75, help="BM25 b (length norm)")
    p.add_argument("--min-df", type=int, default=1,
                   help="drop terms appearing in fewer than this many docs")
    p.add_argument("--min-len", type=int, default=2, help="min token length")
    p.add_argument("--keep-numeric", action="store_true",
                   help="keep purely numeric tokens (e.g. '2016')")
    p.add_argument("--top-k", type=int, default=0,
                   help="keep only top-K terms per doc (0 = keep all present)")
    p.add_argument("--demo-rewrite", metavar="QUERY",
                   help="after building, print an example rewritten query")
    p.add_argument("--plot-dir", metavar="DIR",
                   help="also write frequency/BM25 figures to DIR (needs matplotlib)")
    p.add_argument("--plot-top-n", type=int, default=25,
                   help="number of terms shown in the bar charts")
    args = p.parse_args(argv)

    lang = "none" if args.lang == "none" else args.lang
    stopwords = frozenset() if lang == "none" else build_stopwords(lang, args.stopwords)

    docs = load_corpus(args.input, args.id_field, args.text_field, args.title_field)
    if not docs:
        print(f"No documents found under {args.input!r}. Make sure manuals have been preprocessed first.")
        return 1

    bow = build_bow(
        docs, stopwords=stopwords, k1=args.k1, b=args.b, min_df=args.min_df,
        min_len=args.min_len, drop_numeric=not args.keep_numeric, top_k=args.top_k,
    )

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(bow, fh, ensure_ascii=False, indent=2)

    m = bow["meta"]
    print(f"[ok] {m['num_documents']} docs, vocab={m['vocabulary_size']}, "
          f"avgdl={m['avgdl']} -> {args.output}")

    if args.plot_dir:
        make_plots(bow, args.plot_dir, top_n=args.plot_top_n)

    if args.demo_rewrite:
        rewritten = rewrite_query(args.demo_rewrite, bow)
        print(f"[rewrite] {args.demo_rewrite!r}  ->  {rewritten!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
