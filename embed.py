"""Embed every raw review and project to 2D for the map view.

Uses a local multilingual model, so Korean and English text share one space and no API is needed.
Output: out/embeddings.json with 2D points (PCA) and the most similar pairs by cosine similarity.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    reviews = [json.loads(l) for l in (ROOT / "data" / "raw_reviews.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = SentenceTransformer(MODEL).encode([r["text"] for r in reviews], normalize_embeddings=True)

    centred = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T

    sim = vectors @ vectors.T
    pairs = sorted(
        ({"a": reviews[i]["id"], "b": reviews[j]["id"], "cosine": round(float(sim[i, j]), 3)}
         for i in range(len(reviews)) for j in range(i + 1, len(reviews))),
        key=lambda p: -p["cosine"],
    )

    out = {
        "model": MODEL,
        "points": [{"id": r["id"], "x": round(float(x), 4), "y": round(float(y), 4)} for r, (x, y) in zip(reviews, xy)],
        "pairs": pairs,
    }
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "embeddings.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for p in pairs[:5]:
        print(f"{p['a']} ~ {p['b']}  cosine={p['cosine']}")


if __name__ == "__main__":
    main()
