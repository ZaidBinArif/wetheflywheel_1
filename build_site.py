"""Bake the pipeline output into a single static page: docs/index.html (served by GitHub Pages)."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    data = {
        "raw": [json.loads(l) for l in (ROOT / "data" / "raw_reviews.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()],
        "records": load("out/claude/reviews.json"),
        "clinics": {c["id"]: c for c in load("data/clinics.json")},
        # optional: the map tab only appears after `python embed.py` has been run
        "embeddings": load("out/embeddings.json") if (ROOT / "out" / "embeddings.json").exists() else None,
    }
    html = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    # "</" is escaped so review text can never close the <script> tag
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
