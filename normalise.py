"""Normalise raw Korean clinic reviews into structured, English, de-duplicated records.

Pipeline (only step 2 calls an LLM; everything that decides something is plain code):
  1. dedupe   - group cross-posted copies of the same review (char-shingle Jaccard)
  2. extract  - Claude translates and pulls structured fields (Pydantic schema)
  3. resolve  - map the clinic *as written* onto a known clinic id (aliases + fuzzy)
  4. verify   - check Claude's price against prices regex-parsed from the raw text
  5. score    - transparent, rule-based trust score

Usage:
  python normalise.py              # full run, needs ANTHROPIC_API_KEY
  python normalise.py --cli        # full run through the Claude Code CLI, no API key
  python normalise.py --offline    # steps 1, 3, 4, 5 only - no API calls
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
MODEL = "claude-opus-5"

# ---------------------------------------------------------------------------
# 1. Dedupe
# ---------------------------------------------------------------------------

def _shingles(text: str, n: int = 3) -> set[str]:
    t = re.sub(r"[\W_]+", "", text.lower())  # \W keeps Hangul, drops spaces/punctuation/emoji
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def dedupe(reviews: list[dict], threshold: float = 0.5) -> list[list[dict]]:
    """Group copies of the same review posted to several sources. Earliest post comes first."""
    groups: list[tuple[set[str], list[dict]]] = []
    for review in sorted(reviews, key=lambda r: r["posted_at"]):
        sh = _shingles(review["text"])
        for group_sh, members in groups:
            if _jaccard(sh, group_sh) >= threshold:
                members.append(review)
                break
        else:
            groups.append((sh, [review]))
    return [members for _, members in groups]

# ---------------------------------------------------------------------------
# 2. Extract (Claude)
# ---------------------------------------------------------------------------

Procedure = Literal[
    "double_eyelid_incisional", "double_eyelid_non_incisional", "ptosis_correction",
    "rhinoplasty", "zygoma_reduction", "jaw_reduction", "fat_grafting", "eye_revision",
    "breast_augmentation", "liposuction", "botox", "filler", "laser_toning", "ulthera", "other",
]


class TrustSignals(BaseModel):
    first_person: bool = Field(description="Author describes their own procedure, not hearsay")
    before_after_photos: bool = Field(description="Author says photos are attached or offered")
    receipt_or_proof: bool = Field(description="Receipt, invoice or other proof of purchase mentioned")
    sponsored_disclosure: bool = Field(description="Any disclosure of payment, free/discounted treatment, 협찬, 체험단, 원고료")
    marketing_tone: bool = Field(description="Reads like promotion: vague praise, no downsides, calls to action")


class Extraction(BaseModel):
    language: Literal["ko", "en", "other"]
    review_type: Literal["procedure", "consultation_only", "not_a_review"]
    translation_en: str = Field(description="Faithful English translation; for English input repeat the text")
    summary_en: str = Field(description="One neutral sentence")
    clinic_mention: Optional[str] = Field(description="Clinic name exactly as written, or null")
    surgeon_mention: Optional[str] = Field(description="Surgeon name exactly as written, or null")
    procedures: list[Procedure]
    price_krw: Optional[int] = Field(description="Final amount paid (or quoted, for consultations) in whole KRW")
    price_verbatim: Optional[str] = Field(description="The price exactly as written in the text")
    price_note: Optional[str] = Field(description="VAT, discounts, packages, earlier quotes")
    outcome: Literal["positive", "mixed", "negative", "unknown"]
    complications: list[str]
    revision_needed: bool
    signals: TrustSignals


SYSTEM_PROMPT = """You extract structured data from reviews of Korean plastic surgery and aesthetic clinics.
Most reviews come from Naver/Daum cafes and blogs and are written in casual Korean.

Rules:
- clinic_mention and surgeon_mention: copy the name exactly as it appears. Do not correct,
  expand or translate it. Downstream code matches it against a clinic registry.
- Never guess a surgeon from the clinic. If no surgeon is named, use null.
- Prices: Korean amounts are usually in 만원 (10,000 KRW). 280만원 = 2,800,000.
  A bare number next to a 만원 amount (e.g. "처음에 320 불렀는데") is also in 만원.
  price_krw is the final amount paid. Earlier quotes, VAT and discounts go in price_note.
  If the author only got a quote, set review_type to consultation_only and use the quote.
- sponsored_disclosure is true for 협찬, 체험단, 원고료, "지원받아/제공받아 작성", or any paid or free
  treatment in exchange for the post. "내돈내산" and "광고 아님" mean NOT sponsored.
- marketing_tone: generic praise (friendly staff, clean facility), no specific downsides,
  and calls to action ("댓글 남겨주세요", "상담 문의") are typical of hidden advertising.
- Translate slang and emoticons by meaning (ㅠ = sad, ㅋㅋ = laughing). Don't add or smooth over
  complaints."""


def extract(client, review: dict, model: str) -> Extraction:
    import anthropic  # only needed for online runs

    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Source: {review['source']}\nPosted: {review['posted_at']}\n\n<review>\n{review['text']}\n</review>",
        }],
        output_format=Extraction,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise anthropic.AnthropicError(f"no structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


def cli_schema(model: type[BaseModel]) -> dict:
    """Self-contained strict schema for --json-schema: inline Pydantic's $ref/$defs,
    drop titles, and forbid extra keys on every object."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(defs[node["$ref"].split("/")[-1]])
            node = {k: fix(v) for k, v in node.items() if k != "title"}
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            return node
        if isinstance(node, list):
            return [fix(v) for v in node]
        return node

    return fix(schema)


def extract_via_cli(review: dict, model: str) -> Extraction:
    """Same extraction, but through the Claude Code CLI (`claude -p`) - uses your Claude login, no API key."""
    import shutil
    import subprocess
    import tempfile

    # On Windows `claude` is a .CMD shim, and cmd.exe cuts arguments at newlines - which silently
    # dropped everything after the multi-line prompt, including --json-schema. Pass the prompt as a file.
    prompt_file = Path(tempfile.gettempdir()) / "gbg_system_prompt.txt"
    prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")
    cmd = [
        shutil.which("claude") or "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
        "--append-system-prompt-file", str(prompt_file),
        "--json-schema", json.dumps(cli_schema(Extraction)),
    ]
    prompt = f"Source: {review['source']}\nPosted: {review['posted_at']}\n\n<review>\n{review['text']}\n</review>"
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                          cwd=tempfile.gettempdir(), timeout=300)  # neutral cwd: no project context leaks in
    result = json.loads(proc.stdout)
    if result.get("is_error") or not result.get("structured_output"):
        raise RuntimeError(result.get("result") or proc.stderr[:300])
    return Extraction.model_validate(result["structured_output"])

# ---------------------------------------------------------------------------
# 3. Resolve clinic
# ---------------------------------------------------------------------------

LOCATION_PREFIXES = ["강남", "신사", "압구정", "청담", "논현", "gangnam", "sinsa", "apgujeong", "cheongdam"]
SUFFIXES = sorted(
    ["성형외과의원", "성형외과", "성형", "피부과의원", "피부과", "의원", "클리닉",
     "plasticsurgery", "clinic", "hospital", "ps"],
    key=len, reverse=True,
)


def clinic_key(name: str) -> str:
    """'강남 HB성형외과' -> 'hb', 'Dain PS (Da-in)' -> 'dain'."""
    s = re.sub(r"\(.*?\)", "", name.lower())
    s = re.sub(r"[\W_]+", "", s)
    for p in LOCATION_PREFIXES:
        if s.startswith(p) and len(s) > len(p):
            s = s[len(p):]
    for suf in SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


class ClinicRegistry:
    def __init__(self, clinics: list[dict]):
        self.clinics = {c["id"]: c for c in clinics}
        self.index: dict[str, str] = {}
        for c in clinics:
            for alias in [c["name_ko"], c["name_en"], *c["aliases"]]:
                self.index[clinic_key(alias)] = c["id"]

    def resolve(self, mention: Optional[str]) -> tuple[Optional[str], str]:
        if not mention:
            return None, "no_mention"
        key = clinic_key(mention)
        if key in self.index:
            return self.index[key], "exact"
        if len(key) >= 3:  # short keys like "hb" fuzzy-match far too much
            score, cid = max((SequenceMatcher(None, key, k).ratio(), cid) for k, cid in self.index.items())
            if score >= 0.85:
                return cid, f"fuzzy:{score:.2f}"
        return None, "unmatched"

    def scan(self, text: str) -> tuple[Optional[str], str]:
        """Offline fallback: find a known alias anywhere in the raw text (longest alias wins)."""
        flat = re.sub(r"[\W_]+", "", text.lower())
        hits = [(len(k), cid) for k, cid in self.index.items() if len(k) >= 2 and k in flat]
        return (max(hits)[1], "text_scan") if hits else (None, "unmatched")

# ---------------------------------------------------------------------------
# 4. Verify price
# ---------------------------------------------------------------------------

PRICE_PATTERNS = [
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*만\s*원"), 10_000),
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*million\s*(?:won|krw)"), 1_000_000),
    (re.compile(r"(\d[\d,]*)\s*(?:원|won|krw)"), 1),
]


def prices_in_text(text: str) -> set[int]:
    found = set()
    for pattern, multiplier in PRICE_PATTERNS:
        for m in pattern.finditer(text.lower()):
            found.add(int(float(m.group(1).replace(",", "")) * multiplier))
    return found


def check_price(price_krw: Optional[int], text: str) -> str:
    found = prices_in_text(text)
    if price_krw is None:
        return "none"
    if price_krw in found:
        return "verified"
    return "mismatch" if found else "unverified"

# ---------------------------------------------------------------------------
# 5. Trust score
# ---------------------------------------------------------------------------

# A regex backstop for sponsorship: cheap, and it catches what the model misses.
SPONSOR_RE = re.compile(r"협찬|체험단|원고료|지원받아|제공받아|광고(?!\s*(?:아님|아닙니다|아니에요|x))|#ad\b|sponsored", re.I)

TRUST_WEIGHTS = {
    "first_person": 25,
    "proof": 20,             # before/after photos or receipt
    "surgeon_named": 15,
    "price_verified": 15,
    "cross_posted": 5,       # same story posted by the same author in several places
    "sponsored": -50,
    "marketing_tone": -20,
}


def trust_score(signals: dict) -> tuple[int, list[str]]:
    reasons = [k for k, v in signals.items() if v]
    raw = sum(TRUST_WEIGHTS[k] for k in reasons)
    return max(0, min(100, 20 + raw)), reasons

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def build_record(group: list[dict], ex: Optional[Extraction], registry: ClinicRegistry) -> dict:
    canonical = group[0]
    text = canonical["text"]

    if ex:
        clinic_id, match = registry.resolve(ex.clinic_mention)
        price_krw = ex.price_krw
        llm_signals = ex.signals
    else:
        clinic_id, match = registry.scan(text)
        found = sorted(prices_in_text(text))
        price_krw = found[-1] if len(found) == 1 else None  # offline: only trust an unambiguous price
        llm_signals = None

    # Offline, the price *came from* the regex, so checking it against the regex proves nothing.
    price_check = check_price(price_krw, text) if ex else ("regex_only" if price_krw else "none")
    sponsored = bool(SPONSOR_RE.search(text)) or bool(llm_signals and llm_signals.sponsored_disclosure)
    signals = {
        "first_person": bool(llm_signals and llm_signals.first_person),
        "proof": bool(llm_signals and (llm_signals.before_after_photos or llm_signals.receipt_or_proof)),
        "surgeon_named": bool(ex and ex.surgeon_mention),
        "price_verified": price_check == "verified",
        "cross_posted": len(group) > 1,
        "sponsored": sponsored,
        "marketing_tone": bool(llm_signals and llm_signals.marketing_tone),
    }
    score, reasons = trust_score(signals)
    clinic = registry.clinics.get(clinic_id)

    return {
        "id": canonical["id"],
        "sources": [{"id": r["id"], "source": r["source"], "posted_at": r["posted_at"]} for r in group],
        "clinic": {
            "id": clinic_id,
            "name_en": clinic["name_en"] if clinic else None,
            "mention": ex.clinic_mention if ex else None,
            "match": match,
        },
        "surgeon_mention": ex.surgeon_mention if ex else None,
        "review_type": ex.review_type if ex else None,
        "procedures": ex.procedures if ex else [],
        "price": {
            "krw": price_krw,
            "verbatim": ex.price_verbatim if ex else None,
            "note": ex.price_note if ex else None,
            "check": price_check,
        },
        "outcome": ex.outcome if ex else None,
        "complications": ex.complications if ex else [],
        "revision_needed": ex.revision_needed if ex else None,
        "summary_en": ex.summary_en if ex else None,
        "translation_en": ex.translation_en if ex else None,
        "trust": {"score": score, "signals": reasons},
        "original_text": text,
    }


def clinic_report(records: list[dict], registry: ClinicRegistry) -> str:
    by_clinic = defaultdict(list)
    for r in records:
        by_clinic[r["clinic"]["id"]].append(r)

    lines = ["# Clinic roll-up", "",
             "| Clinic | Reviews | Avg trust | Procedures | Prices found (KRW) | Negative |",
             "|---|---|---|---|---|---|"]
    for cid, recs in sorted(by_clinic.items(), key=lambda kv: kv[0] or "~"):
        name = registry.clinics[cid]["name_en"] if cid else "**Unmatched - needs review**"
        procs = Counter(p for r in recs for p in r["procedures"])
        prices = [r["price"]["krw"] for r in recs if r["price"]["check"] in ("verified", "regex_only")]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            name,
            len(recs),
            round(statistics.mean(r["trust"]["score"] for r in recs)),
            ", ".join(f"{p} x{n}" for p, n in procs.most_common()) or "-",
            ", ".join(f"{p:,}" for p in sorted(prices)) or "-",
            sum(r["outcome"] == "negative" for r in recs),
        ))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="skip the Claude extraction step")
    parser.add_argument("--cli", action="store_true", help="extract via the `claude -p` CLI instead of the API")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw_reviews.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    out_dir = args.out or ROOT / "out" / ("offline" if args.offline else "claude")
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = ClinicRegistry(json.loads((ROOT / "data" / "clinics.json").read_text(encoding="utf-8")))
    reviews = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]

    groups = dedupe(reviews)
    print(f"{len(reviews)} raw reviews -> {len(groups)} unique (dedupe runs before any API call)")

    if args.offline:
        run = None
    elif args.cli:
        run = lambda review: extract_via_cli(review, args.model)
    else:
        import anthropic
        client = anthropic.Anthropic()
        run = lambda review: extract(client, review, args.model)

    def safe_run(review):
        try:
            return run(review) if run else None
        except Exception as e:  # keep going; one bad review shouldn't sink the batch
            print(f"  {review['id']}: extraction failed - {e}")
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        extractions = list(pool.map(safe_run, [g[0] for g in groups]))

    records = []
    for group, ex in zip(groups, extractions):
        rec = build_record(group, ex, registry)
        records.append(rec)
        print(f"  {rec['id']}: clinic={rec['clinic']['id'] or '?':<11} ({rec['clinic']['match']}) "
              f"price={rec['price']['krw']} [{rec['price']['check']}] trust={rec['trust']['score']}")

    (out_dir / "reviews.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "clinics.md").write_text(clinic_report(records, registry), encoding="utf-8")
    print(f"wrote {out_dir / 'reviews.json'} and {out_dir / 'clinics.md'}")


if __name__ == "__main__":
    main()
