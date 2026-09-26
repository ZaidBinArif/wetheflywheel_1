# Gangnam review normaliser

A small pipeline for [Gangnam Beauty Guide](https://gangnambeautyguide.com)'s hardest problem: turning scattered
Korean clinic reviews (Naver/Daum cafes, blogs, English forums) into **one de-duplicated, translated, structured
record per real experience**, tied to a **canonical clinic**, with a **trust score you can explain**.

> All reviews in `data/` are synthetic and all clinic and surgeon names are fictional.

## What it does

```
raw_reviews.jsonl ──► 1. dedupe ──► 2. extract (Claude) ──► 3. resolve clinic ──► 4. verify price ──► 5. trust score
   10 posts            9 unique       translation + fields     aliases + fuzzy       regex vs. model       rule-based
```

| Step | How | Why this way |
|---|---|---|
| 1. Dedupe | Character 3-gram Jaccard ≥ 0.5, earliest post is canonical | The same review gets cross-posted to cafe + blog. Removing copies **before** the API calls means you don't pay to process them twice. Korean has no reliable word boundaries, so character shingles work better than word ones. |
| 2. Extract | `claude-opus-5`, structured output against a Pydantic schema | Claude translates, pulls out procedures (fixed list), price, outcome, complications and trust signals. |
| 3. Resolve clinic | Claude returns the clinic name **as written**; code maps it to an ID | Claude never assigns clinic IDs. A made-up ID would quietly merge two clinics. `강남 HB성형외과`, `Haneulbit PS` and `하늘빛성형` all reduce to one key. Anything unknown goes to a *needs review* queue instead of being guessed. |
| 4. Verify price | Regex over the raw text (`만원`, `원`, `won`, `million KRW`) | Korean prices are quoted in 만원 (×10,000), so an off-by-10,000 mistake is the likeliest error. The model's number only counts as *verified* if the source text contains it. |
| 5. Trust score | Transparent weights (`TRUST_WEIGHTS` in the code) | +first-person, +proof (photos/receipt), +surgeon named, +verified price, +cross-posted; **−50 sponsored**, −20 marketing tone. A regex for 협찬 / 체험단 / 원고료 / 지원받아 backs up the model, and `광고 아님` ("not an ad") is excluded. |

## Edge cases in the sample data

- **r01 / r02**: the same eyelid review on Naver Cafe and Naver Blog → merged into one record with two sources.
- **r01**: two prices, `처음에 320 불렀는데` (quoted 320만원) and `280만원` (paid) → price is 2,800,000, and the quote goes in `price_note`.
- **r03**: `HB성형외과` alias + sponsorship disclosure at the very end + generic praise → low trust.
- **r04**: negative review with nerve damage and a revision → complications captured, nothing softened.
- **r05 / r09**: English posts; r09 is a *quote only*, not a procedure.
- **r08**: a clinic that isn't in the registry → *unmatched*, flagged for a human.
- **r10**: same clinic, surgeon and procedure family as r01, but a different person → **not** merged.

## Run it

```bash
pip install -r requirements.txt
python normalise.py --cli       # full run through the Claude Code CLI (your Claude login, no API key)
python normalise.py             # full run through the API: needs ANTHROPIC_API_KEY
python normalise.py --offline   # no Claude at all: dedupe + clinic text-scan + price regex
python embed.py                 # local multilingual embeddings for the map view
python build_site.py            # bakes everything into docs/index.html (GitHub Pages)
```

Output goes to `out/<mode>/reviews.json` (one record per unique review) and `out/<mode>/clinics.md` (a summary per clinic).
The committed `out/claude/` is a real run through `--cli`.

## The page

`docs/index.html` is one static page with a switch at the top:

- **Before**: the 10 raw posts, as scraped.
- **After**: 9 normalised records sorted by trust score, showing the resolved clinic, a price check, flags and the translation.
- **Embeddings**: every raw post embedded with a local multilingual model and projected to 2D. Dashed lines join posts that were merged as duplicates.
  The table shows the *closest* pairs. Some are different people describing the same clinic and procedure. That's why the merge uses
  exact text overlap, and the map is only for exploring.

## What's next at real scale

- **Batch API** for the extraction step: 50% cheaper, and speed doesn't matter for a crawler.
- Benchmark a smaller model on the extraction step against a hand-labelled set of about 100 reviews before paying Opus prices for millions of reviews.
- Replace the pairwise dedupe with MinHash/LSH once there are more than about 10k reviews per clinic.
- A clinic registry built from Korean business registration data (사업자등록), so the list of aliases grows by itself.
- Surgeon matching against the Korean Society of Plastic Surgeons member list, which gives you *verified surgeon*.
