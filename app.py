from __future__ import annotations
import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")          # set automatically by Railway Redis plugin
CHANNEL_ID = "1147227076655595520"
CACHE_TTL_HOURS = 24

# ── Storage backend (Redis if available, else local files) ────────────────────

if REDIS_URL:
    import redis as redislib
    _redis = redislib.from_url(REDIS_URL, decode_responses=True)

    def storage_get(key: str) -> str | None:
        return _redis.get(key)

    def storage_set(key: str, value: str, ex_seconds: int | None = None):
        _redis.set(key, value, ex=ex_seconds)

    def storage_sadd(key: str, member: str):
        _redis.sadd(key, member)

    def storage_smembers(key: str) -> set:
        return _redis.smembers(key)
else:
    # Local file fallback (for dev / local Mac use)
    _DATA = Path(__file__).parent
    CACHE_FILE = _DATA / "papers_cache.json"
    SEEN_FILE  = _DATA / "seen_papers.json"

    def storage_get(key: str) -> str | None:
        f = _DATA / f"{key}.json"
        if f.exists():
            return f.read_text()
        return None

    def storage_set(key: str, value: str, ex_seconds: int | None = None):
        (_DATA / f"{key}.json").write_text(value)

    def storage_sadd(key: str, member: str):
        f = _DATA / f"{key}.json"
        members = set(json.loads(f.read_text())) if f.exists() else set()
        members.add(member)
        f.write_text(json.dumps(list(members)))

    def storage_smembers(key: str) -> set:
        f = _DATA / f"{key}.json"
        return set(json.loads(f.read_text())) if f.exists() else set()


app = FastAPI()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini = None


# ── Discord ───────────────────────────────────────────────────────────────────

async def fetch_discord_messages(limit: int = 100) -> list:
    if not DISCORD_TOKEN:
        raise HTTPException(500, "DISCORD_TOKEN not set")
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
            params={"limit": limit},
            headers={"Authorization": DISCORD_TOKEN},
            timeout=15,
        )
        if r.status_code == 401:
            raise HTTPException(401, "Invalid Discord token")
        r.raise_for_status()
        return r.json()


def extract_arxiv_ids(messages: list) -> list:
    pattern = r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)'
    seen, results = set(), []
    for msg in messages:
        text = msg.get("content", "")
        for embed in msg.get("embeds", []):
            text += " " + embed.get("url", "") + " " + embed.get("description", "")
        for m in re.findall(pattern, text, re.IGNORECASE):
            base = re.sub(r'v\d+$', '', m)
            if base not in seen:
                seen.add(base)
                results.append({
                    "arxiv_id": base,
                    "posted_at": msg.get("timestamp"),
                })
    return results


# ── Semantic Scholar ──────────────────────────────────────────────────────────

async def fetch_s2_paper(arxiv_id: str) -> dict | None:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
            params={"fields": "title,authors,year,abstract"},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    return None


# ── Summarization ─────────────────────────────────────────────────────────────

async def one_sentence_summary(title: str, abstract: str) -> str:
    if not abstract:
        return "No abstract available."
    if gemini:
        try:
            prompt = (
                "Summarize this ML paper in ONE crisp sentence (max 25 words). "
                "Lead with what it does, end with the key result or insight.\n\n"
                f"Title: {title}\nAbstract: {abstract[:800]}"
            )
            response = await asyncio.to_thread(gemini.generate_content, prompt)
            return response.text.strip().rstrip('.')
        except Exception:
            pass
    sentence = re.split(r'(?<=[.!?])\s+', abstract.strip())[0]
    return sentence[:197] + "…" if len(sentence) > 200 else sentence


# ── Cache ─────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    raw = storage_get("paperswipe_cache")
    if raw:
        data = json.loads(raw)
        ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
        if datetime.now() - ts < timedelta(hours=CACHE_TTL_HOURS):
            return data
    return {}

def save_cache(papers: list):
    storage_set(
        "paperswipe_cache",
        json.dumps({"timestamp": datetime.now().isoformat(), "papers": papers}),
        ex_seconds=int(CACHE_TTL_HOURS * 3600 * 2),  # keep in Redis for 2x TTL
    )


# ── Seen papers ───────────────────────────────────────────────────────────────

def load_seen() -> set:
    return storage_smembers("paperswipe_seen")

def mark_seen(arxiv_id: str):
    storage_sadd("paperswipe_seen", arxiv_id)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def build_papers(max_papers: int = 25) -> list:
    messages = await fetch_discord_messages()
    entries = extract_arxiv_ids(messages)[:max_papers]
    papers = []
    for entry in entries:
        arxiv_id = entry["arxiv_id"]
        data = await fetch_s2_paper(arxiv_id)
        if not data or not data.get("title"):
            continue
        title = data["title"]
        abstract = data.get("abstract", "")
        authors_raw = data.get("authors", [])
        author_names = [a["name"] for a in authors_raw[:5]]
        if len(authors_raw) > 5:
            author_names.append(f"+ {len(authors_raw) - 5} more")
        affiliations = list(dict.fromkeys(
            aff
            for a in authors_raw[:8]
            for aff in a.get("affiliations", [])
        ))[:4]
        summary = await one_sentence_summary(title, abstract)
        papers.append({
            "id": arxiv_id,
            "title": title,
            "year": data.get("year"),
            "authors": author_names,
            "affiliations": affiliations,
            "summary": summary,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "posted_at": entry["posted_at"],
        })
        await asyncio.sleep(0.35)
    return papers


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/papers")
async def get_papers():
    cached = load_cache()
    if not cached:
        all_papers = await build_papers()
        save_cache(all_papers)
    else:
        all_papers = cached["papers"]
    seen = load_seen()
    return [p for p in all_papers if p["id"] not in seen]


@app.post("/api/seen/{arxiv_id}")
async def mark_paper_seen(arxiv_id: str):
    mark_seen(arxiv_id)
    return {"ok": True}


@app.get("/api/refresh")
async def refresh_papers():
    papers = await build_papers()
    save_cache(papers)
    seen = load_seen()
    return [p for p in papers if p["id"] not in seen]


@app.get("/api/status")
async def status():
    cached = load_cache()
    seen = load_seen()
    all_papers = cached.get("papers", [])
    return {
        "cached": bool(cached),
        "total": len(all_papers),
        "seen": len(seen),
        "unseen": len([p for p in all_papers if p["id"] not in seen]),
        "timestamp": cached.get("timestamp"),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
