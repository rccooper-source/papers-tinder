from __future__ import annotations
import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
CHANNEL_ID = "1147227076655595520"
CACHE_TTL_HOURS = 24

# ── Storage backend ───────────────────────────────────────────────────────────

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

    def storage_hset(key: str, field: str, value: str):
        _redis.hset(key, field, value)

    def storage_hgetall(key: str) -> dict:
        return _redis.hgetall(key)

else:
    _DATA = Path(__file__).parent

    def storage_get(key: str) -> str | None:
        f = _DATA / f"{key}.json"
        return f.read_text() if f.exists() else None

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

    def storage_hset(key: str, field: str, value: str):
        f = _DATA / f"{key}.json"
        data = json.loads(f.read_text()) if f.exists() else {}
        data[field] = value
        f.write_text(json.dumps(data))

    def storage_hgetall(key: str) -> dict:
        f = _DATA / f"{key}.json"
        return json.loads(f.read_text()) if f.exists() else {}


app = FastAPI()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini = None


# ── Discord ───────────────────────────────────────────────────────────────────

async def fetch_discord_messages(days_back: int = 7) -> list:
    """Fetch all messages from the last N days, paginating as needed."""
    if not DISCORD_TOKEN:
        raise HTTPException(500, "DISCORD_TOKEN not set")
    
    cutoff = datetime.now() - timedelta(days=days_back)
    all_messages = []
    before_id = None
    
    async with httpx.AsyncClient() as client:
        while True:
            params = {"limit": 100}
            if before_id:
                params["before"] = before_id
            
            r = await client.get(
                f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                params=params,
                headers={"Authorization": DISCORD_TOKEN},
                timeout=15,
            )
            if r.status_code == 401:
                raise HTTPException(401, "Invalid Discord token")
            r.raise_for_status()
            
            batch = r.json()
            if not batch:
                break
            
            for msg in batch:
                ts = msg.get("timestamp", "")
                if ts:
                    # Parse Discord ISO timestamp (e.g., "2026-03-31T22:15:30.123456+00:00")
                    ts_clean = ts.replace("Z", "+00:00")
                    msg_time = datetime.fromisoformat(ts_clean).replace(tzinfo=None)
                    if msg_time < cutoff:
                        return all_messages
                all_messages.append(msg)
            
            before_id = batch[-1]["id"]
            
            if len(batch) < 100:
                break
            
            await asyncio.sleep(0.5)  # rate limit
    
    return all_messages


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

async def fetch_s2_paper(arxiv_id: str, retries: int = 3) -> dict | None:
    """Fetch paper from Semantic Scholar with retry on rate limit."""
    async with httpx.AsyncClient() as client:
        for attempt in range(retries):
            r = await client.get(
                f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
                params={"fields": "title,authors,year,abstract"},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait)
                continue
            break
    return None


# ── Summarization ─────────────────────────────────────────────────────────────

async def summarize(title: str, abstract: str) -> str:
    if not abstract:
        return "No abstract available."
    if gemini:
        try:
            prompt = (
                "Summarize this ML paper in exactly 4 sentences:\n"
                "1. What problem does it solve?\n"
                "2. What is the proposed method or approach?\n"
                "3. What are the key results or findings?\n"
                "4. Why does it matter or what's the broader impact?\n"
                "Be concise and plain-language. No bullet points, just 4 flowing sentences.\n\n"
                f"Title: {title}\nAbstract: {abstract[:1000]}"
            )
            response = await asyncio.to_thread(gemini.generate_content, prompt)
            return response.text.strip()
        except Exception:
            pass
    sentences = re.split(r'(?<=[.!?])\s+', abstract.strip())
    return " ".join(sentences[:4])


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
        ex_seconds=int(CACHE_TTL_HOURS * 3600 * 2),
    )


# ── Decisions ─────────────────────────────────────────────────────────────────

def load_decisions() -> dict:
    """Returns {arxiv_id: 'yes'|'no'} for all swiped papers."""
    return storage_hgetall("paperswipe_decisions")

def record_decision(arxiv_id: str, decision: str):
    storage_hset("paperswipe_decisions", arxiv_id, decision)
    storage_sadd("paperswipe_seen", arxiv_id)


# ── Notebooks (audio overviews) ───────────────────────────────────────────────

def load_notebooks() -> dict:
    """Returns {arxiv_id: notebook_id} for papers with generated audio."""
    return storage_hgetall("paperswipe_notebooks")


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def build_papers(days_back: int = 7) -> list:
    messages = await fetch_discord_messages(days_back=days_back)
    entries = extract_arxiv_ids(messages)
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
            aff for a in authors_raw[:8] for aff in a.get("affiliations", [])
        ))[:4]
        summary = await summarize(title, abstract)
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
        await asyncio.sleep(1.5)  # Semantic Scholar rate limit: ~1 req/sec
    return papers


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/papers")
async def get_papers():
    """Unseen papers only — for the swipe queue."""
    cached = load_cache()
    if not cached:
        all_papers = await build_papers()
        save_cache(all_papers)
    else:
        all_papers = cached["papers"]
    seen = storage_smembers("paperswipe_seen")
    notebooks = load_notebooks()
    result = []
    for p in all_papers:
        if p["id"] not in seen:
            notebook_id = notebooks.get(p["id"])
            result.append({
                **p,
                "notebook_id": notebook_id,
                "audio_url": f"https://notebooklm.google.com/notebook/{notebook_id}" if notebook_id else None,
            })
    return result


@app.get("/api/all")
async def get_all_papers():
    """All papers with swipe decision and notebook info attached — for the home list."""
    cached = load_cache()
    if not cached:
        all_papers = await build_papers()
        save_cache(all_papers)
    else:
        all_papers = cached["papers"]
    decisions = load_decisions()
    seen = storage_smembers("paperswipe_seen")
    notebooks = load_notebooks()
    result = []
    for p in all_papers:
        notebook_id = notebooks.get(p["id"])
        result.append({
            **p,
            "decision": decisions.get(p["id"]),          # "yes", "no", or None
            "read": p["id"] in seen,
            "notebook_id": notebook_id,
            "audio_url": f"https://notebooklm.google.com/notebook/{notebook_id}" if notebook_id else None,
        })
    # Sort: unread first, then by posted_at desc
    result.sort(key=lambda p: (p["read"], -(datetime.fromisoformat(p["posted_at"].replace("Z","")).timestamp() if p.get("posted_at") else 0)))
    return result


@app.post("/api/seen/{arxiv_id}")
async def mark_paper_seen(arxiv_id: str, decision: Optional[str] = Query(None)):
    """Record a swipe. decision = 'yes' or 'no'."""
    if decision in ("yes", "no"):
        record_decision(arxiv_id, decision)
    else:
        storage_sadd("paperswipe_seen", arxiv_id)
    return {"ok": True}


@app.get("/api/refresh")
async def refresh_papers():
    papers = await build_papers()
    save_cache(papers)
    seen = storage_smembers("paperswipe_seen")
    notebooks = load_notebooks()
    result = []
    for p in papers:
        if p["id"] not in seen:
            notebook_id = notebooks.get(p["id"])
            result.append({
                **p,
                "notebook_id": notebook_id,
                "audio_url": f"https://notebooklm.google.com/notebook/{notebook_id}" if notebook_id else None,
            })
    return result


@app.get("/api/status")
async def status():
    cached = load_cache()
    seen = storage_smembers("paperswipe_seen")
    all_papers = cached.get("papers", [])
    return {
        "cached": bool(cached),
        "total": len(all_papers),
        "seen": len(seen),
        "unseen": len([p for p in all_papers if p["id"] not in seen]),
        "timestamp": cached.get("timestamp"),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
