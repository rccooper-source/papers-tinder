"""
NotebookLM worker — polls for new "yes" decisions and triggers audio generation.
Runs as a separate Railway service (or locally).
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REDIS_URL      = os.getenv("REDIS_URL")
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL_SECONDS", "120"))  # check every 2 min
AUDIO_LANGUAGE = os.getenv("AUDIO_LANGUAGE", "en")

NOTEBOOKLM_AUTH = os.getenv("NOTEBOOKLM_AUTH_JSON")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(message)s")
log = logging.getLogger(__name__)

# ── Storage (same backend as app.py) ─────────────────────────────────────────

if REDIS_URL:
    import redis as redislib
    _redis = redislib.from_url(REDIS_URL, decode_responses=True)

    def get_decisions() -> dict:
        return _redis.hgetall("paperswipe_decisions")

    def get_cache() -> list:
        raw = _redis.get("paperswipe_cache")
        return json.loads(raw).get("papers", []) if raw else []

    def mark_queued(arxiv_id: str):
        _redis.sadd("paperswipe_notebooklm_queued", arxiv_id)

    def is_queued(arxiv_id: str) -> bool:
        return bool(_redis.sismember("paperswipe_notebooklm_queued", arxiv_id))

    def save_notebook_id(arxiv_id: str, notebook_id: str):
        _redis.hset("paperswipe_notebooks", arxiv_id, notebook_id)

else:
    _DATA = Path(__file__).parent

    def get_decisions() -> dict:
        f = _DATA / "paperswipe_decisions.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def get_cache() -> list:
        f = _DATA / "paperswipe_cache.json"
        return json.loads(f.read_text()).get("papers", []) if f.exists() else []

    def _queued_set() -> set:
        f = _DATA / "paperswipe_notebooklm_queued.json"
        return set(json.loads(f.read_text())) if f.exists() else set()

    def mark_queued(arxiv_id: str):
        s = _queued_set(); s.add(arxiv_id)
        (_DATA / "paperswipe_notebooklm_queued.json").write_text(json.dumps(list(s)))

    def is_queued(arxiv_id: str) -> bool:
        return arxiv_id in _queued_set()

    def save_notebook_id(arxiv_id: str, notebook_id: str):
        f = _DATA / "paperswipe_notebooks.json"
        data = json.loads(f.read_text()) if f.exists() else {}
        data[arxiv_id] = notebook_id
        f.write_text(json.dumps(data))


# ── NotebookLM ────────────────────────────────────────────────────────────────

async def create_notebook_for_paper(paper: dict) -> str | None:
    """Creates a NotebookLM notebook with the arxiv PDF and triggers 15-min audio."""
    try:
        from notebooklm import NotebookLMClient

        title = paper["title"][:60]
        pdf_url = paper["pdf_url"]
        arxiv_url = paper["arxiv_url"]

        log.info(f"Creating notebook for: {title}")

        async with await NotebookLMClient.from_storage() as client:
            # Create notebook
            nb = await client.notebooks.create(f"[PaperSwipe] {title}")
            log.info(f"  Notebook created: {nb.id}")

            # Add arxiv abstract page + PDF as sources
            await client.sources.add_url(nb.id, arxiv_url, wait=True)
            log.info(f"  Source added: {arxiv_url}")

            # Generate the longer (~15 min) audio overview
            status = await client.artifacts.generate_audio(
                nb.id,
                instructions=(
                    "Create a detailed, engaging deep-dive podcast episode about this ML paper. "
                    "Cover: the problem being solved, the key method or architecture, "
                    "experimental results, and why this matters to the field. "
                    "Make it accessible but technically rich. Use the longer format."
                ),
                length="long",
                language=AUDIO_LANGUAGE,
            )
            log.info(f"  Audio generation triggered, task: {status.task_id}")

            # Don't wait for completion — it takes 15 min, just fire and forget
            # The notebook will be ready in NotebookLM when done

            return nb.id

    except Exception as e:
        log.error(f"  Failed to create notebook: {e}")
        return None


# ── Poll loop ─────────────────────────────────────────────────────────────────

async def verify_notebooklm_auth() -> bool:
    """Check if NotebookLM authentication is available."""
    if NOTEBOOKLM_AUTH:
        log.info("NotebookLM auth: using NOTEBOOKLM_AUTH_JSON env var")
        return True
    
    storage_path = Path.home() / ".notebooklm" / "profiles" / "default" / "storage_state.json"
    if storage_path.exists():
        log.info(f"NotebookLM auth: using {storage_path}")
        return True
    
    legacy_path = Path.home() / ".notebooklm" / "storage_state.json"
    if legacy_path.exists():
        log.info(f"NotebookLM auth: using {legacy_path}")
        return True
    
    log.error(
        "NotebookLM auth not configured! Either:\n"
        "  1. Set NOTEBOOKLM_AUTH_JSON env var with storage_state.json contents, or\n"
        "  2. Run: notebooklm login (locally)"
    )
    return False


async def poll():
    log.info(f"Worker started. Polling every {POLL_INTERVAL}s")
    
    if not await verify_notebooklm_auth():
        log.warning("Continuing without NotebookLM — will skip audio generation")

    while True:
        try:
            decisions = get_decisions()
            papers = get_cache()
            paper_map = {p["id"]: p for p in papers}

            yes_ids = [aid for aid, dec in decisions.items() if dec == "yes"]
            pending = [aid for aid in yes_ids if not is_queued(aid)]

            if pending:
                log.info(f"Found {len(pending)} new 'yes' paper(s) to process")
                for arxiv_id in pending:
                    paper = paper_map.get(arxiv_id)
                    if not paper:
                        log.warning(f"  Paper {arxiv_id} not in cache, skipping")
                        mark_queued(arxiv_id)  # don't retry missing papers
                        continue

                    notebook_id = await create_notebook_for_paper(paper)
                    mark_queued(arxiv_id)
                    if notebook_id:
                        save_notebook_id(arxiv_id, notebook_id)
                        log.info(f"  ✓ Queued notebook {notebook_id} for {arxiv_id}")
                    else:
                        log.warning(f"  ✗ Failed for {arxiv_id}, will not retry")
            else:
                log.info("No new papers to process")

        except Exception as e:
            log.error(f"Poll error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(poll())
