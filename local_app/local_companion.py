import os, json, pathlib, time, secrets
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Reuse existing local logic
from local_embed import build_app, get_face_vectors, scan_files, imread_utf8, IMG_EXTS
from local_apply_decisions import ensure_dir, symlink_or_copy
import numpy as np
import shutil

HOST = "127.0.0.1"
PORT = 8765
STATE_DIR = pathlib.Path.home() / ".photo-sorter"
STATE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_PATH = STATE_DIR / "companion_token"

SERVER_BASE = os.environ.get("SERVER_BASE", "").rstrip("/")  # e.g. http://<ip>:8080
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

def _get_or_create_token() -> str:
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    tok = secrets.token_urlsafe(24)
    TOKEN_PATH.write_text(tok)
    return tok

COMPANION_TOKEN = _get_or_create_token()

app = FastAPI(title="Photo Sorter Local Companion")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # token will protect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def token_guard(request: Request, call_next):
    if request.url.path == "/status":
        return await call_next(request)
    tok = request.headers.get("X-Companion-Token", "")
    if tok != COMPANION_TOKEN:
        return JSONResponse({"status": "error", "message": "invalid token"}, status_code=403)
    return await call_next(request)

@app.get("/status")
def status():
    return {
        "status": "ok",
        "token_required": True,
        "server_base": SERVER_BASE,
        "ts": int(time.time()),
        "hint": "Paste this token into the web UI: " + COMPANION_TOKEN
    }

@app.post("/compute-refs")
def compute_refs(payload: dict):
    refs_path = payload.get("refs_path", "")
    mode = payload.get("mode", "merge")
    det_size = int(payload.get("det_size", 640))
    root = pathlib.Path(refs_path)
    if not root.exists():
        raise HTTPException(400, f"Refs path not found: {root}")

    face_app = build_app(det_size=(det_size, det_size))

    persons = []
    for person_dir in [p for p in root.iterdir() if p.is_dir()]:
        pid = person_dir.name
        vectors: List[List[float]] = []
        files = scan_files(person_dir)
        for fp in files:
            img = imread_utf8(str(fp))
            if img is None:
                continue
            vecs = get_face_vectors(face_app, img, max_faces=None)
            for v in vecs:
                vectors.append(v.tolist())
        if vectors:
            persons.append({"person_id": pid, "vectors": vectors})

    out = {"persons": persons, "mode": mode}
    if not persons:
        return {"status": "ok", "message": "no faces found", "payload": out}

    if SERVER_BASE:
        import requests
        headers = {"Content-Type": "application/json"}
        if ADMIN_TOKEN:
            headers["X-Admin-Token"] = ADMIN_TOKEN
        r = requests.post(f"{SERVER_BASE}/api/refs/register-batch", headers=headers, json=out, timeout=120)
        if r.status_code != 200:
            raise HTTPException(502, f"Server register failed: {r.text}")
        return {"status": "ok", "message": "registered via server", "server_response": r.json()
