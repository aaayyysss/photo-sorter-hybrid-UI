import os, json, pathlib, time, secrets, shutil
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from local_embed import build_app, get_face_vectors, scan_files, imread_utf8, IMG_EXTS
from local_apply_decisions import ensure_dir, symlink_or_copy

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
    allow_origins=["*"],
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
        return {"status": "ok", "message": "registered via server", "server_response": r.json()}

    return {"status": "ok", "message": "computed refs locally", "payload": out}

@app.post("/compute-inbox")
def compute_inbox(payload: dict):
    inbox_path = payload.get("inbox_path", "")
    det_size = int(payload.get("det_size", 640))

    thr = int(payload.get("global_threshold_pct", 32))
    adaptive_on = bool(payload.get("adaptive_on", True))
    adaptive_k = float(payload.get("adaptive_k", 1.0))
    policy = payload.get("multi_face_policy", "copy_all")

    root = pathlib.Path(inbox_path)
    if not root.exists():
        raise HTTPException(400, f"Inbox path not found: {root}")

    face_app = build_app(det_size=(det_size, det_size))

    items = []
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files))
    for fp in files:
        img = imread_utf8(str(fp))
        if img is None:
            continue
        vecs = get_face_vectors(face_app, img, max_faces=None)
        faces = [{"face_id": f"{fp.name}#{i}", "vector": v.tolist()} for i, v in enumerate(vecs)]
        items.append({"image_id": str(fp), "faces": faces})

    if not SERVER_BASE:
        return {"status": "ok", "payload": {"items": items}}

    import requests
    body = {
        "items": items,
        "params": {
            "global_threshold_pct": thr,
            "adaptive_on": adaptive_on,
            "adaptive_k": adaptive_k,
            "multi_face_policy": policy if policy in ("copy_all", "best_single") else "copy_all",
        }
    }
    r = requests.post(f"{SERVER_BASE}/api/sort", headers={"Content-Type": "application/json"}, json=body, timeout=600)
    if r.status_code != 200:
        raise HTTPException(502, f"Server sort failed: {r.text}")
    return {"status": "ok", "message": "sorted via server", "server_response": r.json()}

@app.post("/apply-decisions")
def apply_decisions(payload: dict):
    dec = payload.get("decisions_json", {})
    inbox = pathlib.Path(payload.get("inbox_path", ""))
    sorted_out = pathlib.Path(payload.get("sorted_path", ""))
    mode = payload.get("mode", "move")
    if not inbox.exists():
        raise HTTPException(400, f"Inbox path not found: {inbox}")
    ensure_dir(sorted_out)

    entries = dec.get("entries", [])
    policy = (dec.get("params", {}) or {}).get("multi_face_policy", "copy_all")

    by_image = {}
    for e in entries:
        img_id = e.get("image_id")
        if not img_id: continue
        by_image.setdefault(img_id, []).append(e)

    moved = copied = linked = skipped = 0

    for img_id, face_list in by_image.items():
        src = pathlib.Path(img_id)
        if not src.is_file():
            src = inbox / img_id
        if not src.is_file():
            skipped += 1
            continue

        accepted = [f for f in face_list if f.get("decision") == "accept" and f.get("best_person")]
        if not accepted:
            skipped += 1
            continue

        if policy == "best_single":
            accepted.sort(key=lambda x: (x.get("score") or -1), reverse=True)
            accepted = accepted[:1]

        persons, seen = [], set()
        for f in accepted:
            p = f.get("best_person")
            if p and p not in seen:
                seen.add(p); persons.append(p)

        for i, person in enumerate(persons):
            dst_dir = sorted_out / person
            ensure_dir(dst_dir)
            dst = dst_dir / src.name

            if mode == "move":
                if len(persons) == 1 or i == 0:
                    shutil.move(str(src), str(dst)); moved += 1
                    src = dst
                else:
                    shutil.copy2(str(src), str(dst)); copied += 1
            elif args.mode == "copy":
                shutil.copy2(str(src), str(dst)); copied += 1
            else:  # link
                try:
                    os.link(src, dst); linked += 1
                except Exception:
                    symlink_or_copy(src, dst); linked += 1

    return {"status": "ok", "moved": moved, "copied_or_linked": copied+linked, "skipped": skipped, "sorted_root": str(sorted_out)}

def run():
    print(f"Local Companion listening on {HOST}:{PORT}")
    print(f"Companion Token: {COMPANION_TOKEN}")
    if SERVER_BASE:
        print(f"Server base set to: {SERVER_BASE}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

if __name__ == "__main__":
    run()
