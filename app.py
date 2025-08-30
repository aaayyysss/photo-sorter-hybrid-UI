import os
import json
import math
import time
import threading

import tempfile
import shutil

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# -----------------------
# Config
# -----------------------
APP_PORT = int(os.getenv("PORT", "8080"))
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "./data"))
os.makedirs(DATA_DIR, exist_ok=True)

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()  # if empty => no auth required

# -----------------------
# Auth helpers
# -----------------------
def is_admin(req) -> bool:
    if not ADMIN_TOKEN:
        return True
    # Header takes priority
    token = req.headers.get("X-Admin-Token", "") or req.args.get("admin_token", "") or ""
    return token == ADMIN_TOKEN

def require_admin():
    return jsonify({"status": "error", "message": "admin token required"}), 403

# -----------------------
# Storage for refs
# -----------------------
REFS_META_PATH = os.path.join(DATA_DIR, "refs_meta.json")
REFS_NPZ_PATH  = os.path.join(DATA_DIR, "refs_store.npz")

def _safe_key(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)

@dataclass
class PersonStats:
    person_id: str
    n_vectors: int
    dims: int
    mu_pairs: float
    sigma_pairs: float

class RefsStore:
    """
    Holds per-person reference embeddings.
    persons[person_id] = np.ndarray of shape (n, d)
    """
    def __init__(self):
        self.persons: Dict[str, np.ndarray] = {}
        self.dims: Optional[int] = None
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self.persons.clear()
            self.dims = None

    def add_person_vectors(self, person_id: str, vectors: np.ndarray, mode: str = "merge"):
        """
        vectors: np.ndarray (n, d) | if dims unset, set from first add.
        mode: 'merge' or 'replace'
        """
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D (n, d)")
        n, d = vectors.shape
        if n == 0:
            return
        if self.dims is None:
            self.dims = d
        elif self.dims != d:
            raise ValueError(f"dimension mismatch: store={self.dims}, incoming={d}")
        with self._lock:
            if mode == "replace" or person_id not in self.persons:
                self.persons[person_id] = vectors.astype(np.float32, copy=False)
            else:
                if self.persons[person_id].size == 0:
                    self.persons[person_id] = vectors.astype(np.float32, copy=False)
                else:
                    self.persons[person_id] = np.vstack([self.persons[person_id], vectors]).astype(np.float32, copy=False)

    def list_people(self) -> List[PersonStats]:
        out = []
        with self._lock:
            for pid, arr in self.persons.items():
                mu, sig = pairwise_stats(arr)
                out.append(PersonStats(
                    person_id=pid,
                    n_vectors=arr.shape[0],
                    dims=arr.shape[1],
                    mu_pairs=mu,
                    sigma_pairs=sig
                ))
        return out
#----------------

# --- replace RefsStore.save with this version
def save(self):
    with self._lock:
        npz_dict = {}
        meta = {"dims": self.dims, "persons": []}
        for pid, arr in self.persons.items():
            key = _safe_key(pid)
            npz_dict[key] = arr.astype(np.float32, copy=False)
            meta["persons"].append({"person_id": pid, "key": key, "n": int(arr.shape[0])})

        # write NPZ atomically
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=DATA_DIR, suffix=".npz") as tf:
            if npz_dict:
                np.savez_compressed(tf, **npz_dict)
            else:
                np.savez_compressed(tf, _empty=np.array([], dtype=np.float32))
            tmp_npz = tf.name
        shutil.move(tmp_npz, REFS_NPZ_PATH)

        # write JSON atomically
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=DATA_DIR, suffix=".json") as tf:
            json.dump(meta, tf, indent=2)
            tmp_json = tf.name
        shutil.move(tmp_json, REFS_META_PATH)

def load(self):
    self.clear()
    if not os.path.exists(REFS_META_PATH) or not os.path.exists(REFS_NPZ_PATH):
        return
    try:
        with open(REFS_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dims = meta.get("dims")
        with np.load(REFS_NPZ_PATH, allow_pickle=False) as npz:
            persons = {}
            for p in meta.get("persons", []):
                pid = p["person_id"]
                key = p["key"]
                if key not in npz:
                    continue
                arr = npz[key]
                persons[pid] = arr
        self.persons = persons
        self.dims = dims
    except Exception as e:
        # leave store empty if corrupted; log message
        print(f"[WARN] Failed to load refs: {e}", flush=True)

#Replaced by enhance version above-------------
#    def save(self):
#        with self._lock:
#            npz_dict = {}
#            meta = {"dims": self.dims, "persons": []}
#            for pid, arr in self.persons.items():
#                key = _safe_key(pid)
#                npz_dict[key] = arr.astype(np.float32, copy=False)
#                meta["persons"].append({"person_id": pid, "key": key, "n": int(arr.shape[0])})
#            if npz_dict:
#                np.savez_compressed(REFS_NPZ_PATH, **npz_dict)
#            else:
#                # write empty npz
#                np.savez_compressed(REFS_NPZ_PATH, _empty=np.array([], dtype=np.float32))
#            with open(REFS_META_PATH, "w", encoding="utf-8") as f:
#                json.dump(meta, f, indent=2)
#
#    def load(self):
#        self.clear()
#        if not os.path.exists(REFS_META_PATH) or not os.path.exists(REFS_NPZ_PATH):
#            return
#        with open(REFS_META_PATH, "r", encoding="utf-8") as f:
#            meta = json.load(f)
#        dims = meta.get("dims")
#        npz = np.load(REFS_NPZ_PATH)
#        persons = {}
#        for p in meta.get("persons", []):
#            pid = p["person_id"]
#            key = p["key"]
#            arr = npz[key]
#            persons[pid] = arr
#        self.persons = persons
#        self.dims = dims

# -----------------------
# Similarity + thresholds
# -----------------------
def l2norm(v: np.ndarray) -> np.ndarray:
    # Normalize rows to unit length
    eps = 1e-12
    norms = np.linalg.norm(v, axis=1, keepdims=True) + eps
    return v / norms

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # assumes both are 1D unit vectors
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def pairwise_stats(vectors: np.ndarray) -> Tuple[float, float]:
    """
    Pairwise cosine among refs to estimate distribution (mu, sigma).
    If <2 vectors, return fallback.
    """
    n = vectors.shape[0]
    if n < 2:
        return 0.90, 0.05
    V = l2norm(vectors.astype(np.float32, copy=False))
    sims = []
    for i in range(n):
        # dot with following
        s = np.dot(V[i+1:], V[i])
        if s.size:
            sims.extend(s.tolist())
    if not sims:
        return 0.90, 0.05
    sims_np = np.array(sims, dtype=np.float32)
    return float(sims_np.mean()), float(sims_np.std(ddof=1) if sims_np.size > 1 else 0.05)

def best_score_against_person(vec_u: np.ndarray, refs_u: np.ndarray) -> float:
    """
    vec_u: (d,) unit vector
    refs_u: (n, d) unit vectors
    Returns max cosine across refs
    """
    if refs_u.size == 0:
        return -1.0
    sims = refs_u @ vec_u  # (n,)
    return float(np.clip(np.max(sims), -1.0, 1.0))

def compute_threshold(global_pct: int, adaptive_on: bool, mu: float, sigma: float, adaptive_k: float) -> float:
    g = max(0, min(100, int(global_pct))) / 100.0
    if not adaptive_on:
        return g
    return max(g, mu - adaptive_k * sigma)

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB JSON limit
CORS(app, resources={r"/api/*": {"origins": "*"}})
store = RefsStore()
store.load()

@app.route("/")
def index():
    return app.send_static_file("index.html")  # index.html is served from /static for simplicity

#@app.route("/api/health")
#def health():
#    ppl = store.list_people()
#    return jsonify({
#        "status": "ok",
#        "people": [asdict(p) for p in ppl],
#        "dims": store.dims
#    })

# Health endpoint: include admin requirement + counts
# Makes it clearer in the UI/logs.

@app.route("/api/health")
def health():
    ppl = store.list_people()
    return jsonify({
        "status": "ok",
        "people": [asdict(p) for p in ppl],
        "dims": store.dims,
        "admin_required": bool(ADMIN_TOKEN),
        "n_persons": len(ppl),
        "timestamp": int(time.time())
    })



# --------- Refs Management (Power user) ----------

@app.route("/api/refs/register", methods=["POST"])
def refs_register():
    if not is_admin(request):
        return require_admin()

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status":"error","message":"JSON body required"}), 400

    person_id = payload.get("person_id", "").strip()
    vectors = payload.get("vectors", [])
    mode = payload.get("mode", "merge")
    if not person_id or not isinstance(vectors, list) or len(vectors) == 0:
        return jsonify({"status":"error","message":"person_id and non-empty vectors[] required"}), 400

#    arr = np.array(vectors, dtype=np.float32)
#    if arr.ndim != 2:
#        return jsonify({"status":"error","message":"vectors must be 2D: [[...],[...]]"}), 400

#Validate incoming vectors (NaNs / shape) and clamp sizes
#Helps avoid weird payloads or accidental megabyte JSONs.
    arr = np.array(vectors, dtype=np.float32)
    if arr.ndim != 2:
        return jsonify({"status":"error","message":"vectors must be 2D: [[...],[...]]"}), 400
    if not np.isfinite(arr).all():
        return jsonify({"status":"error","message":"vectors contain NaN/Inf"}), 400
    if arr.shape[1] > 2048:
        return jsonify({"status":"error","message":"vector dimension too large"}), 400

    # Normalize refs once to unit
    arr_u = l2norm(arr)
    try:
        store.add_person_vectors(person_id, arr_u, mode=mode)
        store.save()
    except ValueError as e:
        return jsonify({"status":"error","message":str(e)}), 400

    stats = store.list_people()
    return jsonify({"status":"success","message":f"registered {arr.shape[0]} vectors for {person_id} ({mode})",
                    "people":[asdict(p) for p in stats]})

@app.route("/api/refs/register-batch", methods=["POST"])
def refs_register_batch():
    if not is_admin(request):
        return require_admin()

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status":"error","message":"JSON body required"}), 400

    persons = payload.get("persons", [])
    mode = payload.get("mode", "merge")
    if not isinstance(persons, list) or not persons:
        return jsonify({"status":"error","message":"persons[] required"}), 400

    added = 0
    try:
        for p in persons:
            pid = p.get("person_id","").strip()
            vecs = p.get("vectors", [])
            if not pid or not vecs:
                continue
           
            #arr = np.array(vecs, dtype=np.float32)
            #if arr.ndim != 2:
            #    continue
            #store.add_person_vectors(pid, l2norm(arr), mode=mode)

            #Validate incoming vectors (NaNs / shape) and clamp sizes
            #Helps avoid weird payloads or accidental megabyte JSONs.
    
            arr = np.array(vecs, dtype=np.float32)
            if arr.ndim != 2 or not np.isfinite(arr).all():
                continue
            if arr.shape[1] > 2048:
                continue
            store.add_person_vectors(pid, l2norm(arr), mode=mode)

            added += arr.shape[0]
        store.save()
    except ValueError as e:
        return jsonify({"status":"error","message":str(e)}), 400

    stats = store.list_people()
    return jsonify({"status":"success","message":f"added {added} vectors","people":[asdict(s) for s in stats]})

@app.route("/api/refs/clear", methods=["POST"])
def refs_clear():
    if not is_admin(request):
        return require_admin()
    store.clear()
    store.save()
    return jsonify({"status":"success","message":"references cleared"})

@app.route("/api/refs/export", methods=["GET"])
def refs_export():
    # produce a downloadable JSON of refs (warning: large)
    out = {"dims": store.dims, "persons": []}
    for s in store.list_people():
        arr = store.persons.get(s.person_id, np.zeros((0, store.dims or 0), dtype=np.float32))
        out["persons"].append({
            "person_id": s.person_id,
            "vectors": arr.tolist()
        })
    # stream as a file-like download
    tmp_path = os.path.join(DATA_DIR, f"refs_export_{int(time.time())}.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return send_file(tmp_path, mimetype="application/json", as_attachment=True, download_name="refs_export.json")

# --------- Sorting (Normal user) ----------

@app.route("/api/sort", methods=["POST"])
def sort_api():
    """
    Request body:
    {
      "items": [
        {
          "image_id": "IMG_001.jpg",
          "faces": [
            {"face_id":"IMG_001#0","vector":[... 512 ...]},
            {"face_id":"IMG_001#1","vector":[...]}
          ]
        },
        ...
      ],
      "params": {
        "global_threshold_pct": 32,
        "adaptive_on": true,
        "adaptive_k": 1.0,
        "multi_face_policy": "copy_all" | "best_single"
      }
    }
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status":"error","message":"JSON body required"}), 400

    items = payload.get("items", [])
    params = payload.get("params", {}) or {}
    if not isinstance(items, list) or not items:
        return jsonify({"status":"error","message":"items[] with embeddings required"}), 400

    gthr = int(params.get("global_threshold_pct", 32))
    adaptive_on = bool(params.get("adaptive_on", False))
    adaptive_k  = float(params.get("adaptive_k", 1.0))
    policy = params.get("multi_face_policy", "copy_all")
    policy = policy if policy in ("copy_all", "best_single") else "copy_all"

    # Build per-person stats once
    people = store.list_people()
    if not people:
        return jsonify({"status":"error","message":"no references available on server"}), 400

    # cache normalized refs
    refs_cache: Dict[str, np.ndarray] = {}
    stats_map: Dict[str, Tuple[float,float]] = {}
    for s in people:
        arr = store.persons[s.person_id]
        refs_cache[s.person_id] = arr  # already unit
        stats_map[s.person_id] = (s.mu_pairs, s.sigma_pairs)

    entries = []
    n_faces = 0
    for it in items:
        image_id = it.get("image_id", "")
        faces = it.get("faces", [])
        if not image_id or not isinstance(faces, list):
            continue

        # Collect candidates for each face
        face_results = []
        for f in faces:
            vec = np.array(f.get("vector", []), dtype=np.float32)
            face_id = f.get("face_id") or f"{image_id}#{len(face_results)}"
            if vec.ndim != 1 or (store.dims and vec.shape[0] != store.dims):
                face_results.append({
                    "image_id": image_id, "face_id": face_id,
                    "decision": "invalid_vector", "score": None, "best_person": None
                })
            continue
            if not np.isfinite(vec).all():
                face_results.append({
                    "image_id": image_id, "face_id": face_id,
                    "decision": "invalid_vector", "score": None, "best_person": None
                })
            continue

            
            # normalize
            u = vec / (np.linalg.norm(vec) + 1e-12)

            # score against all persons (max-of-refs)
            best_person = None
            best_score = -2.0
            alts = []
            for pid, rarr in refs_cache.items():
                sc = best_score_against_person(u, rarr)
                alts.append((pid, sc))
                if sc > best_score:
                    best_score = sc
                    best_person = pid

            # compute threshold for best_person
            mu, sig = stats_map.get(best_person, (0.90, 0.05))
            thr = compute_threshold(gthr, adaptive_on, mu, sig, adaptive_k)
            above = best_score >= thr

            face_results.append({
                "image_id": image_id,
                "face_id": face_id,
                "best_person": best_person,
                "score": float(best_score),
                "threshold": float(thr),
                "above": bool(above),
                "alternatives": sorted(
                    [{"person": p, "score": float(s)} for (p, s) in alts],
                    key=lambda x: x["score"], reverse=True
                )[:5],
                "decision": "accept" if above else "reject"
            })
            n_faces += 1

        # Apply multi-face policy if needed
        if policy == "best_single":
            # keep only the highest-score accepted; reject others
            accepted = [fr for fr in face_results if fr.get("decision") == "accept"]
            if accepted:
                top = max(accepted, key=lambda x: x["score"])
                for fr in face_results:
                    if fr is not top:
                        fr["decision"] = "reject"
            # else: all rejected already
        # else copy_all => keep as is

        entries.extend(face_results)

    summary = {
        "n_images": len(items),
        "n_faces": n_faces,
        "global_threshold": max(0, min(100, gthr)) / 100.0,
        "adaptive_on": adaptive_on,
        "adaptive_k": adaptive_k,
        "multi_face_policy": policy
    }

    return jsonify({"status":"success", "summary": summary, "entries": entries})

# -----------------------
# Static UI (served from /static)
# -----------------------
# we serve index.html from /static to simplify (no templating)

#if __name__ == "__main__":
#    print(f"Hybrid server listening on 0.0.0.0:{APP_PORT}")
#    app.run(host="0.0.0.0", port=APP_PORT)


if __name__ == "__main__":
    print(f"Hybrid server listening on 0.0.0.0:{APP_PORT}")
    print(f"DATA_DIR={DATA_DIR} | ADMIN_TOKEN={'set' if ADMIN_TOKEN else 'not set'}", flush=True)
    app.run(host="0.0.0.0", port=APP_PORT)

