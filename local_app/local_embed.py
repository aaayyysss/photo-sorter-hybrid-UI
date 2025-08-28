import os
import sys
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Force CPU unless user overrides; keeps it simple/portable
os.environ.setdefault("INSIGHTFACE_ONNX_EXECUTION_PROVIDER", "CPUExecutionProvider")

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("Failed to import insightface. Did you pip install -r requirements.txt ?", file=sys.stderr)
    raise

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def imread_utf8(path):
    # robust read (handles non-ascii paths)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def get_face_vectors(app, img_bgr, max_faces=None):
    faces = app.get(img_bgr)
    vecs = []
    for i, f in enumerate(faces):
        if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
            v = np.asarray(f.normed_embedding, dtype=np.float32)
        else:
            v = np.asarray(f.embedding, dtype=np.float32)
            n = np.linalg.norm(v) + 1e-12
            v = v / n
        if v.ndim == 1:
            vecs.append(v)
        if max_faces and len(vecs) >= max_faces:
            break
    return vecs

def build_app(det_size=(640,640)):
    app = FaceAnalysis(name="buffalo_l")
    # ctx_id = -1 => CPU in older insightface; for newer, provider env var is used
    app.prepare(ctx_id=-1, det_size=det_size)
    return app

def scan_files(root):
    root = Path(root)
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))

def cmd_make_refs(args):
    root = Path(args.refs)
    if not root.exists():
        raise SystemExit(f"Refs path not found: {root}")
    people = [p for p in root.iterdir() if p.is_dir()]
    if not people:
        raise SystemExit("No person subfolders found in refs path.")

    app = build_app()
    persons = []
    for person_dir in people:
        pid = person_dir.name
        vectors = []
        files = scan_files(person_dir)
        for fp in tqdm(files, desc=f"Refs:{pid}"):
            img = imread_utf8(str(fp))
            if img is None: 
                continue
            vecs = get_face_vectors(app, img, max_faces=args.max_faces)
            for v in vecs:
                vectors.append(v.tolist())
        if vectors:
            persons.append({"person_id": pid, "vectors": vectors})

    out = {"persons": persons, "mode": args.mode}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} with {sum(len(p['vectors']) for p in persons)} vectors for {len(persons)} persons.")

def cmd_make_inbox(args):
    root = Path(args.inbox)
    if not root.exists():
        raise SystemExit(f"Inbox path not found: {root}")
    files = scan_files(root)
    if not files:
        raise SystemExit("No images found in inbox.")

    app = build_app()
    items = []
    for fp in tqdm(files, desc="Inbox"):
        img = imread_utf8(str(fp))
        if img is None:
            continue
        vecs = get_face_vectors(app, img, max_faces=args.max_faces)
        faces = []
        for i, v in enumerate(vecs):
            faces.append({"face_id": f"{fp.name}#{i}", "vector": v.tolist()})
        items.append({
            "image_id": str(fp),  # absolute or relative path; will be used later by apply script
            "faces": faces
        })

    out = {"items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}: {len(items)} images, {sum(len(x['faces']) for x in items)} faces.")

def main():
    ap = argparse.ArgumentParser(description="Local embeddings builder for Photo Sorter (Hybrid)")
    ap.add_argument("--max-faces", type=int, default=None, help="Limit faces per image (default: all)")
    sub = ap.add_subparsers(dest="cmd")

    ap_refs = sub.add_parser("refs", help="Create refs_register_batch.json from Refs/<person> folders")
    ap_refs.add_argument("--refs", required=True, help="Path to Refs root (each subfolder = person)")
    ap_refs.add_argument("--out", required=True, help="Output JSON file (refs_register_batch.json)")
    ap_refs.add_argument("--mode", choices=["merge","replace"], default="merge", help="Register mode")
    ap_refs.set_defaults(func=cmd_make_refs)

    ap_inb = sub.add_parser("inbox", help="Create inbox_embeddings.json from Inbox folder")
    ap_inb.add_argument("--inbox", required=True, help="Path to Inbox images")
    ap_inb.add_argument("--out", required=True, help="Output JSON file (inbox_embeddings.json)")
    ap_inb.set_defaults(func=cmd_make_inbox)

    # Short aliases
    ap.add_argument("--make-refs", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--make-inbox", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--refs", help=argparse.SUPPRESS)
    ap.add_argument("--inbox", help=argparse.SUPPRESS)

    args = ap.parse_args()

    # Backward-compatible convenience flags
    if args.make_refs:
        if not args.refs or not args.out:
            ap.error("--make-refs requires --refs and --out")
        args.cmd = "refs"
        args.mode = getattr(args, "mode", "merge")
        cmd_make_refs(args)
        return
    if args.make_inbox:
        if not args.inbox or not args.out:
            ap.error("--make-inbox requires --inbox and --out")
        args.cmd = "inbox"
        cmd_make_inbox(args)
        return

    if not args.cmd:
        print("Examples:\n"
              "  python local_embed.py refs --refs D:\\Photos\\Refs --out refs_register_batch.json\n"
              "  python local_embed.py inbox --inbox D:\\Photos\\Inbox --out inbox_embeddings.json",
              file=sys.stderr)
        sys.exit(2)
    args.func(args)

if __name__ == "__main__":
    main()
