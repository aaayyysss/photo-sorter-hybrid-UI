import os
import sys
import json
import argparse
import shutil
from pathlib import Path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def hard_link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)  # hardlink
    except Exception:
        shutil.copy2(src, dst)

def symlink_or_copy(src: Path, dst: Path):
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser(description="Apply decisions.json locally (move/copy/link)")
    ap.add_argument("--decisions", required=True, help="decisions.json from server UI")
    ap.add_argument("--inbox", required=True, help="Path to original Inbox folder (where images are)")
    ap.add_argument("--sorted", required=True, help="Target base folder for sorted results")
    ap.add_argument("--mode", choices=["move","copy","link"], default="move", help="File operation")
    args = ap.parse_args()

    with open(args.decisions, "r", encoding="utf-8") as f:
        dec = json.load(f)

    entries = dec.get("entries", [])
    policy = (dec.get("params", {}) or {}).get("multi_face_policy", "copy_all")

    # Group entries by image_id
    by_image = {}
    for e in entries:
        img_id = e.get("image_id")
        if not img_id: 
            continue
        by_image.setdefault(img_id, []).append(e)

    inbox_root = Path(args.inbox).resolve()
    sorted_root = Path(args.sorted).resolve()
    ensure_dir(sorted_root)

    moved = 0
    copied = 0
    linked = 0
    skipped = 0

    for img_id, face_list in by_image.items():
        # find source file; image_id may be absolute or relative
        src_path = Path(img_id)
        if not src_path.is_file():
            # try relative to inbox root
            src_path = inbox_root / img_id
        if not src_path.is_file():
            print(f"[WARN] missing file: {img_id}")
            skipped += 1
            continue

        # Filter accepted
        accepted = [f for f in face_list if f.get("decision") == "accept" and f.get("best_person")]
        if not accepted:
            skipped += 1
            continue

        # For best_single: keep only the highest score
        if policy == "best_single":
            accepted.sort(key=lambda x: (x.get("score") or -1), reverse=True)
            accepted = accepted[:1]

        # Unique persons
        persons = []
        seen = set()
        for f in accepted:
            p = f.get("best_person")
            if p and p not in seen:
                seen.add(p)
                persons.append(p)

        # Perform operation
        for i, person in enumerate(persons):
            dst_dir = sorted_root / person
            ensure_dir(dst_dir)
            dst_path = dst_dir / src_path.name

            if args.mode == "move":
                # If multiple persons, do one move + copies for the rest to avoid losing the file
                if len(persons) == 1 or i == 0:
                    shutil.move(str(src_path), str(dst_path))
                    moved += 1
                    # Update src_path for potential extra persons (file moved now)
                    src_path = dst_path
                else:
                    shutil.copy2(str(src_path), str(dst_path))
                    copied += 1
            elif args.mode == "copy":
                shutil.copy2(str(src_path), str(dst_path))
                copied += 1
            else:  # link
                # Prefer hardlink; fall back to copy
                try:
                    os.link(src_path, dst_path)
                    linked += 1
                except Exception:
                    symlink_or_copy(src_path, dst_path)
                    linked += 1

    print(f"Done. moved={moved}, copied/linked={copied+linked}, skipped={skipped}")
    print(f"Output base: {sorted_root}")

if __name__ == "__main__":
    main()
