# 📸 Photo Sorter — Hybrid UI

A **hybrid face-photo sorting system** where:
- **Embeddings are computed locally** (no heavy image uploads).
- **Server manages reference vectors** and performs classification.
- **UI supports two roles**:
  - **Normal user** → Sort inbox embeddings JSONs.
  - **Power user** → Manage reference embeddings with optional Admin Token.
- **Local client** applies results (move/copy/link) directly on your machine.

---

## 🚀 Features
- Server (Flask): 
  - Stores per-person reference embeddings.
  - Supports global + adaptive thresholds.
  - Multi-face policy (`copy_all` / `best_single`).
  - Admin token for restricted endpoints.
- Web UI:
  - Upload embeddings JSONs.
  - Manage references (register/clear/export).
  - Download `decisions.json`.
- Local Client:
  - Compute embeddings locally via **InsightFace buffalo_l**.
  - Generate refs + inbox JSONs.
  - Apply `decisions.json` to move/copy/link files.

---

## 📂 Repository Structure
photo-sorter-hybrid-UI/
├── app.py                          # Flask server
├── requirements.txt                # Server requirements
├── data/                           # Created automatically (refs, npz, etc.)
├── static/
│   ├── index.html                  # Web UI (Normal + Power users)
│   ├── app.js                      # UI logic
│   └── style.css                   # UI styles
├── local_app/
│   ├── README_LOCAL.md             # Local client instructions
│   ├── requirements.txt            # Local app dependencies
│   ├── local_embed.py              # Build refs + inbox embeddings
│   └── local_apply_decisions.py    # Apply decisions.json locally
└── README.md                       # Main repo documentation (below)


