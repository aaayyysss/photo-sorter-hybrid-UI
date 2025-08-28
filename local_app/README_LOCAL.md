# Local App (Hybrid)

This app computes face embeddings on your machine and only sends vectors to the server.

## 1) Install
- Use Python 3.10.x or 3.11 (3.10.x recommended).
- `pip install -r requirements.txt`

The first run downloads InsightFace `buffalo_l` models to `~/.insightface/models`.

## 2) Build reference vectors
Each sub-folder under `--refs` is a person:
