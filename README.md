# Photo Sorter Hybrid UI

This is the **Hybrid version** of the Photo Sorter app.

## Architecture
- **Server (Oracle VM / Cloud):**
  - Hosts face embedding + matching logic
  - Lightweight API: `/refs/register`, `/sort`
- **Local PC:**
  - Stores actual photos
  - Sends only embeddings/metadata to server
  - Handles final file operations

## Run locally (development)
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
