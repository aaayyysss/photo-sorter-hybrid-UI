import os
import json
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

# App + Socket
app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

# Placeholder: Here you'd import your model/embedding code
# from photo_sorter import compute_embeddings, match_faces

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/refs/register", methods=["POST"])
def register_refs():
    """
    In hybrid mode: expect metadata or embeddings, not full images.
    """
    data = request.json
    # TODO: save embeddings to DB or memory
    return jsonify({"status": "success", "message": "Reference embeddings registered."})

@app.route("/sort", methods=["POST"])
def sort_photos():
    """
    In hybrid mode: server matches embeddings only.
    """
    data = request.json
    # TODO: run your face matching logic here
    result = {
        "status": "success",
        "sorted": [
            {"file": "img001.jpg", "person": "Alice"},
            {"file": "img002.jpg", "person": "Bob"}
        ]
    }
    return jsonify(result)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)