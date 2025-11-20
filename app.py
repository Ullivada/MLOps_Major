from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for

from mlops_major.preprocess import image_bytes_to_vector

ARTIFACT_DIR = Path("artifacts")
MODEL_FILE = ARTIFACT_DIR / "savedmodel.pth"
METADATA_FILE = ARTIFACT_DIR / "model_metadata.json"


def load_model() -> Any:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Run `python train.py` before starting the server."
        )
    return joblib.load(MODEL_FILE)


def load_metadata() -> dict[str, Any]:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {"target_names": list(range(40))}


app = Flask(__name__)
app.secret_key = "mlops-major-secret"
model = load_model()
metadata = load_metadata()
target_names = metadata.get("target_names", list(range(40)))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image file before submitting.")
            return redirect(url_for("index"))
        try:
            vector = image_bytes_to_vector(file.read())
            predicted_class = model.predict(np.array([vector]))[0]
            prediction = {
                "label": int(predicted_class),
                "target_names": target_names,
            }
        except Exception as exc:  # pragma: no cover - user feedback path
            flash(f"Failed to process image: {exc}")
            return redirect(url_for("index"))

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

