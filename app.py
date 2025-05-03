from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from model import load_model, predict_image, generate_gradcam

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcams"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return "Please upload via the main page."

    if "image" not in request.files:
        return "No file part", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # 1. Predict with model
    label, confidence = predict_image(model, file_path)

    # 2. Generate Grad-CAM
    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
    generate_gradcam(model, file_path, label, gradcam_path)

    # 3. Prepare a human-readable diagnosis
    if label == "glaucoma":
        diagnosis = f"⚠️ Signs of Glaucoma detected with confidence {confidence*100:.2f}%."
    else:
        diagnosis = "✅ No signs of Glaucoma detected."

    # 4. Performance metrics (example)
    metrics = {
        "accuracy": 97.28,
        "f1_score": 0.95,
        "auc_roc": 0.997,
    }

    return render_template(
        "result.html",
        uploaded_image=url_for('static', filename=f"uploads/{filename}"),
        gradcam_image=url_for('static', filename=f"gradcams/{gradcam_filename}"),
        prediction=label,
        confidence=f"{confidence*100:.2f}%",
        diagnosis=diagnosis,
        metrics=metrics
    )

if __name__ == "__main__":
    app.run(debug=True)
