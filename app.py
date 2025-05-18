# app.py

from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as T

# Import your model definition
from model import KidneyStoneCNN

# Create Flask app
app = Flask(
    __name__,
    static_folder="static",    # serve static assets
    template_folder="templates"  # serve your HTML templates
)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate your model and load weights
model = KidneyStoneCNN().to(device)
state_dict = torch.load("models/kidney_stone_cnn.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Define the same transforms you used during training
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# Routes

@app.route("/")
def index():
    """Landing page with your three cards."""
    return render_template("index.html")


@app.route("/docs")
def docs():
    """Project documentation page."""
    return render_template("docs.html")


@app.route("/real_example")
def real_example():
    """Upload & predict UI."""
    return render_template("real_example.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data POST with key 'image',
    returns JSON: { "prediction": "Stone"|"Non-stone", "confidence": float }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Preprocess
    x = transform(img).unsqueeze(0).to(device)  # shape: [1,3,128,128]

    # Inference
    with torch.no_grad():
        output = model(x).squeeze(0)  # shape: [1]
        prob = output.item()          # between 0 and 1
        label = "Stone" if prob > 0.5 else "Non-stone"

    return jsonify({
        "prediction": label,
        "confidence": round(prob, 3)
    })


if __name__ == "__main__":
    # Running in debug mode for development.
    # In production, use a WSGI server (Gunicorn, Waitress, etc.).
    app.run(debug=True)
