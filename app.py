import torch
torch.set_num_threads(1)

import io
import torch.nn.functional as F
from flask import Flask, request, render_template
from PIL import Image
from torchvision import transforms
from src.model import build_model

# --- Configuration ---
MODEL_PATH = "models/resnet18_binary.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Feature #2: Carbon Footprint estimations (in kg CO2e saved per item)
EXPLANATIONS = {
    "recyclable_item": {
        "label": "Geri dönüştürülebilir ürün",
        "text": "Bu ürün geri dönüştürülebilir görünüyor. Lütfen temizleyip atın.",
        "co2_saved_kg": 0.05
    },
    "non_recyclable_item": {
        "label": "Geri dönüştürülemez ürün",
        "text": "Bu ürün normal geri dönüşüme uygun değildir.",
        "co2_saved_kg": 0.0
    },
    "unknown": {
        "label": "Bilinmeyen ürün",
        "text": "Ürün tam olarak tanımlanamadı. Emin değilseniz genel atığa atın.",
        "co2_saved_kg": 0.0
    }
}

app = Flask(__name__)

# --- Load Model ---
device = torch.device("cpu")
model = build_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="Dosya yüklenmedi")

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template("index.html", error="Desteklenmeyen dosya türü.")

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return render_template("index.html", error="Geçersiz görsel dosyası.")

    # ML Inference
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item()
    if confidence < 0.55:
        item_key, result = "unknown", "geri dönüştürülemez (hayır)"
    else:
        item_key = "recyclable_item" if pred.item() == 1 else "non_recyclable_item"
        result = "geri dönüştürülebilir (evet)" if pred.item() == 1 else "geri dönüştürülemez (hayır)"

    info = EXPLANATIONS[item_key]
    return render_template(
        "index.html",
        result=result,
        item=info["label"],
        explanation=info["text"],
        confidence=f"{confidence:.2f}",
        co2_saved=info["co2_saved_kg"]
    )

if __name__ == "__main__":
    app.run(debug=True)