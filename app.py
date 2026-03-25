import torch
# Optimization for limited-resource environments (like Render Free Tier)
torch.set_num_threads(1)

import io
import torch.nn.functional as F
from flask import Flask, request, render_template_string
from PIL import Image
from torchvision import transforms
from src.model import build_model

# --- Configuration ---
MODEL_PATH = "models/resnet18_binary.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Feature #2: Carbon Footprint estimations (kg CO2e saved per item)
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
# Using the model builder from your src/model.py
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

# --- Routes ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML_TEMPLATE, error="Dosya yüklenmedi")

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template_string(HTML_TEMPLATE, error="Desteklenmeyen dosya türü.")

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return render_template_string(HTML_TEMPLATE, error="Geçersiz görsel dosyası.")

    # ML Inference Logic
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item()
    if confidence < 0.55:
        item_key, result = "unknown", "geri dönüştürülemez (hayır)"
    else:
        is_recyclable = pred.item() == 1
        item_key = "recyclable_item" if is_recyclable else "non_recyclable_item"
        result = "geri dönüştürülebilir (evet)" if is_recyclable else "geri dönüştürülemez (hayır)"

    info = EXPLANATIONS[item_key]
    
    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        item=info["label"],
        explanation=info["text"],
        confidence=f"{confidence:.2f}",
        co2_saved=info["co2_saved_kg"]
    )

# --- UI Template (Stored at bottom to keep logic clean) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>Geri Dönüşüm Sınıflandırıcısı</title>
  <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Ubuntu', sans-serif; background-color: rgb(18, 95, 18); color: wheat; text-align: center; padding: 40px; }
    h1 { font-size: 60px; margin-bottom: 20px; }
    .upload-label { display: inline-block; background-color: #2f7d2f; padding: 14px 30px; border-radius: 10px; cursor: pointer; margin-top: 20px; transition: 0.3s; }
    .upload-label:hover { background-color: #3b9c3b; }
    input[type="file"] { display: none; }
    button { font-size: 24px; padding: 10px 24px; cursor: pointer; border-radius: 8px; background-color: wheat; color: #125f12; font-weight: bold; border: none; margin-top: 10px; }
    .card { background-color: rgba(0, 0, 0, 0.25); border-radius: 16px; padding: 30px; margin: 40px auto; max-width: 500px; border: 1px solid rgba(255,255,255,0.1); }
    .result-yes { color: #9cff9c; }
    .result-no { color: #ffb3b3; }
    .co2-box { margin-top: 20px; padding: 15px; border: 1px solid #9cff9c; border-radius: 8px; background: rgba(156, 255, 156, 0.1); }
  </style>
</head>
<body>
  <h1>Geri Dönüşüm Sınıflandırıcısı</h1>
  <form action="/predict" method="POST" enctype="multipart/form-data">
    <label class="upload-label">Görsel seç <input type="file" name="file" required></label><br>
    <button type="submit">Analiz et</button>
  </form>

  {% if error %}<p style="color: #ffb3b3; margin-top: 20px;">{{ error }}</p>{% endif %}

  {% if result %}
    <div class="card">
      <h2 class="{{ 'result-yes' if 'evet' in result else 'result-no' }}">{{ result | upper }}</h2>
      <p><strong>Ürün:</strong> {{ item }}</p>
      <p><strong>Güven Skoru:</strong> %{{ (confidence|float * 100)|round(2) }}</p>
      <p><strong>Bilgi:</strong> {{ explanation }}</p>
      
      {% if co2_saved > 0 %}
      <div class="co2-box">
        <p style="margin:0;">🌍 <strong>Tahmini Kurtarılan Karbon:</strong> {{ co2_saved }} kg CO2e</p>
      </div>
      {% endif %}
    </div>
  {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)