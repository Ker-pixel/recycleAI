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

# --- UI Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VisionRecycle | AI Classifier</title>
  <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-color: #070707;
      --card-bg: #121212;
      --volt: #ccff00;
      --volt-hover: #b3e600;
      --text-main: #f4f4f4;
      --text-muted: #888888;
      --border-color: #262626;
      --error: #ff3366;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body { 
      font-family: 'Ubuntu', sans-serif; 
      background-color: var(--bg-color); 
      color: var(--text-main); 
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
      background-image: radial-gradient(circle at 50% 0%, rgba(204, 255, 0, 0.05) 0%, transparent 60%);
    }

    .container {
      width: 100%;
      max-width: 520px;
      text-align: center;
    }

    .logo {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 3px;
      color: var(--volt);
      font-weight: 700;
      margin-bottom: 12px;
    }

    h1 { 
      font-size: 40px; 
      font-weight: 700;
      margin-bottom: 10px;
      letter-spacing: -1px;
    }

    .subtitle {
      color: var(--text-muted);
      font-size: 16px;
      margin-bottom: 40px;
      font-weight: 300;
    }

    .upload-area {
      background-color: var(--card-bg);
      border: 1px dashed var(--border-color);
      border-radius: 16px;
      padding: 40px 20px;
      margin-bottom: 20px;
      transition: all 0.3s ease;
    }

    .upload-area:hover {
      border-color: var(--volt);
      box-shadow: 0 0 20px rgba(204, 255, 0, 0.05);
    }

    .upload-label { 
      display: inline-block; 
      background-color: transparent; 
      color: var(--volt);
      border: 1px solid var(--volt);
      padding: 14px 32px; 
      border-radius: 8px; 
      cursor: pointer; 
      font-weight: 500;
      font-size: 14px;
      transition: 0.3s; 
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .upload-label:hover { 
      background-color: var(--volt); 
      color: #000;
      box-shadow: 0 0 15px rgba(204, 255, 0, 0.4);
    }

    input[type="file"] { display: none; }

    button { 
      width: 100%;
      font-family: 'Ubuntu', sans-serif;
      font-size: 16px; 
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 18px 24px; 
      cursor: pointer; 
      border-radius: 12px; 
      background-color: var(--volt); 
      color: #000; 
      font-weight: 700; 
      border: none; 
      transition: all 0.3s ease;
    }

    button:hover {
      background-color: var(--volt-hover);
      box-shadow: 0 0 25px rgba(204, 255, 0, 0.3);
      transform: translateY(-2px);
    }

    .card { 
      background-color: var(--card-bg); 
      border-radius: 16px; 
      padding: 35px; 
      margin-top: 35px; 
      border: 1px solid var(--border-color); 
      text-align: left;
      box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    .card h2 {
      font-size: 22px;
      margin-bottom: 25px;
      padding-bottom: 15px;
      border-bottom: 1px solid var(--border-color);
      letter-spacing: 1px;
    }

    .result-yes { color: var(--volt); }
    .result-no { color: var(--error); }

    .stat-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 15px;
      font-size: 15px;
    }

    .stat-label { color: var(--text-muted); }
    .stat-value { font-weight: 500; }

    .co2-box { 
      margin-top: 30px; 
      padding: 18px; 
      border: 1px solid rgba(204, 255, 0, 0.3); 
      border-radius: 12px; 
      background: rgba(204, 255, 0, 0.05); 
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .co2-box p {
      margin: 0;
      font-size: 14px;
      color: var(--volt);
    }

    .error-msg {
      color: var(--error);
      margin-top: 20px;
      font-size: 14px;
      background: rgba(255, 51, 102, 0.1);
      padding: 15px;
      border-radius: 12px;
      border: 1px solid rgba(255, 51, 102, 0.2);
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">VisionRecycle AI</div>
    <h1>Geri Dönüşüm Sınıflandırıcısı</h1>
    <p class="subtitle">Yapay zeka destekli atık analizi ve karbon ayak izi tahmini</p>

    <form action="/predict" method="POST" enctype="multipart/form-data">
      <div class="upload-area">
        <label class="upload-label">
          Görsel Seç
          <input type="file" name="file" required onchange="document.getElementById('file-name').textContent = this.files[0].name;">
        </label>
        <p id="file-name" style="margin-top: 15px; font-size: 13px; color: var(--text-muted);"></p>
      </div>
      <button type="submit">Analiz Et</button>
    </form>

    {% if error %}
      <div class="error-msg">{{ error }}</div>
    {% endif %}

    {% if result %}
      <div class="card">
        <h2 class="{{ 'result-yes' if 'evet' in result else 'result-no' }}">SONUÇ: {{ result | upper }}</h2>
        
        <div class="stat-row">
          <span class="stat-label">Tespit Edilen Ürün</span>
          <span class="stat-value">{{ item }}</span>
        </div>
        
        <div class="stat-row">
          <span class="stat-label">Model Güven Skoru</span>
          <span class="stat-value">%{{ (confidence|float * 100)|round(2) }}</span>
        </div>
        
        <div class="stat-row" style="flex-direction: column; gap: 8px; margin-top: 25px;">
          <span class="stat-label">Analiz Özeti</span>
          <span class="stat-value" style="line-height: 1.6; color: var(--text-main);">{{ explanation }}</span>
        </div>
        
        {% if co2_saved > 0 %}
        <div class="co2-box">
          <span style="font-size: 18px;">⚡</span>
          <p><strong>Tahmini Kurtarılan Karbon:</strong> {{ co2_saved }} kg CO2e</p>
        </div>
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)