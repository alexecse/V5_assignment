import os
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
from playwright.sync_api import sync_playwright

# Load EfficientNetB0 model and preprocessing pipeline
visual_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def image_embedding(path):
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = visual_model.features(img_tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return pooled.view(-1).numpy()
    except Exception as e:
        print("⚠️ Fallback image embedding failed:", e)
        return np.zeros(1280)

def generate_screenshot_if_missing(html_path, output_folder="screenshots"):
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(html_path) + ".png"
    output_path = os.path.join(output_folder, filename)
    if os.path.exists(output_path):
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("file://" + os.path.abspath(html_path), wait_until="load", timeout=10000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
    except Exception as e:
        print(f"Could not generate screenshot for {html_path}: {e}")
