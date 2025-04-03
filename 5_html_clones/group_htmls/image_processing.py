import os
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
from playwright.sync_api import sync_playwright

# Load a pre-trained model, set to evaluation mode
visual_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()

# Define preprocessing steps for image input
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),      # Resize image to expected input shape
    transforms.ToTensor(),              # Convert to tensor and scale to [0, 1]
])

def image_embedding(path):
    # Extracts a feature vector from a screenshot image using EfficientNet-B0.
    # It returns a np.ndarray: 1280-dimensional feature vector as NumPy array.
    try:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        # Forward pass through EfficientNet features
        with torch.no_grad():
            features = visual_model.features(img_tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Global average pooling
            return pooled.view(-1).numpy()  # Flatten to 1D vector

    except Exception as e:
        print(" Fallback image embedding failed:", e)
        return np.zeros(1280)  # Return zero-vector as fallback


def generate_screenshot_if_missing(html_path, output_folder="screenshots"):
    # Generates a full-page screenshot for a given HTML file using Playwright if it doesn't already exist.
    
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(html_path) + ".png"
    output_path = os.path.join(output_folder, filename)

    # Skip generation if screenshot already exists
    if os.path.exists(output_path):
        return

    try:
        # Launch headless browser and render the HTML page
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("file://" + os.path.abspath(html_path), wait_until="load", timeout=10000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
    except Exception as e:
        print(f"Could not generate screenshot for {html_path}: {e}")