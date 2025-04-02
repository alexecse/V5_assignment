import os
import shutil
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import sync_playwright


def extract_tag_frequency(html_path):
    USELESS_TAGS = {'script', 'style', 'svg', 'link', 'meta', 'noscript', 'defs', 'filter'}
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
    tags = [tag.name for tag in soup.find_all() if tag.name not in USELESS_TAGS]
    return Counter(tags)

def extract_text_content(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
    return soup.get_text(separator=' ', strip=True)

def build_tag_matrix(counters, min_total_freq=5):
    total_freq = Counter()
    for counter in counters:
        total_freq.update(counter)

    filtered_tags = sorted(tag for tag, freq in total_freq.items() if freq >= min_total_freq)
    tag_matrix = []
    for counter in counters:
        row = [counter.get(tag, 0) for tag in filtered_tags]
        tag_matrix.append(row)

    return np.array(tag_matrix), filtered_tags

def chi2_distance_matrix(X, epsilon=1e-10):
    X = X.astype(np.float32)
    n_samples = X.shape[0]
    D = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = 0.5 * np.sum(((X[i] - X[j]) ** 2) / (X[i] + X[j] + epsilon))
            D[i, j] = D[j, i] = d
    return D

def compute_semantic_similarity(html_files):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [extract_text_content(path) for path in html_files]
    embeddings = model.encode(texts, show_progress_bar=True)
    cos_sim = cosine_similarity(embeddings)
    textual_dist = 1 - cos_sim
    textual_dist[textual_dist < 0] = 0
    return textual_dist

def combine_distances_dynamic(chi2_dist, textual_dist, alpha_range=(0.2, 0.8)):
    n = chi2_dist.shape[0]
    combined_dist = np.zeros_like(chi2_dist)
    for i in range(n):
        for j in range(n):
            chi2_val = chi2_dist[i, j]
            text_val = textual_dist[i, j]
            if chi2_val < 1.0:
                alpha = alpha_range[0]
            elif chi2_val > 3.0:
                alpha = alpha_range[1]
            else:
                alpha = np.interp(chi2_val, [1.0, 3.0], alpha_range)
            combined_dist[i, j] = alpha * chi2_val + (1 - alpha) * text_val
    return combined_dist

def generate_screenshot_if_missing(html_path):
    os.makedirs("screenshots", exist_ok=True)
    screenshot_path = os.path.join("screenshots", os.path.basename(html_path) + ".png")
    if not os.path.exists(screenshot_path):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(viewport={"width": 1280, "height": 800})
                page = context.new_page()
                file_url = f"file://{os.path.abspath(html_path)}"
                page.goto(file_url)
                page.wait_for_timeout(1000)
                page.screenshot(path=screenshot_path, full_page=True)
                browser.close()
        except Exception as e:
            print(f"⚠️ Failed to generate screenshot for {html_path}: {e}")

def load_img(path):
    try:
        img = Image.open(path).convert("L").resize((300, 300))
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"⚠️ Failed to load image {path}: {e}")
        return np.zeros((300, 300), dtype=np.uint8)

def postprocessing(labels, distance_matrix, html_files, threshold_merge=20, threshold_attach=3.5):
    n = len(html_files)
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    clusters = dict(clusters)

    outlier_indices = clusters.get(-1, [])
    assigned_outliers = []

    if outlier_indices:
        print("Outliers found:", [os.path.basename(html_files[i]) for i in outlier_indices])
    else:
        print("No outliers found.")

    for idx in outlier_indices:
        dists = []
        for group_label, members in clusters.items():
            if group_label == -1 or not members:
                continue
            avg_dist = np.mean([distance_matrix[idx][m] for m in members])
            dists.append((group_label, avg_dist))

        if not dists:
            continue

        closest_group, min_dist = min(dists, key=lambda x: x[1])
        filename = os.path.basename(html_files[idx])

        print(f"[OUTLIER DEBUG] {filename} → Closest group: {closest_group}")
        print(f"  - Avg distance to group: {min_dist:.4f}")
        print(f"  - Threshold (attach):    {threshold_attach:.4f}")
        diff = min_dist - threshold_attach
        if diff <= 0.05:
            print(f"Outlier is close, (only {diff:.4f} above threshold)")
        elif diff <= 0.15:
            print(f"Consider lowering threshold slightly if you want it attached.")

        if min_dist <= threshold_attach:
            print(f"  ✅ Attached to group {closest_group}\n")
            clusters[closest_group].append(idx)
            assigned_outliers.append(idx)
        else:
            print(f"  ❌ Kept as outlier\n")

    remaining_outliers = [idx for idx in outlier_indices if idx not in assigned_outliers]
    if remaining_outliers:
        clusters[-1] = remaining_outliers
    elif -1 in clusters:
        del clusters[-1]

    group_labels = [l for l in clusters if l != -1]
    merged = set()
    label_mapping = {}

    for i in range(len(group_labels)):
        label_i = group_labels[i]
        if label_i in merged:
            continue
        for j in range(i + 1, len(group_labels)):
            label_j = group_labels[j]
            if label_j in merged:
                continue
            members_i = clusters[label_i]
            members_j = clusters[label_j]
            all_dists = [distance_matrix[a][b] for a in members_i for b in members_j]
            avg_dist = np.mean(all_dists)

            print(f"[MERGE DEBUG] Group {label_i} ↔ Group {label_j} → avg_dist = {avg_dist:.4f}")
            if avg_dist <= threshold_merge:
                f1 = random.choice(members_i)
                f2 = random.choice(members_j)

                html_path1 = html_files[f1]
                html_path2 = html_files[f2]

                generate_screenshot_if_missing(html_path1)
                generate_screenshot_if_missing(html_path2)

                screenshot1 = os.path.join("screenshots", os.path.basename(html_path1) + ".png")
                screenshot2 = os.path.join("screenshots", os.path.basename(html_path2) + ".png")

                img1 = load_img(screenshot1)
                img2 = load_img(screenshot2)

                if np.count_nonzero(img1) == 0 or np.count_nonzero(img2) == 0:
                    print("  ⚠️ Skipped visual merge check — invalid image(s)\n")
                    continue

                sim, _ = ssim(img1, img2, full=True, data_range=255)
                print(f"  [SSIM DEBUG] Visual similarity index = {sim:.4f}")

                if sim > 0.95:
                    clusters[label_i].extend(clusters[label_j])
                    print(f"  ✅ Merged based on visual similarity\n")
                    merged.add(label_j)
                    label_mapping[label_j] = label_i
                else:
                    print(f"  ❌ Not merged — visual similarity too low\n")

    new_labels = np.full(n, -1)
    current_label = 0
    label_map = {}

    for old_label in sorted(clusters.keys()):
        if old_label in merged:
            continue
        if old_label == -1:
            for idx in clusters[old_label]:
                new_labels[idx] = -1
        else:
            for idx in clusters[old_label]:
                new_labels[idx] = current_label
            label_map[old_label] = current_label
            current_label += 1

    return new_labels


def group_similar_htmls(directory, eps, min_samples=2, do_postprocessing=0):
    tier_name = os.path.basename(directory.strip("/"))
    output_dir = os.path.join("output", tier_name)
    stats_dir = os.path.join("statistics", tier_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    html_files = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.endswith('.html') and not f.startswith('._')]

    tag_counters = [extract_tag_frequency(f) for f in tqdm(html_files, desc="Parsing tags")]
    tag_matrix, tags = build_tag_matrix(tag_counters)

    chi2_dist = chi2_distance_matrix(tag_matrix)
    text_dist = compute_semantic_similarity(html_files)

    combined_dist = combine_distances_dynamic(chi2_dist, text_dist)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(combined_dist)

    if do_postprocessing:
        labels = postprocessing(labels, combined_dist, html_files)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(html_files[idx])

    for label, files in clusters.items():
        group_folder = os.path.join(output_dir, f"group_{label}" if label != -1 else "outliers")
        os.makedirs(group_folder, exist_ok=True)
        for file_path in files:
            shutil.copy(file_path, os.path.join(group_folder, os.path.basename(file_path)))

    print(f"Grupuri salvate în {output_dir}: {len([k for k in clusters if k != -1])} grupuri, {len(clusters.get(-1, []))} outlieri")

    filenames = [os.path.basename(f) for f in html_files]
    cluster_labels = ['outlier' if lbl == -1 else f'group_{lbl}' for lbl in labels]
    y_labels = [f"{name} ({group})" for name, group in zip(filenames, cluster_labels)]

    plt.figure(figsize=(14, max(6, len(y_labels) * 0.2)))
    sns.heatmap(tag_matrix, cmap='YlGnBu', xticklabels=tags, yticklabels=y_labels)
    plt.title("HTML Tag Frequency per File")
    plt.xlabel("HTML Tags")
    plt.ylabel("Files")
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{tier_name}_tag_frequency_heatmap.png"))
    plt.close()


if __name__ == "__main__":
    for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
        print(f"\n📂 Grupare pentru: {tier}")
        group_similar_htmls(tier, eps=3, min_samples=2, do_postprocessing=1)
