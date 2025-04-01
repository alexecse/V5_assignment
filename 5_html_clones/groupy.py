import os
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter, Counter as CountLabels


# Extract frequency of HTML tags from a file
def extract_tag_frequency(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Parse the HTML content using the lxml parser
        soup = BeautifulSoup(f, 'lxml')

    tags = [tag.name for tag in soup.find_all()]
    return Counter(tags)

# Build a tag frequency matrix for a list of HTML files
def build_tag_matrix(counters, min_total_freq = 5):
    total_freq = Counter()
    for counter in counters:
        total_freq.update(counter)

    # Filter out tags that appear too infrequently
    filtered_tags = sorted(tag for tag, freq in total_freq.items() if freq >= min_total_freq)

    # Rebuild the matrix only using the selected tags
    tag_matrix = []
    for counter in counters:
        row = [counter.get(tag, 0) for tag in filtered_tags]
        tag_matrix.append(row)

    return np.array(tag_matrix), filtered_tags

# Compute pairwise Chi-squared distances between frequency vectors
def chi2_distance_matrix(X, epsilon=1e-10):
    # Convert input to float for precision
    X = X.astype(np.float32)
    n_samples = X.shape[0]

    D = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Compute Chi-squared distance between each pair of rows
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = 0.5 * np.sum(((X[i] - X[j]) ** 2) / (X[i] + X[j] + epsilon))
            D[i, j] = D[j, i] = d  # symmetric matrix
    return D

def group_similar_htmls(directory, eps=0.5, min_samples=2):
    # Create the output/ and statistics/ directories
    tier_name = os.path.basename(directory.strip("/"))
    output_dir = os.path.join("output", tier_name)
    stats_dir = os.path.join("statistics", tier_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Load and parse HTML files
    html_files = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.endswith('.html') and not f.startswith('._')]
    counters = [extract_tag_frequency(f) for f in tqdm(html_files, desc=f"Parsing {directory}")]
    matrix, tags = build_tag_matrix(counters)

    # Compute Chi² distance matrix
    chi2_dist = chi2_distance_matrix(matrix)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(chi2_dist)

    # Prepare heatmap labels (filename + group)
    filenames = [os.path.basename(f) for f in html_files]
    cluster_labels = ['outlier' if lbl == -1 else f'group_{lbl}' for lbl in labels]
    y_labels = [f"{name} ({group})" for name, group in zip(filenames, cluster_labels)]

    # Heatmap with tag frequency and cluster labels
    plt.figure(figsize=(14, max(6, len(y_labels) * 0.2)))
    sns.heatmap(matrix, cmap='YlGnBu', xticklabels=tags, yticklabels=y_labels)
    plt.title("HTML Tag Frequency per File")
    plt.xlabel("HTML Tags")
    plt.ylabel("Files")
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{tier_name}_tag_frequency_heatmap.png"))
    plt.close()

    # Group HTML files into folders
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(html_files[idx])

    grouped_count = 0
    for label, files in clusters.items():
        if label == -1:
            group_folder = os.path.join(output_dir, "outliers")
        else:
            group_folder = os.path.join(output_dir, f"group_{label}")
        os.makedirs(group_folder, exist_ok=True)

        for file_path in files:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(group_folder, filename))

        if label != -1:
            grouped_count += 1

    total_files = len(html_files)
    outlier_count = len(clusters.get(-1, []))
    print(f"Saved groups in {output_dir}: {grouped_count} groups + {outlier_count}/{total_files} outliers")


if __name__ == "__main__":
    for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
        print(f"Grouping for: {tier}")
        group_similar_htmls(tier, eps=1.0, min_samples=2)
