import os                              # File and directory operations
from bs4 import BeautifulSoup          # HTML parsing
from collections import Counter        # Counting HTML tag frequencies
import numpy as np                     # Numerical operations with arrays/matrices
from sklearn.cluster import DBSCAN     # Unsupervised clustering algorithm
from tqdm import tqdm                  # Progress bar for loops
import shutil                          # File copying between folders
import matplotlib.pyplot as plt
import seaborn as sns


# Extract frequency of HTML tags from a file
def extract_tag_frequency(html_path):
    # Open the HTML file with UTF-8 encoding and ignore errors
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Parse the HTML content using the lxml parser
        soup = BeautifulSoup(f, 'lxml')

    # Extract all tag names from the document
    tags = [tag.name for tag in soup.find_all()]

    # Return a dictionary (Counter) with the frequency of each tag
    return Counter(tags)

# Build a tag frequency matrix for a list of HTML files
def build_tag_matrix(counters, min_total_freq = 5):
    # Count total frequency of each tag across all files
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

    # Initialize the distance matrix
    D = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Compute Chi-squared distance between each pair of rows
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = 0.5 * np.sum(((X[i] - X[j]) ** 2) / (X[i] + X[j] + epsilon))
            D[i, j] = D[j, i] = d  # symmetric matrix
    return D

# Group similar HTML files and save them into folders
def group_similar_htmls(directory, eps=0.5, min_samples=2, output_base="output"):
    html_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.html')]
    counters = [extract_tag_frequency(f) for f in tqdm(html_files, desc=f"Parsing {directory}")]
    matrix, tags = build_tag_matrix(counters)

    # Show tag frequency matrix as heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap='YlGnBu', xticklabels=tags, yticklabels=False)
    plt.title("HTML Tag Frequency per File")
    plt.xlabel("HTML Tags")
    plt.ylabel("Files")
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{tier_name}_tag_frequency_heatmap.png"))
    plt.close()

    # print(matrix)
    # print("\n\n\n")

    chi2_dist = chi2_distance_matrix(matrix)

    print(chi2_dist)

    # Apply DBSCAN clustering on the distance matrix
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(chi2_dist)

    # Organize files into clusters based on DBSCAN labels
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(html_files[idx])  # group files by label

    # Create the output directory for this tier
    tier_name = os.path.basename(directory.strip("/"))
    output_dir = os.path.join(output_base, tier_name)
    os.makedirs(output_dir, exist_ok=True)

    grouped_count = 0
    # For each cluster (including outliers), create a separate folder and copy files
    for label, files in clusters.items():
        if label == -1:
            group_folder = os.path.join(output_dir, "outliers")  # -1 means outliers
        else:
            group_folder = os.path.join(output_dir, f"group_{label}")
        os.makedirs(group_folder, exist_ok=True)

        # Copy each HTML file into its corresponding group folder
        for file_path in files:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(group_folder, filename))

        if label != -1:
            grouped_count += 1

    # Print summary of how many groups were created
    print(f"Saved groups in {output_dir}: {grouped_count} groups + outliers")

# Run clustering for all tier folders
if __name__ == "__main__":
    # List of subdirectories to process
    for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
        print(f"Grouping for: {tier}")
        group_similar_htmls(tier, eps=1.0, min_samples=2)
