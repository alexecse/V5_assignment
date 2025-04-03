from group_htmls.image_processing import image_embedding, generate_screenshot_if_missing
from group_htmls.text_analysis import extract_text_content, compute_textual_similarity
from group_htmls.html_analysis import extract_tag_frequency, build_tag_matrix, chi2_distance_matrix, combine_distances_dynamic
from group_htmls.postprocessing import postprocessing, save_logs, save_stats

import os
import shutil
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor

def parallel_extract_tag_frequencies(html_files):
    with ProcessPoolExecutor() as executor:
        return list(tqdm(executor.map(extract_tag_frequency, html_files), total=len(html_files), desc="Extracting tag frequencies"))

def parallel_extract_texts(html_files):
    with ProcessPoolExecutor() as executor:
        return list(tqdm(executor.map(extract_text_content, html_files), total=len(html_files), desc="Extracting text content"))


def group_similar_htmls(directory, eps, min_samples, do_postprocessing=1):
    tier_name = os.path.basename(directory.strip("/"))
    output_dir = os.path.join("output", tier_name)
    stats_dir = os.path.join("statistics", tier_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    html_files = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.endswith('.html') and not f.startswith('._')]
    
    # Parallelized tag extraction
    counters = parallel_extract_tag_frequencies(html_files)
    tag_matrix, tags = build_tag_matrix(counters)

    # Parallelized text extraction
    texts = parallel_extract_texts(html_files)

    chi2_dist = chi2_distance_matrix(tag_matrix)
    textual_dist = compute_textual_similarity(html_files, texts)
    combined_dist = combine_distances_dynamic(chi2_dist, textual_dist)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(combined_dist)

    if do_postprocessing:
        labels, stats, logs = postprocessing(labels, combined_dist, html_files)
        save_logs(logs, output_dir)
        save_stats(stats, output_dir)

    final_clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        final_clusters[label].append(html_files[idx])

    for label, files in final_clusters.items():
        group_folder = os.path.join(output_dir, "outliers" if label == -1 else f"group_{label}")
        os.makedirs(group_folder, exist_ok=True)
        for file_path in files:
            shutil.copy(file_path, os.path.join(group_folder, os.path.basename(file_path)))

    print(f"\tPages grouped in {output_dir}: {len([k for k in final_clusters if k != -1])} groups, {len(final_clusters.get(-1, []))} stand-alone pages (outliers)")

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
        print(f"Grouping for: {tier}")
        group_similar_htmls(tier, eps=2, min_samples=2, do_postprocessing=1)
