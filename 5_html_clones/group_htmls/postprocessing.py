import os
import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from group_htmls.image_processing import generate_screenshot_if_missing, image_embedding


def postprocessing(labels, distance_matrix, html_files, threshold_merge=25, threshold_attach=5, can_print=0):
    print("\tPostprocessing...")
    n = len(html_files)
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    clusters = dict(clusters)

    log_messages = []

    def log(msg):
        if can_print:
            print(msg)
        log_messages.append(msg)

    outlier_indices = clusters.get(-1, [])
    assigned_outliers = []
    stats = {
        "initial_outliers": len(outlier_indices),
        "attached_outliers": 0,
        "final_outliers": 0,
        "group_merges": 0,
        "total_groups_before": len([l for l in clusters if l != -1]),
        "total_groups_after": 0
    }

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
        if min_dist <= threshold_attach:
            clusters[closest_group].append(idx)
            assigned_outliers.append(idx)
            stats["attached_outliers"] += 1

    remaining_outliers = [idx for idx in outlier_indices if idx not in assigned_outliers]
    stats["final_outliers"] = len(remaining_outliers)
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
            if avg_dist <= threshold_merge:
                html_path1 = html_files[members_i[0]]
                html_path2 = html_files[members_j[0]]
                generate_screenshot_if_missing(html_path1)
                generate_screenshot_if_missing(html_path2)
                ss1 = os.path.join("screenshots", os.path.basename(html_path1) + ".png")
                ss2 = os.path.join("screenshots", os.path.basename(html_path2) + ".png")
                emb1 = image_embedding(ss1)
                emb2 = image_embedding(ss2)
                visual_sim = cosine_similarity([emb1], [emb2])[0][0]
                if visual_sim > 0.93:
                    clusters[label_i].extend(clusters[label_j])
                    merged.add(label_j)
                    label_mapping[label_j] = label_i
                    stats["group_merges"] += 1

    new_labels = np.full(n, -1)
    current_label = 0
    for old_label in sorted(clusters.keys()):
        if old_label in merged:
            continue
        if old_label == -1:
            for idx in clusters[old_label]:
                new_labels[idx] = -1
        else:
            for idx in clusters[old_label]:
                new_labels[idx] = current_label
            current_label += 1
    stats["total_groups_after"] = current_label

    return new_labels, stats, log_messages


def save_logs(log_messages, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "postprocessing.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_messages))


def save_stats(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, "postprocessing_stats.csv")
    with open(stats_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])
