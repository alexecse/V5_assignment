import os
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, Counter as CountLabels
from collections import defaultdict

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

def integrate_outliers(labels, distance_matrix, html_files, threshold_merge = 2, threshold_attach = 3.5):
	n = len(html_files)
	clusters = defaultdict(list)
	for idx, label in enumerate(labels):
		clusters[label].append(idx)
	clusters = dict(clusters)

	# 1. Procesare outlieri
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
			print(f"  → ⚠ Outlier is very close! (only {diff:.4f} above threshold)")
		elif diff <= 0.15:
			print(f"  → ℹ Consider lowering threshold slightly if you want it attached.")

		if min_dist <= threshold_attach:
			print(f"  ✓ Attached to group {closest_group}\n")
			clusters[closest_group].append(idx)
			assigned_outliers.append(idx)
		else:
			print(f"  ✗ Kept as outlier\n")

	# Păstrăm outlierii care n-au fost atașați
	remaining_outliers = [idx for idx in outlier_indices if idx not in assigned_outliers]
	if remaining_outliers:
		clusters[-1] = remaining_outliers
	elif -1 in clusters:
		del clusters[-1]

	# 2. Fuzionare grupuri similare
	group_labels = [l for l in clusters if l != -1]
	merged = set()
	label_mapping = {}

	for i in range(len(group_labels)):
		label_i = group_labels[i]
		if label_i in merged:
			continue
		for j in range(i+1, len(group_labels)):
			label_j = group_labels[j]
			if label_j in merged:
				continue
			members_i = clusters[label_i]
			members_j = clusters[label_j]
			all_dists = [distance_matrix[a][b] for a in members_i for b in members_j]
			avg_dist = np.mean(all_dists)

			print(f"[MERGE DEBUG] Group {label_i} ↔ Group {label_j} → avg_dist = {avg_dist:.4f}")
			if avg_dist <= threshold_merge:
				print(f"  ✓ Merged group {label_j} into group {label_i}\n")
				clusters[label_i].extend(clusters[label_j])
				merged.add(label_j)
				label_mapping[label_j] = label_i
			else:
				print(f"  ✗ Not merged (avg_dist > threshold)\n")

	# Regenerăm etichetele coerent
	new_labels = np.full(n, -1)
	current_label = 0
	label_map = {}

	for old_label in sorted(clusters.keys()):
		if old_label in merged:
			continue

		if old_label == -1:
			# păstrăm -1 ca etichetă pentru outlieri nealocați
			for idx in clusters[old_label]:
				new_labels[idx] = -1
		else:
			for idx in clusters[old_label]:
				new_labels[idx] = current_label
			label_map[old_label] = current_label
			current_label += 1

	return new_labels

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

	# Compute Chi-squared distance matrix
	chi2_dist = chi2_distance_matrix(matrix)

	# Perform DBSCAN clustering
	clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
	labels = clustering.fit_predict(chi2_dist)

	# postprocesare - try and integrate the outliers
	labels = integrate_outliers(labels, chi2_dist, html_files)

	final_clusters = defaultdict(list)
	for idx, label in enumerate(labels):
		final_clusters[label].append(html_files[idx])

	for label, files in final_clusters.items():
		if label == -1:
			group_folder = os.path.join(output_dir, "outliers")
		else:
			group_folder = os.path.join(output_dir, f"group_{label}")
		os.makedirs(group_folder, exist_ok=True)

		for file_path in files:
			filename = os.path.basename(file_path)
			shutil.copy(file_path, os.path.join(group_folder, filename))
	
	grouped_count = sum(1 for label in final_clusters if label != -1)
	outlier_count = len(final_clusters.get(-1, []))
	total_files = len(html_files)
	print(f"Saved groups in {output_dir}: {grouped_count} groups + {outlier_count}/{total_files} outliers")

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
	# for tier in ['./test_clones']:
	for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
		print(f"Grouping for: {tier}")
		group_similar_htmls(tier, eps=1.0, min_samples=2)
