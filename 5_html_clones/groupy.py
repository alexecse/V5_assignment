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
	USELESS_TAGS = {'script', 'style', 'svg', 'link', 'meta', 'noscript', 'defs', 'filter'}

	with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
		# Parse the HTML content using the lxml parser
		soup = BeautifulSoup(f, 'lxml')

	tags = [tag.name for tag in soup.find_all() if tag.name not in USELESS_TAGS]
	return Counter(tags)

def extract_text_content(html_path):
	with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
		soup = BeautifulSoup(f, 'lxml')
	return soup.get_text(separator=' ', strip=True)

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

def compute_textual_similarity(html_files):
	texts = [extract_text_content(path) for path in html_files]
	
	vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
	
	tfidf_matrix = vectorizer.fit_transform(texts)
	
	cosine_sim = cosine_similarity(tfidf_matrix)

	textual_dist = 1 - cosine_sim
	textual_dist[textual_dist < 0] = 0

	return textual_dist

# threshold_attach >6 devine ft permisiv
# threshold_merge > 20 avantajos pentru site-uri mai complicate. s-ar da merge mult la site-uri simple (tier1) si nu ar fi avantajos
# merge cercetat daca e chiar intr-ajutorul site-urilor complexe
def postprocessing(labels, distance_matrix, html_files, threshold_merge=2, threshold_attach=6, can_print=1):
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

	if outlier_indices:
		log("Outliers found: " + str([os.path.basename(html_files[i]) for i in outlier_indices]))
	else:
		log("No outliers found.")

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

		log(f"[OUTLIER DEBUG] {filename} → Closest group: {closest_group}")
		log(f"  - Avg distance to group: {min_dist:.4f}")
		log(f"  - Threshold (attach):    {threshold_attach:.4f}")

		diff = min_dist - threshold_attach
		if diff <= 0.05:
			log(f"  → ⚠ Outlier is very close! (only {diff:.4f} above threshold)")
		elif diff <= 0.15:
			log(f"  → ℹ Consider lowering threshold slightly if you want it attached.")

		if min_dist <= threshold_attach:
			log(f"  ✓ Attached to group {closest_group}\n")
			clusters[closest_group].append(idx)
			assigned_outliers.append(idx)
			stats["attached_outliers"] += 1
		else:
			log(f"  ✗ Kept as outlier\n")

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
		for j in range(i+1, len(group_labels)):
			label_j = group_labels[j]
			if label_j in merged:
				continue
			members_i = clusters[label_i]
			members_j = clusters[label_j]
			all_dists = [distance_matrix[a][b] for a in members_i for b in members_j]
			avg_dist = np.mean(all_dists)

			log(f"[MERGE DEBUG] Group {label_i} ↔ Group {label_j} → avg_dist = {avg_dist:.4f}")

			if avg_dist <= threshold_merge:
				log(f"  ✓ Merged group {label_j} into group {label_i}\n")
				clusters[label_i].extend(clusters[label_j])
				merged.add(label_j)
				label_mapping[label_j] = label_i
				stats["group_merges"] += 1
			else:
				log(f"  ✗ Not merged (avg_dist > threshold)\n")

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

	stats["total_groups_after"] = current_label

	return new_labels, stats, log_messages

def save_logs(log_messages, output_dir, label):
	group_folder = os.path.join(output_dir, f"group_{label}" if label != -1 else "outliers")
	os.makedirs(group_folder, exist_ok=True)
	log_file = os.path.join(group_folder, "postprocessing.log")
	with open(log_file, "w", encoding="utf-8") as f:
		f.write("\n".join(log_messages))

def save_stats(stats, output_dir, label):
	import csv
	group_folder = os.path.join(output_dir, f"group_{label}" if label != -1 else "outliers")
	os.makedirs(group_folder, exist_ok=True)
	stats_file = os.path.join(group_folder, "postprocessing_stats.csv")
	with open(stats_file, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["metric", "value"])
		for key, value in stats.items():
			writer.writerow([key, value])

def group_similar_htmls(directory, eps, min_samples=2, do_postprocessing=1):
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
	tag_matrix, tags = build_tag_matrix(counters)

	# Compute Chi-squared distance matrix
	chi2_dist = chi2_distance_matrix(tag_matrix)
	# Compute textual distance
	textual_dist = compute_textual_similarity(html_files)

	# Combine structural and textual distances
	combined_dist = combine_distances_dynamic(chi2_dist, textual_dist)
	# Perform DBSCAN clustering
	clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
	labels = clustering.fit_predict(combined_dist)

	# Postprocessing - try and integrate the outliers
	if do_postprocessing:
		labels, stats, logs = postprocessing(labels, combined_dist, html_files)

		# Save logs for each group
		for label in set(labels):
			save_logs(logs, output_dir, label)
			save_stats(stats, output_dir, label)

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

	print(f"Grupuri salvate în {output_dir}: {len([k for k in clusters if k != -1])} grupuri, {len(clusters.get(-1, []))} outlieri")

	# Prepare heatmap labels (filename + group)
	filenames = [os.path.basename(f) for f in html_files]
	cluster_labels = ['outlier' if lbl == -1 else f'group_{lbl}' for lbl in labels]
	y_labels = [f"{name} ({group})" for name, group in zip(filenames, cluster_labels)]

	# Heatmap with tag frequency and cluster labels
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
	# for tier in ['./test_clones']:
	for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
		print(f"Grouping for: {tier}")
		group_similar_htmls(tier, eps = 3, min_samples=2, do_postprocessing=1)
