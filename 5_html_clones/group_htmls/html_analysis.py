from bs4 import BeautifulSoup
from collections import Counter
import numpy as np


def extract_tag_frequency(html_path):
    USELESS_TAGS = {'svg', 'link', 'meta', 'noscript', 'defs', 'filter'}
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
    tags = [tag.name for tag in soup.find_all() if tag.name not in USELESS_TAGS]
    return Counter(tags)


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
        for j in range(i + 1, n_samples):
            d = 0.5 * np.sum(((X[i] - X[j]) ** 2) / (X[i] + X[j] + epsilon))
            D[i, j] = D[j, i] = d
    return D


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
