from bs4 import BeautifulSoup
from collections import Counter
import numpy as np


def extract_tag_frequency(html_path):
    # Parses the HTML file and returns a frequency count of HTML tags, excluding what is considered not-relevant
    # Tags that don't contribute meaningful structural information
    USELESS_TAGS = {'svg', 'link', 'meta', 'noscript', 'defs', 'filter'}

    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')

    # Extract tag names
    tags = [tag.name for tag in soup.find_all() if tag.name not in USELESS_TAGS]

    return Counter(tags)


def build_tag_matrix(counters, min_total_freq=5):
    # Builds a tag frequency matrix for all HTML files, filtered by minimum total frequency.
    #   counters (list of Counter): One counter per HTML file.
    #   min_total_freq (int): Only include tags that appear at least this many times in total.
    
    total_freq = Counter()
    
    # Combine all counters to get total frequency of each tag
    for counter in counters:
        total_freq.update(counter)

    # Filter tags that are too rare across all files
    filtered_tags = sorted(tag for tag, freq in total_freq.items() if freq >= min_total_freq)

    # Build matrix: each row is a file, each column is a tag count
    tag_matrix = []
    for counter in counters:
        row = [counter.get(tag, 0) for tag in filtered_tags]
        tag_matrix.append(row)

    # returns the tag_matrix with size num_files x num_tags;
    # and the list of tag names.
    return np.array(tag_matrix), filtered_tags


def chi2_distance_matrix(tag_matrix, epsilon=1e-10):
    # Computes the pairwise Chi-squared distance matrix between all rows in tag_matrix
    #   tag_matrix = matrix where each row is a feature vector - tag histogram
    #   epsilon = a small constant to avoid division by zero

    tag_matrix = tag_matrix.astype(np.float32)
    n_samples = tag_matrix.shape[0]
    D = np.zeros((n_samples, n_samples), dtype=np.float32)

    # Compute only upper triangle and mirror (D is symmetrical)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Chi-squared distance formula (for histograms)
            d = 0.5 * np.sum(((tag_matrix[i] - tag_matrix[j]) ** 2) / (tag_matrix[i] + tag_matrix[j] + epsilon))
            D[i, j] = D[j, i] = d

    return D

def chi2_distance_matrix_fast(tag_matrix, epsilon=1e-10):
    # Vectorized Chi-squared distance matrix. 
    # Input tag_matrix must be a NumPy array of shape (n_samples, n_features)
    tag_matrix = tag_matrix.astype(np.float64)
    n = tag_matrix.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        xi = tag_matrix[i]
        denom = xi + tag_matrix + epsilon  # shape: (n, d)
        num = (xi - tag_matrix) ** 2       # shape: (n, d)
        chi2 = 0.5 * np.sum(num / denom, axis=1)
        dist_matrix[i, :] = chi2

    return dist_matrix


def combine_distances_dynamic(chi2_dist, textual_dist, alpha_range=(0.2, 0.8)):
    # Combines Chi-squared and textual distances dynamically based on Chi-squared value
    #   chi2_dist = the Chi-squared distance matrix
    #   textual_dist = Text-based distance matrix (e.g. cosine)
    #   alpha_range = min and max weight to assign to chi2_dist

    # This function greatly helps the grouping process
    
    n = chi2_dist.shape[0]
    combined_dist = np.zeros_like(chi2_dist)

    for i in range(n):
        for j in range(n):
            chi2_val_i_j = chi2_dist[i, j]
            text_val_i_j = textual_dist[i, j]

            # Dynamically choose weight alpha based on chi2 value
            if chi2_val_i_j < 1.0:
                alpha = alpha_range[0]
            elif chi2_val_i_j > 3.0:
                alpha = alpha_range[1]
            else:
                # Linear interpolation between alpha_min and alpha_max
                alpha = np.interp(chi2_val_i_j, [1.0, 3.0], alpha_range)

            # Weighted combination of the two distance sources
            combined_dist[i, j] = alpha * chi2_val_i_j + (1 - alpha) * text_val_i_j

    return combined_dist
