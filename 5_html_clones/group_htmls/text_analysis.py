from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def extract_text_content(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'lxml')
    return soup.get_text(separator=' ', strip=True)


def compute_textual_similarity(html_files):
    texts = [extract_text_content(path) for path in html_files]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix)
    textual_dist = 1 - cosine_sim
    textual_dist[textual_dist < 0] = 0
    return textual_dist


def compute_semantic_similarity(html_files):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [extract_text_content(path) for path in html_files]
    embeddings = model.encode(texts, show_progress_bar=True)
    cos_sim = cosine_similarity(embeddings)
    textual_dist = 1 - cos_sim
    textual_dist[textual_dist < 0] = 0
    return textual_dist
