# 🔍 HTML Grouping Pipeline

> Grupare automată a fișierelor HTML pe baza similarității structurale și textuale.  
> Proiect orientat pe analiză statistică aplicată, modularitate si scalabilitate 

---

## Ce face acest proiect?

- Parsează fișiere HTML și extrage trăsături structurale (taguri)
- Le grupează automat folosind clustering nesupravegheat (DBSCAN), pe baza unei distante statistice bazate pe Chi-patrat
- Evaluează și vizualizează coerența grupurilor
- Suportă extensii cu metrici textuali & vizuali (ex: TF-IDF, embeddings)

---

## Instalare

```bash
git clone https://github.com/alexecse/V5_assignment.git
cd V5_assignment

unzip clones.zip

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to run this project?

```bash
python run.py
python server_html.py
streamlit run dashboard.py - for integrated output, stats, and logs in a dashboard
python clean.py - for cleaning
```

---

## 📚 Documentație tehnică


- [1. Teorie și Metodologie](#1-teorie--metodologie)
- [2. Implementare](#2-implementare)
- [3. Concluzii & Extindere](#3-concluzii--extindere)

---

## 1. Teorie & Metodologie

### Obiectiv

- Grupare automată a paginilor HTML similare
- Evaluare riguroasă a metodelor de comparare structurală
- Explorare metrici: structurale, textuale și vizuale
- Analiză comparativă a metodelor de clustering

### Abordare

- HTML = structură ierarhică → analizabilă statistic
- Metrici testate:
  - Chi² + frecvență (structural)
  - Cosine + binarizare (structural simplificat)
  - TF-IDF sau SBERT (text)
  - Vizual embedding (ex: CLIP, DINO)

<details><summary>Long live Numerical Methods: Fast Chi-Squared Distance Numerical Optimization</summary>

# Fast Chi-Squared Distance Computation: Numerical Optimization

## Overview
In the context of clustering HTML documents based on tag distributions, we use the **Chi-squared distance** as a dissimilarity metric. The original implementation involved a nested loop over all pairs of documents, which resulted in high computational complexity.

To address this, we introduce a **vectorized implementation** of Chi-squared distance using NumPy. This approach provides substantial performance gains, especially for large datasets.

---

## Chi-Squared Distance Formula
Given two vectors \( \mathbf{x}, \mathbf{y} \in \mathbb{R}^d \), the Chi-squared distance is defined as:

\[ \chi^2(\mathbf{x}, \mathbf{y}) = \frac{1}{2} \sum_{i=1}^{d} \frac{(x_i - y_i)^2}{x_i + y_i + \varepsilon} \]

Where:
- \( x_i, y_i \) are the components of the vectors
- \( \varepsilon \) is a small constant added to avoid division by zero

---

## Vectorized Implementation
Instead of using nested loops over all \( n \) pairs, we perform the computation row-wise in a vectorized fashion using NumPy:

```python
for i in range(n):
    xi = X[i]
    denom = xi + X + epsilon  # shape: (n, d)
    num = (xi - X) ** 2       # shape: (n, d)
    chi2 = 0.5 * np.sum(num / denom, axis=1)
    dist_matrix[i, :] = chi2
```

This approach avoids redundant computation and leverages NumPy's optimized internals.

---

## Complexity Analysis

### Original Implementation
- **Time complexity**: \( \mathcal{O}(n^2 \cdot d) \)
- **Space complexity**: \( \mathcal{O}(n^2) \) for distance matrix
- Bottleneck: Python-level loops over all pairs of documents

### Vectorized Version
- **Time complexity**: Still \( \mathcal{O}(n^2 \cdot d) \), but drastically faster in practice
- Gains come from memory locality, CPU vectorization, and NumPy broadcasting

---

## Advantages of the Fast Method
- **10x to 50x faster** than naive Python implementation
- Easy to integrate with existing pipelines
- Retains full numerical accuracy of Chi-squared distance
- Can be extended with Numba or run in <b>batches</b> for <b>scalability</b>

---

## Concluzie
This optimized Chi-squared computation method makes it feasible to cluster large sets of HTML documents using document structure features, without incurring massive runtime penalties. It is a crucial enhancement for any scalable unsupervised clustering pipeline.

</details>

## 2. Implementare

### Arhitectura pipeline

1. Parsing HTML → extragere frecvențe + text
2. Vectorizare:
   - structură: Chi² pe frecvențe
   - text: TF-IDF / SBERT
   - imagine: CLIP / vizual
3. Clustering: DBSCAN (sau HDBSCAN)
4. Postprocesare: reatașare outlieri, merge grupuri
5. Output: heatmaps, foldere organizate, statistici

### Complexitate

| Etapă                   | Complexitate            |
|------------------------|--------------------------|
| Parsing HTML           | O(n)                     |
| Matrice frecvențe      | O(n × t)                 |
| Distanță Chi²          | O(n² × t) 🔺 Bottleneck  |
| DBSCAN                 | O(n²)                    |
| Postprocesare          | O(n × o)                 |

---

## 3. Concluzii & Extindere

### Concluzii

- Chi² + frecvență = cea mai robustă metrică structurală
- Cosine + binarizare = mai rapidă, dar mai slabă calitativ
- Hibridizare (structură + text + vizual) îmbunătățește semnificativ clusteringul

### Extensii viitoare

- [ ] FAISS persistent index pentru incremental search
- [ ] Înlocuire DBSCAN cu HDBSCAN
- [ ] Auto-evaluare calitate clustere (metrice interne)
- [ ] Dashboard interactiv pentru vizualizare și debugging

---

> 💡 _Nu doar un script, ci un framework complet pentru analiză și grupare HTML, construit cu grijă la detalii și performanță._