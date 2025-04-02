# HTML Grouping Pipeline

Acest proiect grupeaza fisiere HTML pe baza **similitutinii vizuale, structurale si textuale**, cu accent pe modularitate, claritate si scalabilitate pe termen lung.

> Creat initial pentru un task tehnic real. Scopul nu a fost doar sa functioneze, ci sa experimentez, sa rafinez si sa demonstrez ca pot construi o solutie de productie.

---

## Obiectiv

Grupare automata a paginilor HTML:
- In functie de layout si asemanare vizuala perceputa
- Prin compararea structurii DOM, continutului textual si optional a screenshot-urilor
- Cu suport pentru identificarea outlierilor si atasarea incrementala a paginilor noi

---

## Evolutia solutiei

### 🟢 Versiunea initiala
- Vectori de frecventa ai tagurilor HTML
- Distanta Chi² pentru comparatie structurala
- DBSCAN pentru clustering nesupravegheat

### 🔴 Probleme observate
- Chi² era prea sensibil la diferente mici de frecventa
- Outlierii nu erau tratati bine
- Scalabilitate limitata la cateva mii de fisiere

### 🟡 Imbunatatiri adaugate
- **Postprocesare outlieri**: atasare si merge pentru grupuri similare vizual
- **Parsing paralel**: cu `ThreadPoolExecutor`
- **Caching local**: salvare rezultate intermediare pentru performanta
- **Similaritate hibrida**: combinare Chi² + textual (TF-IDF/semantic) + vizual (embedding imagine)

### 🔵 Experimente cu metrici

| Strategie               | Rezultate tier1-4       | Observatii                              |
|------------------------|--------------------------|------------------------------------------|
| Chi² + frecventa       | 7 + 2 + 5 + 3 grupuri     | Cea mai robusta si precisa varianta      |
| Cosine + binarizare    | 3 + 1 + 1 + 2 grupuri     | Prea permisiva, pierde informatia        |
| Cu postprocesare       | 7 + 2 + 3 + 3 + outlieri  | Calitate vizibil imbunatatita            |

Am incercat, comparat si rafinat pe baza rezultatelor reale obtinute pe cele 4 dataseturi (tier1 - tier4).

---

## Arhitectura pipeline

1. **Parsing HTML** → frecventa taguri + text curat
2. **Vectorizare**:
    - structura: matrice frecvente Chi²
    - text: TF-IDF sau embedding semantic
3. **Clustering**: DBSCAN pe distanta combinata
4. **Postprocesare**: reatasare outlieri, merge grupuri apropiate vizual
5. **Output**: foldere organizate, heatmap, statistici, loguri

---

## Complexitate

| Etapa                  | Complexitate            |
|------------------------|--------------------------|
| Parsing HTML           | O(n)                     |
| Matrice frecvente      | O(n × t)                 |
| Distanţa Chi²         | O(n² × t) ✅ Bottleneck   |
| DBSCAN                 | O(n²)                    |
| Integrare outlieri     | O(n × o)                 |

Punct critic: **matricea Chi-squared** → se poate optimiza prin filtrare taguri sau matrix sparse.

---

## Scalabilitate

Sistemul actual functioneaza eficient pentru cateva mii de fisiere. Pentru volume mari:

### Matching incremental
- FAISS pentru salvarea vectorilor
- La fiecare pagina noua:
  - Generezi vector
  - Cauti top-k similaritati in index
  - Atasezi la grupul cel mai apropiat (daca trece pragul)

### Alternative la DBSCAN
- **HDBSCAN** – mai robust si mai scalabil
- **MiniBatchKMeans** – daca estimezi numarul de grupuri

### Approximate Nearest Neighbors (ANN)
- FAISS / Annoy pentru cautare rapida
- Cosine similarity devine O(log n) in loc de O(n²)

### Procesare pe batch-uri
- Spargi inputul in sharduri (ex: foldere)
- Rulezi clustering pe fiecare independent
- Faci merge global intre batch-uri (post-hoc)

---

## Formula combinata (ajustabila)
```python
similarity = (
    0.4 * structural_similarity_chi2 +
    0.3 * textual_similarity +
    0.3 * visual_similarity
)
```

---

## TODO si extensii posibile
- [ ] FAISS index persistent pentru matching in timp real
- [ ] Inlocuire DBSCAN cu HDBSCAN
- [ ] Evaluare automata a calitatii clusterelor
- [ ] Dashboard interactiv pentru explorarea grupurilor

---

## Concluzie

Aceasta solutie demonstreaza:
- Gindire modulara si separare clara pe responsabilitati
- Atentie la complexitate algoritmica si performanta
- O abordare completa, nu doar un script
- Capacitatea de a experimenta si rafina pana la un rezultat solid


---