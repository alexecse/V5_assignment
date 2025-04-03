# 🔍 HTML Grouping Pipeline

> Grupare automată a fișierelor HTML pe baza similarității structurale și textuale.  
> Proiect orientat pe analiză statistică aplicată, modularitate si scalabilitate 

---

## 📦 Ce face acest proiect?

- Parsează fișiere HTML și extrage trăsături structurale (taguri)
- Le grupează automat folosind clustering nesupravegheat (DBSCAN)
- Evaluează și vizualizează coerența grupurilor
- Suportă extensii cu metrici textuali & vizuali (ex: TF-IDF, embeddings)

---

## ⚙️ Instalare

```bash
git clone https://github.com/username/html-grouping-pipeline.git
cd html-grouping-pipeline
pip install -r requirements.txt
```

---

## ▶️ Cum se rulează

```bash
python src/main.py --input ./data/html_pages --output ./results
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

---

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