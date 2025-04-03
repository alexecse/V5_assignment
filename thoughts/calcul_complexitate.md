Vom presupune că:

n = numărul de fișiere HTML

t = numărul total de taguri diferite (în urma filtrării)

g = număr de grupuri (de obicei mult mai mic decât n)

🔍 1. extract_tag_frequency(...)
Se aplică pe fiecare fișier o dată → O(n) apeluri

Fiecare parcurgere este proporțională cu numărul de taguri din fișier → presupunem constantă ≈ O(1)

✅ Complexitate totală: O(n)

🔍 2. build_tag_matrix(...)
Combină toate Counter-ele → O(n × t)

Reconstituie matricea de frecvențe → tot O(n × t)

✅ Total: O(n × t)

🔍 3. chi2_distance_matrix(...)
Matrice pătratică simetrică → pentru fiecare pereche (i, j) calculează o distanță

Fiecare calcul de distanță este peste un vector de t dimensiuni → O(t)

Sunt n(n - 1)/2 ≈ O(n²) perechi

✅ Complexitate totală: O(n² × t)

👉 Acesta e cel mai scump pas.

🔍 4. DBSCAN.fit_predict(...) (pe distanțe precompute)
În versiunea metric='precomputed', DBSCAN rulează în:

O(n log n) (medie), dar în cel mai rău caz poate ajunge la O(n²)

✅ Practic: O(n²)

🔍 5. integrate_outliers(...)
Fie o = număr de outlieri

Fiecare outlier e comparat cu toate grupurile → în medie g grupuri cu n/g membri

Fiecare outlier face o medie peste ≈ n distanțe → O(n × o)

✅ Total: O(n × o)

🔍 6. shutil.copy(...) și os.makedirs(...)
Creează foldere și copiază fiecare fișier → O(n)

🔍 7. Plot: heatmap
Matrice de dimensiune n × t

Plotarea în sine este O(n × t), dar nu influențează performanța critică

✅ Concluzie – Complexitate totală
Componentă	Complexitate
HTML parsing	O(n)
Tag matrix	O(n × t)
Chi² distance matrix	O(n² × t)
DBSCAN clustering	O(n²)
Postprocesare outlieri	O(n × o)
Copiere fișiere	O(n)
🎯 Dominant:
O(n² × t) – calculul matricii de distanță Chi²
urmat de O(n²) – DBSCAN

📌 Optimizări posibile:
Reducerea dimensionalității tagurilor (ex: selectezi top k taguri)

Folosirea unui vectorizer mai eficient (TF-IDF pe HTML-ul curățat, dacă incluzi textual similarity)

Paralelizare la calculul matricii Chi²

Folosirea unui aproximator de distanțe (ex: hashing) dacă scalează la n > 1000