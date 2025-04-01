🟢 BEFORE: Chi² + frecvență absolută
🔧 Cum funcționa:
Pentru fiecare HTML, numărai de câte ori apare fiecare tag (ex: <div> = 50, <a> = 10)

Creai un vector cu frecvențele tagurilor

Comparai paginile folosind Chi² distance, o metrică foarte sensibilă la diferențe de distribuție

✅ Avantaje:
Capturează nu doar ce taguri sunt, ci și cât de mult se folosesc

Foarte bun pentru a diferenția template-uri cu comportamente diferite: pagini cu multe <script>, <iframe>, etc.

🔴 Dezavantaje:
Dacă două pagini au aceeași structură dar un număr diferit de apariții, Chi² poate să le considere foarte diferite

E mai sensibil la outliers dacă tagurile sunt rare dar apar masiv într-un singur HTML

🔴 AFTER: Cosine + binarizare (0/1)
🔧 Ce am făcut:
În loc de „câte ori apare tagul”, am folosit doar: tagul apare sau nu

Am comparat cu cosine distance: unghiul dintre doi vectori binari

✅ Avantaje:
Foarte tolerant la numărul de apariții

Poate prinde similaritate în structură, dacă layoutul e aproape identic (aceleași taguri, în linii mari)

🔴 Dezavantaje:
Dacă două pagini au aceleași taguri dar în proporții foarte diferite, cosine le vede ca aproape identice

Nu ține cont de intensitatea utilizării tagurilor (ex: una are 3 scripturi, alta 300 → sunt la fel pentru cosine dacă ambele au <script>)

🔍 Ce am observat la tine:
Paginile erau template-uri similare, dar nu identice la nivel de taguri

Frecvențele tagurilor erau informative: pagini similare aveau profiluri comune de utilizare a elementelor

Prin binarizare, ai pierdut acea semnătură cantitativă, deci gruparea a devenit prea relaxată și a ratat similarități subtile

✅ Ce am învățat:
Chi² + frecvență ✔	Cosine + binar ❌
Captează cât de mult apar tagurile	✅ Da	❌ Nu
Bine pentru pagini care par similare vizual	✅ Da	🔸 Uneori
Prea sensibil la zgomot	🔸 Da	✅ Mai puțin
Bun pentru clustering precis	✅ Da	❌ Mai slab
🧠 Recomandare:
Folosim o combinație:

Chi² + frecvență (ca în versiunea ta inițială) este cea mai bună variantă pentru template detection

Poți adăuga extra features (tag-uri cheie, lungime, adâncime DOM etc.) dacă vrei un model hibrid (statistic + ML)



cu Chi patrat si frecvente absolute :
	Saved groups in output/tier1: 7 groups + 9/101 outliers
	Saved groups in output/tier2: 2 groups + 6/22 outliers
	Saved groups in output/tier3: 5 groups + 4/40 outliers
	Saved groups in output/tier4: 3 groups + 4/30 outliers

cu binarizare, o prostie:
	Saved groups in output/tier1: 3 groups + 0/101 outliers
	Saved groups in output/tier2: 1 groups + 0/22 outliers
	Saved groups in output/tier3: 1 groups + 1/40 outliers
	Saved groups in output/tier4: 2 groups + 0/30 outliers


1:15 AM

Incerc sa integrez o postprocesare ca sa mai iau din outliers:):)

threshold_attach	
	Cât de aproape trebuie să fie un outlier de un grup	
	Doar outlieri (label = -1)	
	Atașează outlierul la cel mai apropiat grup

threshold_merge		
	Cât de apropiate trebuie să fie două grupuri existente	
	Grupuri diferite (label != -1)	
	Combină două grupuri într-unul singur

threshold_merge = 0.25, threshold_attach = 0.15 - .................
threshold_merge = 0.30, threshold_attach = 0.25 - .................
threshold_merge = 0.30, threshold_attach = 0.30 
	- tier4 a iesit splendid
	- tier1 e cam permisiv lmaoo, nasol
	- prob si celelalte, reduc threshold-urile

threshold_merge = 0.15, threshold_attach = 3.50 - merge binisor, in tier4 chiar se mai apropie datele

threshold_merge = 2, threshold_attach = 3.5
	Saved groups in output/tier1: 7 groups + 8/101 outliers
	Saved groups in output/tier2: 2 groups + 6/22 outliers
	Saved groups in output/tier3: 3 groups + 4/40 outliers
	Saved groups in output/tier4: 3 groups + 1/30 outliers


Cu pipeline-ul tău actual (Chi² + DBSCAN), nu poți merge peste ~10.000 fișiere fără modificări majore.

Chi² și DBSCAN nu sunt scalabile direct la 1B de pagini

Soluția industrială folosește:

ANN (approximate nearest neighbors)

distributed clustering

bucketing / hashing pentru reducerea spațiului de comparare