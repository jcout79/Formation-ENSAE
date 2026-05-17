# Démarche d'analyse — Guide d'utilisation

---

## Cas 1 — Y quantitative (Régression)

**Objectif** : prédire une valeur numérique (ex. concentration d'ozone, prix, etc.)

### Notebooks à lancer et dans quel ordre

| Étape | Notebook | Ce qu'il fait | Fichiers produits |
|---|---|---|---|
| 1 | `01_preparation.ipynb` | Charge les données depuis `DONNEES/`, encode les variables qualitatives, construit les 3 jeux de features | `dfbase.csv`, `dfpoly.csv`, `dfinter.csv` |
| 2 | `02_ComparaisonReg_final.ipynb` | Compare MCO, Lasso, Ridge, ElasticNet, PCR, PLS, Arbre, Forêt en validation croisée (10 blocs) | `PREV.csv` / `PREVpoly.csv` / `PREVinter.csv` |
| 3 | `03_analyseReg_complet.ipynb` | Calcule le MSE par méthode, trace les graphiques de comparaison, sélection par stabilité bootstrap | — |

### Procédure détaillée

1. Lancer **`01_preparation.ipynb`**
   - Modifier le chemin de chargement des données (`DONNEES/ozone.txt`, `eucalyptus.txt`, etc.)
   - Toutes les cellules produisent `dfbase.csv`, `dfpoly.csv`, `dfinter.csv` dans `FICHIERS FINAUX/`

2. Dans **`02_ComparaisonReg_final.ipynb`**
   - Charger le jeu souhaité : modifier `don = pd.read_csv("dfbase.csv" | "dfpoly.csv" | "dfinter.csv")`
   - Lancer toute la boucle de validation croisée
   - Sauvegarder avec la cellule correspondante uniquement :
     - `dfbase.csv` → lancer la cellule `PREV.to_csv("PREV.csv")`
     - `dfpoly.csv` → lancer la cellule `PREV.to_csv("PREVpoly.csv")`
     - `dfinter.csv` → lancer la cellule `PREV.to_csv("PREVinter.csv")`
   - Répéter pour chaque jeu de features souhaité

3. Dans **`03_analyseReg_complet.ipynb`**
   - Ajuster `JEUX_ACTIFS` en cellule de configuration (`["base"]`, `["base","poly"]`, etc.)
   - Lancer tout le notebook

---

## Cas 2 — Y binaire (Classification Supervisée)

**Objectif** : prédire une classe 0/1 (ex. maladie présente/absente, spam/non-spam, etc.)

### Notebooks à lancer et dans quel ordre

| Étape | Notebook | Ce qu'il fait | Fichiers produits |
|---|---|---|---|
| 1 | `01_preparation_CS.ipynb` | Charge les données depuis `DONNEES/`, encode les variables qualitatives, construit les 3 jeux de features | `dfbase.csv`, `dfpoly.csv`, `dfinter.csv` |
| 2 | `02_ComparaisonCS_final.ipynb` | Compare Logistique, BIC/AIC forward, Lasso/Ridge/ElasticNet logistiques, Arbre, Forêt en validation croisée **stratifiée** (10 blocs) | `PROB.csv` / `PROBpoly.csv` / `PROBinter.csv` |
| 3 | `03_analyseCS_complet.ipynb` | Calcule accuracy, sensibilité, spécificité, F1, AUC ; trace les courbes ROC ; sélection par stabilité bootstrap | — |

### Procédure détaillée

1. Lancer **`01_preparation_CS.ipynb`**
   - Modifier le chemin de chargement des données (`DONNEES/SAheart.data`, `spambase.data`, etc.)
   - Toutes les cellules produisent `dfbase.csv`, `dfpoly.csv`, `dfinter.csv` dans `FICHIERS FINAUX/`

2. Dans **`02_ComparaisonCS_final.ipynb`**
   - Charger le jeu souhaité : modifier `don = pd.read_csv("dfbase.csv" | "dfpoly.csv" | "dfinter.csv")`
   - Lancer toute la boucle de validation croisée
   - Sauvegarder avec la cellule correspondante uniquement :
     - `dfbase.csv` → lancer la cellule `PROB.to_csv("PROB.csv")`
     - `dfpoly.csv` → lancer la cellule `PROB.to_csv("PROBpoly.csv")`
     - `dfinter.csv` → lancer la cellule `PROB.to_csv("PROBinter.csv")`
   - Répéter pour chaque jeu de features souhaité

3. Dans **`03_analyseCS_complet.ipynb`**
   - Ajuster `JEUX_ACTIFS` en cellule de configuration (`["base"]`, `["base","poly"]`, etc.)
   - Lancer tout le notebook

---

## Description des 4 modules `.py`

Ces modules implémentent la **sélection de variables par critères d'information (AIC/BIC)**.
Ils existent en deux versions selon le type de problème (régression vs classification) et selon
l'approche (stepwise vs exhaustive).

---

### `logistic_step_sk.py` — Sélection stepwise pour la classification (sklearn)

**Utilisé dans** : `02_ComparaisonCS_final.ipynb`

Fournit la classe `LogisticRegressionSelectionFeatureIC` qui hérite de `LogisticRegression` (sklearn).

**Algorithme** :
- Part d'un modèle vide (juste l'intercept) ou d'un modèle de départ défini par `start`
- À chaque étape, teste l'ajout (forward) ou le retrait (backward) de chaque variable
- Calcule l'AIC ou le BIC du modèle candidat :

  **AIC** = −2 log L + 2k  
  **BIC** = −2 log L + k log n

- Retient la variable qui minimise le critère
- S'arrête quand aucune modification n'améliore le critère

**Avantage** : compatible avec les pipelines et la validation croisée sklearn → utilisable directement dans la boucle CV de `02_ComparaisonCS_final.ipynb`.

**Complexité** : O(p²) — scalable même avec beaucoup de variables.

---

### `choixglmstats.py` — Sélection exhaustive pour la classification (statsmodels)

**Utilisé dans** : codes de travail uniquement (`CODES DE TRAVAIL/`), **pas dans les notebooks finaux**

Fournit la fonction `bestglm(data, upper, mustbe, family)`.

**Algorithme** :
- Énumère **toutes les combinaisons** possibles de variables (2^p sous-modèles)
- Ajuste un GLM binomial (logistique) via statsmodels pour chaque combinaison
- Retourne un DataFrame avec AIC, BIC et déviance pour chaque modèle
- L'utilisateur choisit ensuite le meilleur modèle

**Limitation** : coût en O(2^p) → inutilisable au-delà de ~15 variables. Non compatible avec les pipelines sklearn.

---

### `ols_step_sk.py` — Sélection stepwise pour la régression (sklearn)

**Utilisé dans** : `02_ComparaisonReg_final.ipynb`

Fournit la classe `LinearRegressionSelectionFeatureIC` qui hérite de `LinearRegression` (sklearn).

**Algorithme** : identique à `logistic_step_sk`, mais pour la régression linéaire (MCO). Le critère AIC/BIC est calculé à partir de la log-vraisemblance gaussienne :

  **AIC** = n log(SSR/n) + 2k  
  **BIC** = n log(SSR/n) + k log n

**Avantage** : compatible sklearn → utilisable dans la boucle CV de `02_ComparaisonReg_final.ipynb`.  
**Note** : les méthodes BIC et AIC sont actuellement commentées dans la boucle CV car coûteuses avec beaucoup de features.

---

### `choixolsstats.py` — Sélection exhaustive pour la régression (statsmodels)

**Utilisé dans** : codes de travail uniquement, **pas dans les notebooks finaux**

Fournit deux fonctions :
- `bestols(data, upper, mustbe)` : recherche exhaustive sur tous les sous-modèles, retourne un DataFrame avec AIC, BIC, SSR, R², R²adj
- `olsstep(data, start, lower, upper, direction, crit)` : sélection stepwise via formules statsmodels (alternative à `ols_step_sk` pour l'exploration interactive)

**Limitation** : même que `choixglmstats` — O(2^p), non compatible sklearn.

---

## Tableau récapitulatif des 4 modules

| Module | Problème | Approche | Complexité | Compatible sklearn | Utilisé dans les finaux |
|---|---|---|---|---|---|
| `logistic_step_sk` | Classification | Stepwise AIC/BIC | O(p²) | ✅ | ✅ `02_ComparaisonCS_final` |
| `choixglmstats` | Classification | Exhaustive 2^p | O(2^p) | ❌ | ❌ (codes de travail) |
| `ols_step_sk` | Régression | Stepwise AIC/BIC | O(p²) | ✅ | ✅ `02_ComparaisonReg_final` |
| `choixolsstats` | Régression | Exhaustive 2^p | O(2^p) | ❌ | ❌ (codes de travail) |
