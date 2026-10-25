# Démarche d'analyse — Guide d'utilisation

---

## Cas 1 — Y quantitative (Régression)

**Objectif** : prédire une valeur numérique (ex. concentration d'ozone, prix, etc.)

### Notebooks à lancer et dans quel ordre

| Étape | Notebook | Ce qu'il fait | Fichiers produits |
|---|---|---|---|
| 1 | `02_Reg_pipeline_complet.ipynb` | Charge les données (URL ozone par défaut), construit la matrice de variables (`X`, `XI`, `XP` ou `XIP`), compare **tous les modèles** en validation croisée (10 blocs) : MCO, BIC/AIC, Lasso, ElasticNet, Ridge, PCR, PLS, Forêt, Gradient boosting (`gbm1`, `gbm2`, `gbms1`, `gbms2`), CatBoost ; produit le tableau des MSE, les graphiques et les hyperparamètres par bloc | `PREV_complet.csv`, `metriques_reg.csv`, `hyperparametres_reg.csv`, `resultats_reg_mse_residus.png` |
| 2 | `03_analyseReg_complet.ipynb` | Lit `PREV_complet.csv`, calcule MSE / RMSE / MAE, trace les graphiques de comparaison, sélection de variables par stabilité bootstrap (Lasso + ElasticNet) | — |

Il n'y a **pas de notebook de préparation** pour la régression : le `02_Reg_pipeline_complet` charge et prépare lui-même ses données.

### Procédure détaillée

1. Dans **`02_Reg_pipeline_complet.ipynb`**
   - Ajuster la cellule de configuration : `URL_DONNEES` (fichier séparé par `;`, première colonne = index), `NOM_Y` (variable à prédire), `ESPACE` (`"X"`, `"XI"`, `"XP"` ou `"XIP"`), `NB_BLOCS`, `CALCULER_IC` (mettre `False` avec `XI` / `XIP`, où BIC/AIC sont très longs)
   - Lancer toutes les cellules (environ 4 minutes avec `ESPACE = "X"`)
   - Toutes les méthodes aléatoires ont une graine (`GRAINE`, `GRAINE_CV`, `GRAINE_BLOCS`) : les résultats sont reproductibles

2. Dans **`03_analyseReg_complet.ipynb`**
   - Lancer tout le notebook (il lit `PREV_complet.csv`)
   - Compléter la conclusion avec les résultats

---

## Cas 2 — Y binaire (Classification Supervisée)

**Objectif** : prédire une classe 0/1 (ex. maladie présente/absente, spam/non-spam, etc.)

### Notebooks à lancer et dans quel ordre

| Étape | Notebook | Ce qu'il fait | Fichiers produits |
|---|---|---|---|
| 1 | `01_preparationCS.ipynb` | Charge les données depuis `DONNEES/`, encode les variables qualitatives, construit les jeux de features | `dfbase.csv`, `dfpoly.csv`, `dfinter.csv`, `dffull.csv` |
| 2 | `02_CS_pipeline_complet.ipynb` | Compare **tous les modèles** en validation croisée **stratifiée** (10 blocs) : Logistique, BIC/AIC forward, Ridge/Lasso/ElasticNet logistiques, Forêt, Gradient boosting (`gbm1`, `gbm2`), CatBoost ; produit le tableau des indicateurs (AUC, accuracy, sensibilité, spécificité, F1), les courbes ROC et les hyperparamètres par bloc | `PROB.csv` / `PROBpoly.csv` / `PROBinter.csv` / `PROBfull.csv`, `metriques_CS.csv`, `hyperparametres_CS.csv`, `resultats_CS_<jeu>.png` |
| 3 | `03_analyseCS_complet.ipynb` | Lit les `PROB*.csv`, calcule accuracy, sensibilité, spécificité, F1 (seuils 0.5 et naturel), AUC ; trace les courbes ROC ; sélection par stabilité bootstrap | — |

### Procédure détaillée

1. Lancer **`01_preparationCS.ipynb`**
   - Modifier le chemin de chargement des données (`DONNEES/SAheart.data`, `spambase.data`, etc.)
   - Toutes les cellules produisent `dfbase.csv`, `dfpoly.csv`, `dfinter.csv`, `dffull.csv` dans `FICHIERS FINAUX/`

2. Dans **`02_CS_pipeline_complet.ipynb`**
   - Choisir dans la cellule de configuration les jeux à comparer : `JEUX = ["base"]`, `["base", "poly"]`, `["base", "poly", "inter", "full"]`… (un fichier `PROB*.csv` est écrit par jeu)
   - `CALCULER_IC = False` accélère le calcul avec les gros jeux (`inter`, `full`), où BIC/AIC forward sont plus longs
   - Lancer toutes les cellules ; les graines (`GRAINE`, `GRAINE_BLOCS`) rendent les résultats reproductibles

3. Dans **`03_analyseCS_complet.ipynb`**
   - Ajuster `JEUX_ACTIFS` en cellule de configuration (`["base"]`, `["base","poly"]`, etc.) : ils doivent avoir été calculés à l'étape 2
   - Lancer tout le notebook

---

## Test final optionnel avec la validation croisée

Les pipelines `02_Reg_pipeline_complet` et `02_CS_pipeline_complet` ont un paramètre `TAILLE_TEST_FINAL` (**0 par défaut = désactivé**). Avec par exemple `0.2`, 20 % des données (stratifiées en classification) sont mises de côté **avant** la validation croisée. La validation croisée compare les méthodes sur le reste ; chaque modèle est ensuite réajusté sur tout ce reste et évalué **une seule fois** sur le test final (tableau « validation croisée vs test » avec intervalles de confiance).

Pourquoi : la validation croisée donne déjà des prédictions hors échantillon, mais le meilleur modèle est choisi sur ces mêmes prédictions (légèrement optimiste). Un test final jamais utilisé donne l'estimation non biaisée à annoncer.

| Cas | Fichiers produits en plus | Lecture par l'analyse |
|---|---|---|
| Régression | `PREV_cvtest.csv`, `metriques_reg_test_final.csv`, `resultats_reg_test_final.png` | `03_analyseReg_complet` avec `FICHIER_PREV = "PREV_cvtest.csv"` |
| Classification | `PROB_cvtest.csv` (+ `PROBpoly_cvtest.csv`…), `metriques_CS_test_final.csv`, `resultats_CS_test_final_<jeu>.png` | `03_analyseCS_complet` avec `SUFFIXE_PROB = "_cvtest"` |

Les fichiers `PREV_complet.csv` / `PROB*.csv` ne contiennent alors que les observations de la validation croisée (hors test final).

---

## Variante sans validation croisée : apprentissage / validation / test

Pour chaque cas, un notebook « split » reprend les mêmes modèles que le pipeline en validation croisée, mais découpe les données **une seule fois** en trois échantillons (stratifié pour la classification) :

| Échantillon | Rôle |
|---|---|
| Apprentissage | ajuste les modèles |
| Validation | choisit les hyperparamètres (λ ou C, nombre de composantes, nombre d'itérations du boosting) et sert d'échantillon d'early stopping |
| Test | évalue une seule fois le modèle final (estimation honnête) |

| Cas | Notebook | Fichiers produits | Analyse (`03_...`) |
|---|---|---|---|
| Régression | `02_Reg_pipeline_split.ipynb` | `PREV_test.csv`, `metriques_reg_split.csv`, `hyperparametres_reg_split.csv`, 2 graphiques | `03_analyseReg_complet.ipynb` avec `FICHIER_PREV = "PREV_test.csv"` |
| Classification | `02_CS_pipeline_split.ipynb` | `PROB_test.csv` (+ `PROBpoly_test.csv`…), `metriques_CS_split.csv`, `hyperparametres_CS_split.csv`, graphiques | `03_analyseCS_complet.ipynb` avec `SUFFIXE_PROB = "_test"` |

**Les notebooks d'analyse ne sont pas dupliqués** : ils lisent des prédictions hors échantillon, que celles-ci viennent de la validation croisée ou de l'échantillon de test. Il suffit de changer une variable de configuration (voir ci-dessus).

**Points d'attention** :
- Les résultats dépendent du tirage (`GRAINE_SPLIT`) et l'échantillon de test est petit : le notebook affiche des intervalles de confiance (bootstrap) ; des méthodes aux intervalles qui se recouvrent ne sont pas départageables. La validation croisée est plus stable.
- Les indicateurs de validation sont optimistes (ils ont servi à régler les modèles) : seul le test fait foi.
- Dans les `03_analyse...`, la section « sélection par stabilité » (bootstrap) utilise toujours **toutes** les données, quelle que soit l'origine des prédictions.

---

## Mode test : vérifier rapidement que le code tourne

Les 4 pipelines (`02_Reg_pipeline_complet`, `02_Reg_pipeline_split`, `02_CS_pipeline_complet`, `02_CS_pipeline_split`) ont, dans la cellule de configuration, une option :

```python
MODE_TEST = False           # True pour activer
FRACTION_TEST = 0.10        # part des lignes conservées
GRAINE_ECHANTILLON = 2025   # graine du tirage aléatoire des lignes
```

Avec `MODE_TEST = True`, seules 10 % des lignes du fichier de données (tirées au hasard) sont utilisées : le notebook s'exécute en quelques secondes ou minutes, ce qui permet de tester le code sur un nouveau jeu de données. **Les résultats n'ont alors aucune valeur** (trop peu d'observations) et les fichiers sont écrits dans le sous-dossier `test_pipeline/`, sans écraser les vrais résultats. Remettre `MODE_TEST = False` pour l'analyse réelle.

---

## Description des 4 modules `.py`

Ces modules implémentent la **sélection de variables par critères d'information (AIC/BIC)**.
Ils existent en deux versions selon le type de problème (régression vs classification) et selon
l'approche (stepwise vs exhaustive).

---

### `logistic_step_sk.py` — Sélection stepwise pour la classification (sklearn)

**Utilisé dans** : `02_CS_pipeline_complet.ipynb`

Fournit la classe `LogisticRegressionSelectionFeatureIC` qui hérite de `LogisticRegression` (sklearn).

**Algorithme** :
- Part d'un modèle vide (juste l'intercept) ou d'un modèle de départ défini par `start`
- À chaque étape, teste l'ajout (forward) ou le retrait (backward) de chaque variable
- Calcule l'AIC ou le BIC du modèle candidat :

  **AIC** = −2 log L + 2k  
  **BIC** = −2 log L + k log n

- Retient la variable qui minimise le critère
- S'arrête quand aucune modification n'améliore le critère

**Avantage** : compatible avec les pipelines et la validation croisée sklearn → utilisable directement dans la boucle CV de `02_CS_pipeline_complet.ipynb`.

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

**Utilisé dans** : `02_Reg_pipeline_complet.ipynb`

Fournit la classe `LinearRegressionSelectionFeatureIC` qui hérite de `LinearRegression` (sklearn).

**Algorithme** : identique à `logistic_step_sk`, mais pour la régression linéaire (MCO). Le critère AIC/BIC est calculé à partir de la log-vraisemblance gaussienne :

  **AIC** = n log(SSR/n) + 2k  
  **BIC** = n log(SSR/n) + k log n

**Avantage** : compatible sklearn → utilisable dans la boucle CV de `02_Reg_pipeline_complet.ipynb`.  
**Note** : BIC et AIC sont actifs dans la boucle CV (`CALCULER_IC = True`) ; avec `ESPACE = "XI"` ou `"XIP"` (beaucoup de variables), ils deviennent coûteux : mettre `CALCULER_IC = False`.

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
| `logistic_step_sk` | Classification | Stepwise AIC/BIC | O(p²) | ✅ | ✅ `02_CS_pipeline_complet` |
| `choixglmstats` | Classification | Exhaustive 2^p | O(2^p) | ❌ | ❌ (codes de travail) |
| `ols_step_sk` | Régression | Stepwise AIC/BIC | O(p²) | ✅ | ✅ `02_Reg_pipeline_complet` |
| `choixolsstats` | Régression | Exhaustive 2^p | O(2^p) | ❌ | ❌ (codes de travail) |
