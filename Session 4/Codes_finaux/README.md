# Codes_finaux — fusion "Codes prof" + "Colab_Notebooks"

Ce dossier fusionne l'ensemble des notebooks de `Session 4/Codes prof` et `Session 4/Colab_Notebooks`. Rien n'a été supprimé : chaque notebook est conservé, et une cellule markdown en tête de chaque fichier précise sa provenance (voir "Traçabilité (fusion Codes_finaux)"). Les commentaires pédagogiques ajoutés côté Colab sont conservés et préfixés `# [Colab]` dans le code, même redondants avec les commentaires existants.

## Notebooks fusionnés (présents dans les deux dossiers, code merge)
| Fichier | Origine base | Contenu Colab intégré |
|---|---|---|
| `MNIST_1_DNN_Keras.ipynb` | Codes prof | commentaires pédagogiques + cellule `y_test[0:5]` |
| `MNIST_2_CNN_Keras.ipynb` | Codes prof | `print(X_train_valid)` de debug + interprétation détaillée du `model.summary()` |
| `spam_1_DNN_Keras_base.ipynb` | Codes prof | aucune différence de code (Colab = mêmes cellules, sorties non ré-exécutées) |
| `spam_4_DNN_Keras_tuner.ipynb` | Codes prof | aucune différence de code avec `Colab_Notebooks/spam_4_DNN_Keras_tuner(1).ipynb` |

## Notebooks uniques à "Codes prof" (copiés tels quels)
- `MNIST_3_CNN_PyTorch.ipynb`
- `MNIST_4_VGG16_Keras.ipynb`
- `MNIST_5_VGG16_augmente_Keras.ipynb`
- `MNIST_Lecture.ipynb`
- `ozone_complet_1_DNN_Keras_tuner.ipynb`
- `ozone_complet_2_DNN_Keras_Optuna.ipynb`
- `spam_2_DNN_Keras_dropout.ipynb`
- `spam_3_DNN_Keras_callback.ipynb`
- `spam_5_DNN_Keras_Optuna.ipynb`
- `spam_6_DNN_Pytorch.ipynb`

## Notebooks uniques à "Colab_Notebooks" (copiés tels quels)
- `spam_1_DNN_Keras_base_Dropout.ipynb` (variante dropout spécifique Colab, différente de `spam_2_DNN_Keras_dropout.ipynb`)
- `spam_1_DNN_Keras_base_JC_avec annotations.ipynb` (copie annotée par un(e) participant(e))
- `spam_4_DNN_Keras_tuner_Colab_enrichi.ipynb` (renommé depuis `Colab_Notebooks/spam_4_DNN_Keras_tuner.ipynb`, version enrichie et divergente, contient une cellule en erreur)
- `spam.csv` (données utilisées par les notebooks spam)
