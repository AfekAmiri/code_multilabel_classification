# Prétraitement des données

- Fusion des descriptions :
  - `prob_desc_input_spec` + `prob_desc_sample_inputs`
  - `prob_desc_output_spec` + `prob_desc_sample_outputs`

- Standardisation de la variable `difficulty` (StandardScaler)

- Imputation des valeurs manquantes de la variable `difficulty` (stratégie : mean)

- Embedding des variables textuelles avec codeBERT
  - CodeBERT est un modèle pré-entraîné spécifiquement pour le code et le langage naturel.
  - Excellentes performances sur des tâches de recherche de code et de génération de documentation.
  - Son architecture bimodale et ses objectifs de pré-entraînement (MLM + RTD) exploitent à la fois des données jumelées (NL-PL) et du code seul.
  - Meilleurs performances dans PTM4Tag: Sharpening Tag Recommendation of Stack Overflow Posts with Pre-trained Models https://arxiv.org/pdf/2203.10965 

> **Remarque 1:** MLM (Masked Language Modeling) : On masque aléatoirement des tokens dans une séquence (texte ou code) et le modèle doit prédire les tokens masqués à partir du contexte. Cela permet d’apprendre des représentations contextuelles riches.

> **Remarque 2:** RTD (Replaced Token Detection) : On remplace certains tokens par des alternatives générées, puis le modèle doit détecter si chaque token est original ou remplacé. C’est une tâche de classification binaire qui aide à mieux distinguer les vrais tokens des faux, et améliore la robustesse du modèle.

- Encodage multi-label des targets (MultiLabelBinarizer)
