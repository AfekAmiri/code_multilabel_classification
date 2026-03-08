# Prétraitement des données

- Fusion des descriptions :
  - `prob_desc_input_spec` + `prob_desc_sample_inputs`
  - `prob_desc_output_spec` + `prob_desc_sample_outputs`

- Standardisation de la variable `difficulty` (StandardScaler)

- Imputation des valeurs manquantes de la variable `difficulty` (stratégie : mean)

- Embedding des variables textuelles avec codeBERT

- Encodage multi-label des targets (MultiLabelBinarizer)
