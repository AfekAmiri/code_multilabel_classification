# Résultats des modèles

| model                    | features        |   accuracy |   f1_weighted |   precision_weighted |   recall_weighted |
|:-------------------------|:----------------|-----------:|--------------:|---------------------:|------------------:|
| dl_desc_code_input_focal | desc,code,input |   0.546642 |      0.712896 |             0.740194 |          0.71021  |
| dl_desc_code_input_bce   | desc,code,input |   0.412313 |      0.705207 |             0.652143 |          0.782282 |

## Détails des métriques


### dl_desc_code_input_focal

               precision    recall  f1-score   support

        games       0.95      0.68      0.79        28
     geometry       0.62      0.64      0.63        33
       graphs       0.73      0.70      0.72       121
         math       0.77      0.88      0.82       273
number theory       0.53      0.30      0.38        63
probabilities       0.60      0.16      0.25        19
      strings       0.80      0.87      0.83        60
        trees       0.77      0.49      0.60        69

    micro avg       0.75      0.71      0.73       666
    macro avg       0.72      0.59      0.63       666
 weighted avg       0.74      0.71      0.71       666
  samples avg       0.76      0.75      0.73       666



### dl_desc_code_input_bce

               precision    recall  f1-score   support

        games       0.85      0.82      0.84        28
     geometry       0.54      0.76      0.63        33
       graphs       0.58      0.82      0.68       121
         math       0.80      0.84      0.82       273
number theory       0.35      0.71      0.47        63
probabilities       0.24      0.32      0.27        19
      strings       0.67      0.87      0.75        60
        trees       0.54      0.62      0.58        69

    micro avg       0.62      0.78      0.69       666
    macro avg       0.57      0.72      0.63       666
 weighted avg       0.65      0.78      0.71       666
  samples avg       0.67      0.80      0.70       666


