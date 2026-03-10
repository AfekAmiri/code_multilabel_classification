## Expériences

- J'ai testé plusieurs combinaisons de features avec BR et CC.
- Meilleur combinaison : description + code + input spec
- J'ai testé quelques modèles de dl avec la meilleur combinaison en utilisant différents loss :
    - BCE
    - Weighted BCE
    - Focal Loss
## Résultats des modèles

| model   | features                                |   accuracy |   f1_weighted |   precision_weighted |   recall_weighted |
|:--------|:----------------------------------------|-----------:|--------------:|---------------------:|------------------:|
| cc      | desc,code,input                         |   0.546642 |      0.702542 |             0.743577 |          0.693694 |
| cc      | desc,code,input,output                  |   0.539179 |      0.701668 |             0.735058 |          0.689189 |
| br      | desc,code,input                         |   0.514925 |      0.698453 |             0.748463 |          0.68018  |
| br      | desc,code,input,output                  |   0.505597 |      0.697543 |             0.739349 |          0.675676 |
| cc      | desc,code                               |   0.546642 |      0.685408 |             0.742473 |          0.668168 |
| br      | desc,input,output                       |   0.501866 |      0.671558 |             0.709838 |          0.656156 |
| br      | desc,code                               |   0.475746 |      0.669823 |             0.738629 |          0.63964  |
| cc      | desc,input,output                       |   0.514925 |      0.664228 |             0.696596 |          0.654655 |
| cc      | desc                                    |   0.492537 |      0.628012 |             0.676962 |          0.621622 |
| cc      | code                                    |   0.453358 |      0.531176 |             0.597992 |          0.537538 |
| br      | code                                    |   0.395522 |      0.51413  |             0.643046 |          0.46997  |
| cc      | desc,code,input,output,notes,difficulty |   0.393657 |      0.506305 |             0.65941  |          0.537538 |
| cc      | desc,code,input,output,notes            |   0.373134 |      0.500034 |             0.652552 |          0.534535 |
| br      | desc,code,input,output,notes,difficulty |   0.358209 |      0.498659 |             0.644338 |          0.521021 |
| br      | desc,code,input,output,notes            |   0.341418 |      0.493573 |             0.611825 |          0.521021 |
| dl      | desc,code,input                         |   0.401119 |      0.471686 |             0.654963 |          0.509009 |

## Comparaison entre weighted BCE et focal loss

| model   | features                                |   accuracy |   f1_weighted |   precision_weighted |   recall_weighted |
|:--------|:----------------------------------------|-----------:|--------------:|---------------------:|------------------:|
| dl_desc_code_input_focal | desc,code,input |   0.546642 |      0.712896 |             0.740194 |          0.71021  |
| dl_desc_code_input_bce   | desc,code,input |   0.412313 |      0.705207 |             0.652143 |          0.782282 |

## Détails des métriques du meilleur modèle

| classe         | précision | rappel | f1-score | support |
|:-------------- |:---------:|:------:|:--------:|:-------:|
| games          |   0.95    |  0.64  |   0.77   |   28    |
| geometry       |   0.74    |  0.61  |   0.67   |   33    |
| graphs         |   0.74    |  0.62  |   0.68   |  121    |
| math           |   0.77    |  0.86  |   0.81   |  273    |
| number theory  |   0.50    |  0.38  |   0.43   |   63    |
| probabilities  |   1.00    |  0.11  |   0.19   |   19    |
| strings        |   0.76    |  0.83  |   0.79   |   60    |
| trees          |   0.72    |  0.55  |   0.62   |   69    |



| moyenne      | précision | rappel | f1-score | support |
|:------------ |:---------:|:------:|:--------:|:-------:|
| micro avg    |   0.74    |  0.69  |   0.72   |  666    |
| macro avg    |   0.77    |  0.57  |   0.62   |  666    |
| weighted avg |   0.74    |  0.69  |   0.70   |  666    |
| samples avg  |   0.75    |  0.73  |   0.72   |  666    |


