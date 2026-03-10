## Objectif 
 Prédire les tags en fonction des données problème de codage disponibles avec un rappel de 70% et un f1-score supérieur à 70%.

## Métriques

 **Rappel** : TP / (TP + FN)

 **Précision** : TP / (TP + FP)

 **F1-Score** : 2 × (Précision × Rappel) / (Précision + Rappel)


## Méthodes d'agrégation

- **Micro :** calcule les métriques globalement en comptant le total des vrais positifs, faux négatifs et faux positifs.
- **Macro :** calcule les métriques pour chaque label, puis fait leur moyenne non pondérée. Cette méthode ne prend pas en compte le déséquilibre entre les classes.
- **Weighted :** calcule les métriques pour chaque label, puis fait leur moyenne pondérée par le support (nombre d’instances réelles pour chaque label). Cette méthode ajuste la macro pour tenir compte du déséquilibre des classes ; elle peut donner un F-score qui n’est pas compris entre précision et rappel.
- **Samples :** calcule les métriques pour chaque instance, puis fait leur moyenne.

J'ai choisi **Weighted** pour contrebalancer le déséquilibre des tags.


