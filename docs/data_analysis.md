# Analyse de forme
* **Variable target :** 
    * T=38 tags individuels, certains très fréquents (greedy, implementation, math)
    * Cardinalité des tags (le nombre moyen de tags par exemple) : 2.8
    * Densité des tags (le nombre moyen de tags par exemple divisé par le nombre total de tags T) : 0.074
    * Diversité des tags : 1907 combinaisons distinctes de tags, certains très fréquents ([implementation], [math], [greedy]). 
    * Comme suggéré par l'énoncé, je vais me concentrer sur les tags **'math', 'graphs','strings','number theory','trees','geometry','games','probabilities'**
        * Cardinalité des tags : 1.273
        * Densité des tags : 0.159
        * Diversité des tags : 77
* **Lignes et colonnes :** (4982, 21)
* **Types de variables :** 17 qualitatives, 3 quantitatives
* **Analyse de valeurs manquantes :** hidden_unit_tests is an empty string, prob_desc_notes : 27%, quelques valeurs manquantes pour prob_desc_output_spec, prob_desc_input_spec et difficulty

**Il n'est pas pertinent d'utiliser la stratégie Label Powerset parce uniquement 30% des 2^8 classes possibles sont représentées, dont beaucoup de classes rares**
# Analyse de fond

**On peux enlever la colonne prob_desc_memory_limit**
**On peux supprimer la colonne prob_desc_time_limit**

Relation Variables / Target :
* Une dépendance entre tags et difficulty.
* Les features time_limit et memory_limit ne sont pas significatives.
* Relation entre variables textuelles et targets : on voit 2 clusters mais avec des tags mélangés.
* codeBERT est mieux adapté que TF-IDF : les données sont mieux distribuées dans l'espace.