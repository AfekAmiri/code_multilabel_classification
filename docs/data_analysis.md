## Analyse de forme

- **Variable target :**
    - $T=38$ tags individuels, certains très fréquents (`greedy`, `implementation`, `math`)
    ![Distribution des tags](_assets/images/tag_distribution.png)
    - Cardinalité des tags (nombre moyen de tags par exemple): 2.8
    - Densité des tags (cardinalité/T): 0.074
    - Diversité des tags : 1907 combinaisons distinctes, certaines très fréquentes (`[implementation]`, `[math]`, `[greedy]`)
    - Focus sur les tags principaux : **math, graphs, strings, number theory, trees, geometry, games, probabilities**
        - Cardinalité : 1.273
        - Densité : 0.159
        - Diversité : 77

    > **Remarque :** La stratégie Label Powerset n'est pas pertinente ici car seulement 30% des 2^8 classes sont représentées, dont beaucoup de classes rares.

    ![Distribution des tags filtrés](_assets/images/filtered_tags_distribution.png)
- **Lignes et colonnes :** (4982, 21) et après filtrage des tags (2678, 21) divisés en 80% train et 20% test.
- **Types de variables :** 17 qualitatives, 3 quantitatives
    ![Types de data](_assets/images/datatypes.png)
- **Analyse des valeurs manquantes :**
    - `hidden_unit_tests` est toujours vide
    - `prob_desc_notes` : 27% de valeurs manquantes
    - Quelques valeurs manquantes pour `prob_desc_output_spec`, `prob_desc_input_spec`, `difficulty`
    ![Matrice des NaN](_assets/images/nan_matrix.png)
    ![Pourcentage de NaN par colonne](_assets/images/nan_percentage_per_column.png)

## Analyse de fond (relation variables/target)

### Analyse des variables numériques

![Distribution de memory_limit](_assets/images/memory_limit_distribution.png)

**On peut enlever la colonne `prob_desc_memory_limit`**

![Distribution de time_limit](_assets/images/time_limit_distribution.png)

![Boxplot de time_limit par tag](_assets/images/boxplot_time_limit_per_tag.png)

**On peut supprimer la colonne `prob_desc_time_limit`**

![Distribution de difficulty](_assets/images/difficulty_distribution.png)

![Boxplot de difficulty par tag](_assets/images/boxplot_difficulty_per_tag.png)

**Dépendance entre `tags` et `difficulty`**

### Analyse des variables textuelles

#### Comparaison TF-IDF et codeBERT
![umap tf-idf description](_assets/images/umap_tf_idf_desc.png)
![umap codebert description](_assets/images/umap_codebert_desc.png)

**codeBERT** est mieux adapté que **TF-IDF** : les données sont mieux distribuées dans l'espace.

#### Analyse de source code
![umap code](_assets/images/umap_code.png)
![code lengths ](_assets/images/code_lengths.png)
![code tokens ](_assets/images/code_tokens.png)
![code math ](_assets/images/code_math.png)

#### Analyse de description
![umap desc](_assets/images/umap_codebert_desc.png)
![description lengths ](_assets/images/desc_lengths.png)
![desc tokens ](_assets/images/desc_tokens.png)
![desc math ](_assets/images/desc_math.png)

#### Analyse de input_spec
![umap input](_assets/images/umap_input.png)
![input lengths ](_assets/images/input_lengths.png)
![input tokens ](_assets/images/input_tokens.png)
![input math ](_assets/images/input_math.png)

#### Analyse de output_spec
![umap output](_assets/images/umap_output.png)
![output lengths ](_assets/images/output_lengths.png)
![output tokens ](_assets/images/output_tokens.png)
![output math ](_assets/images/output_math.png)

- Relation entre variables textuelles et targets : 2 ou plusieurs clusters mais tags mélangés
- les scatter plots des variables textuelles ont des distribution très différentes -> A priori pas trop de correlation entre eux. Il fallait creuser les relations variable/variable et regarder les statistiques de la cosine similarity entre les embeddings.
