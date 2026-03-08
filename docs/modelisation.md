# Modélisation

- **Binary Relevance (BR)**
  - 8 classifieurs binaires OneVsRest
  - Ne prend pas en compte les relations entre labels

- **Classifier Chains (CC)**
  - Prend en compte les dépendances entre labels
  - Améliore BR en utilisant la prédiction d’un label comme feature pour le suivant
  - Ordre des labels laissé par défaut (pas de randomisation)

- **Deep Learning (DL)**
  - Architecture :
    - Multi-Layer Perceptron (MLP)
    - 2 à 3 couches cachées
    - Dimensions typiques : 256, 128, 64
    - Activation ReLU pour chaque couche cachée
    - Dropout pour régularisation (ex : 0.2)
    - Couche de sortie avec activation sigmoïde
  - Fonctions de coût :
    - Binary Cross-Entropy (BCE) weighted
    - Focal Loss
  - Hyperparamètres :
    - Learning rate (lr) : 1e-3
    - Nombre d’epochs : 500 - 1000

