# Modélisation

- **Binary Relevance (BR)**
  - 8 classifieurs binaires OneVsRest
  - Ne prend pas en compte les relations entre labels

- **Classifier Chains (CC)**
  - Prend en compte les dépendances entre labels
  - Améliore BR en utilisant la prédiction d’un label comme feature pour le suivant
  - Ordre des labels laissé par défaut (pas de randomisation)

- **Base Estimator**
  - Logistic Regression : C=1.0, penalty='l2'
  - Equivalent à un perceptron à une seule couche avec une activation sigmoïd
  - Résiste au sur-apprentissage (régularisation)

- **Deep Learning (DL)**
  - Architecture :
    - Multi-Layer Perceptron (MLP)
    - 2 à 3 couches cachées
    - Dimensions typiques : 256, 128, 64
    - Activation ReLU pour chaque couche cachée
    - Dropout pour régularisation (ex : 0.2)
    - Couche de sortie avec activation sigmoïde
  - Fonctions de coût :
    - Binary Cross-Entropy (BCE) (weighted et non)
    - Focal Loss a.(1-pt)^g.BCE_loss (https://arxiv.org/pdf/2401.16549) :
      - pt : proba prédite pour la classe vraie.
      - (1-pt)^g : pondère l'erreur, en donnant plus de poids aux exemples difficiles (faible pt).
      - a : facteur d'équilibrage global.
  - Hyperparamètres :
    - Learning rate (lr) : 1e-3
    - Nombre d’epochs : 500 - 1000

