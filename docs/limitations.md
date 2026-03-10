# Limitations

- Texte tronqué à 512 tokens :
  - Peut entraîner une perte d’information
  - Il aurait fallu utiliser des techniques de découpage ou de gestion de séquences longues (ex : sliding window, modèles adaptés)

- Base_estimator :
  - Utilisation exclusive de LogisticRegression pour BR et CC
  - Pas d’optimisation des hyperparamètres
  - Il aurait été pertinent de tester d’autres modèles (RandomForest, SVM, etc.) et d’optimiser avec GridSearchCV

- Modèle Deep Learning (DL) :
  - Peu d’expérimentations
  - Les modèles classiques ont donné de meilleures performances, donc moins d’exploration DL

- Fine-tuning CodeBERT :
  - Tentative de fine-tuning sur Colab
  - Problème : épuisement de la RAM, impossible de poursuivre l’expérimentation
