# Roadmap du Projet Olist Sales Forecast

Ce document recapitule l'avancement du projet et définit les prochaines étapes pour passer d'un notebook d'analyse à une application de prévision robuste.

## Ce qui a ete fait (Completed)

1. **Exploration des Données (EDA)**
    * Analyse des fichiers sources (`olist_*.csv`).
    * Nettoyage des données et fusion des tables.
    * Analyse exploratoire (Top produits, États, Distribution des paiements).
    * Export du dataset consolidé : `data/processed/olist_full_dataset.csv`.

2. **Modelisation (Model Training)**
    * Création du script `notebooks/model_training.marimo.py` et du notebook `.ipynb`.
    * **Preparation Time Series** : Agrégation des ventes à la semaine (`W`).
    * **Feature Engineering** : Création de variables retardées (Lags) et temporelles (Mois, Semaine).
    * **Implémentation de Prophet** : Modèle principal pour la gestion de la saisonnalité.
    * **Implémentation de Random Forest** : Modèle de comparaison (Baseline).
    * **Correction de bugs** : Alignement des dimensions entre les jeux de données d'entraînement.

## Ce qu'il reste a faire (Next Steps)

Voici les étapes logiques pour finaliser et professionnaliser le projet :

### Phase 1 : Amelioration du Modele

* [ ] **Cross-Validation (Backtesting)** : Tester le modèle sur plusieurs périodes passées (pas juste les 20 derniers %) pour valider sa robustesse.
* [ ] **Hyperparameter Tuning** : Optimiser les paramètres de Prophet (changepoint_prior_scale, seasonality_mode) pour réduire le MAE.
* [ ] **Ajout de régresseurs** : Inclure les jours fériés brésiliens ou des événements promotionnels (Black Friday).

### Phase 2 : Industrialisation (ML Engineering)

* [ ] **Refactoring** : Déplacer le code propre (nettoyage, training) des notebooks vers des scripts Python dans `src/` (ex: `src/data_loader.py`, `src/model.py`).
* [ ] **Pipeline d'entraînement** : Créer un script unique `train.py` qui génère et sauvegarde le modèle (fichier `.json` ou `.pkl`).
* [ ] **API de Prédiction** : Créer une petite API (FastAPI) pour servir les prédictions.

### Phase 3 : Visualisation (Dashboarding)

* [ ] **Création d'un Dashboard** : Utiliser **Streamlit** ou **Marimo** pour créer une interface web interactive.
  * Sélecteur de catégories de produits.
  * Visualisation des prévisions vs réel.
  * KPIs clés (Chiffre d'affaires prévisionnel mois prochain).
