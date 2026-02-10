# %% Section 0: Config Jupyter et librairies
# Permet d'afficher les graphiques inline dans Jupyter / VSCode Interactive
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Ne fait rien si on n'est pas dans Jupyter

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config graphiques
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# %% Section 1: Chemins du projet
# Compatible script normal et Jupyter
try:
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
except NameError:
    # Si on est dans Jupyter Notebook
    BASE_DIR = "/Users/amaury/olist-sales-forecast"

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Vérification des fichiers
print("RAW_DIR :", RAW_DIR)
print("Fichiers disponibles :", os.listdir(RAW_DIR))

# %% Section 2: Chargement des données
customers = pd.read_csv(os.path.join(RAW_DIR, "olist_customers_dataset.csv"))
orders = pd.read_csv(os.path.join(RAW_DIR, "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(RAW_DIR, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(RAW_DIR, "olist_products_dataset.csv"))
sellers = pd.read_csv(os.path.join(RAW_DIR, "olist_sellers_dataset.csv"))
payments = pd.read_csv(os.path.join(RAW_DIR, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(RAW_DIR, "olist_order_reviews_dataset.csv"))
categories = pd.read_csv(os.path.join(RAW_DIR, "product_category_name_translation.csv"))

# %% Section 3: Merge datasets
df = orders.merge(customers, on="customer_id", how="left") \
           .merge(order_items, on="order_id", how="left") \
           .merge(products, on="product_id", how="left") \
           .merge(sellers, on="seller_id", how="left") \
           .merge(payments, on="order_id", how="left") \
           .merge(reviews, on="order_id", how="left") \
           .merge(categories, on="product_category_name", how="left")

# %% Section 4: Aperçu des données
print("Dataset final :")
print(df.head())
print("\nShape :", df.shape)

# %% Section 5: Analyses rapides avec visualisations
# Nombre de commandes par état
state_counts = df.groupby("customer_state")["order_id"].count().sort_values(ascending=False)
print("\nNombre de commandes par état :")
print(state_counts)

# Graphique commandes par état
sns.barplot(x=state_counts.index, y=state_counts.values, palette="viridis")
plt.title("Nombre de commandes par état")
plt.ylabel("Nombre de commandes")
plt.xlabel("État")
plt.show()

# Top 10 produits les plus vendus
top_products = df.groupby("product_id")["order_id"].count().sort_values(ascending=False).head(10)
print("\nTop 10 produits les plus vendus :")
print(top_products)

# Graphique top produits
sns.barplot(x=top_products.values, y=top_products.index, palette="magma")
plt.title("Top 10 produits les plus vendus")
plt.xlabel("Nombre de commandes")
plt.ylabel("Product ID")
plt.show()

# Retard moyen de livraison (jours)
if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
    df['delivery_delay'] = (pd.to_datetime(df['order_delivered_customer_date']) - 
                            pd.to_datetime(df['order_estimated_delivery_date'])).dt.days
    print("\nRetard moyen de livraison (jours) :", df['delivery_delay'].mean())

    # Histogramme des retards
    sns.histplot(df['delivery_delay'], bins=50, kde=True)
    plt.title("Distribution des retards de livraison (jours)")
    plt.xlabel("Jours de retard")
    plt.ylabel("Nombre de commandes")
    plt.show()


# %% Section 7: Évolution des commandes dans le temps
print("\n--- Évolution des commandes dans le temps ---")

# Convertir la colonne en datetime dans df
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')

# Extraire le mois
df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')

# Nombre de commandes par mois
monthly_orders = df.groupby('order_month')['order_id'].count()
print(monthly_orders.head(10))

# Graphique
sns.lineplot(x=monthly_orders.index.astype(str), y=monthly_orders.values, marker='o')
plt.title("Évolution des commandes par mois")
plt.xlabel("Mois")
plt.ylabel("Nombre de commandes")
plt.xticks(rotation=45)
plt.show()


# %% Section 8: Analyse des retards de livraison par état
print("\n--- Analyse des retards de livraison par état ---")
# On utilise la colonne delivery_delay déjà créée
state_delay = df.groupby("customer_state")["delivery_delay"].mean().sort_values(ascending=False)
print(state_delay)

# Boxplot des retards par état
sns.boxplot(x='customer_state', y='delivery_delay', data=df)
plt.title("Distribution des retards de livraison par état")
plt.xlabel("État")
plt.ylabel("Jours de retard")
plt.show()

# %% Section 9: Top clients et top vendeurs
print("\n--- Top 10 clients les plus actifs ---")
top_customers = df.groupby("customer_id")["order_id"].count().sort_values(ascending=False).head(10)
print(top_customers)

sns.barplot(x=top_customers.values, y=top_customers.index, palette="coolwarm")
plt.title("Top 10 clients par nombre de commandes")
plt.xlabel("Nombre de commandes")
plt.ylabel("Client ID")
plt.show()

print("\n--- Top 10 vendeurs les plus actifs ---")
top_sellers = df.groupby("seller_id")["order_id"].count().sort_values(ascending=False).head(10)
print(top_sellers)

sns.barplot(x=top_sellers.values, y=top_sellers.index, palette="cool")
plt.title("Top 10 vendeurs par nombre de commandes")
plt.xlabel("Nombre de commandes")
plt.ylabel("Seller ID")
plt.show()

# %% Section 10: Analyse des ventes par catégorie de produit
# Top catégories de produits
category_counts = df.groupby("product_category_name_english")["order_id"].count().sort_values(ascending=False)
print("\nTop 10 catégories de produits les plus vendues :")
print(category_counts.head(10))

# Graphique barres top 10 catégories
sns.barplot(x=category_counts.head(10).values, y=category_counts.head(10).index, palette="Spectral")
plt.title("Top 10 catégories de produits par nombre de commandes")
plt.xlabel("Nombre de commandes")
plt.ylabel("Catégorie")
plt.show()


# %% Section 11: Analyse des modes de paiement
# Répartition des types de paiement
payment_counts = df.groupby("payment_type")["order_id"].count()
print("\nNombre de commandes par type de paiement :")
print(payment_counts)

# Graphique barres
sns.barplot(x=payment_counts.values, y=payment_counts.index, palette="coolwarm")
plt.title("Nombre de commandes par type de paiement")
plt.xlabel("Nombre de commandes")
plt.ylabel("Type de paiement")
plt.show()

# Montant moyen par type de paiement
payment_mean = df.groupby("payment_type")["payment_value"].mean().sort_values(ascending=False)
print("\nMontant moyen par type de paiement :")
print(payment_mean)

# %% Section 10: Distribution des montants des commandes
print("\n--- Distribution des montants des commandes ---")

# Histogramme
sns.histplot(df["payment_value"], bins=50, kde=True)
plt.title("Distribution du montant des commandes")
plt.xlabel("Montant de la commande")
plt.ylabel("Nombre de commandes")
plt.show()

# Boxplot
sns.boxplot(x=df["payment_value"])
plt.title("Boxplot du montant des commandes (outliers)")
plt.xlabel("Montant de la commande")
plt.show()

# %% Section 11: Montant moyen des commandes par État
print("\n--- Montant moyen des commandes par État ---")

state_payment = (
    df.groupby("customer_state")["payment_value"]
    .mean()
    .sort_values(ascending=False)
)

print(state_payment)

sns.barplot(
    x=state_payment.values,
    y=state_payment.index
)
plt.title("Montant moyen des commandes par État")
plt.xlabel("Montant moyen")
plt.ylabel("État")
plt.show()

# %% Section 12: Retards de livraison par catégorie de produit
print("\n--- Retards de livraison par catégorie de produit ---")

# On enlève les NaN pour éviter les graphiques cassés
delay_by_category = df.dropna(subset=["delivery_delay", "product_category_name_english"])

sns.boxplot(
    data=delay_by_category,
    x="delivery_delay",
    y="product_category_name_english"
)
plt.title("Retards de livraison par catégorie de produit")
plt.xlabel("Jours de retard")
plt.ylabel("Catégorie de produit")
plt.show()

# %% Section 13: Retard moyen par mois
print("\n--- Retard moyen de livraison par mois ---")

monthly_delay = (
    df.groupby("order_month")["delivery_delay"]
    .mean()
)

print(monthly_delay.head())

sns.lineplot(
    x=monthly_delay.index.astype(str),
    y=monthly_delay.values,
    marker="o"
)
plt.title("Retard moyen de livraison par mois")
plt.xlabel("Mois")
plt.ylabel("Jours de retard")
plt.xticks(rotation=45)
plt.show()

# %% Section 14: Répartition des avis clients
print("\n--- Répartition des notes clients ---")

review_counts = df["review_score"].value_counts().sort_index()
print(review_counts)

sns.barplot(
    x=review_counts.index,
    y=review_counts.values
)
plt.title("Répartition des notes clients")
plt.xlabel("Note")
plt.ylabel("Nombre d'avis")
plt.show()

# %% Section 15: Note moyenne par catégorie de produit
print("\n--- Note moyenne par catégorie de produit ---")

review_by_category = (
    df.groupby("product_category_name_english")["review_score"]
    .mean()
    .sort_values(ascending=False)
)

print(review_by_category)

sns.barplot(
    x=review_by_category.values,
    y=review_by_category.index
)
plt.title("Note moyenne par catégorie de produit")
plt.xlabel("Note moyenne")
plt.ylabel("Catégorie de produit")
plt.show()


# %% Section X: Nettoyage et vérification des données
print("\n--- Nettoyage et vérification des données ---")

# Vérifier les doublons
num_duplicates = df.duplicated().sum()
print(f"Nombre de doublons dans le dataset : {num_duplicates}")

# Supprimer les doublons si nécessaire
if num_duplicates > 0:
    df = df.drop_duplicates()
    print("Doublons supprimés.")

# Vérification des valeurs manquantes par colonne
missing_values = df.isnull().sum()
print("\nValeurs manquantes par colonne :")
print(missing_values[missing_values > 0])

# Optionnel : on peut remplir ou supprimer certaines valeurs manquantes
# Exemple : remplir les valeurs manquantes de delivery_delay par 0 si nécessaire
if 'delivery_delay' in df.columns:
    df['delivery_delay'] = df['delivery_delay'].fillna(0)

# Vérification des types de données
print("\nTypes de données des colonnes :")
print(df.dtypes)

# Statistiques rapides pour les colonnes numériques
print("\nStatistiques descriptives des colonnes numériques :")
print(df.describe())

# Analyse rapide des montants et frais de livraison
print("\nMontant moyen et médian des commandes :")
if 'price' in df.columns and 'freight_value' in df.columns:
    print("Prix moyen :", df['price'].mean())
    print("Prix médian :", df['price'].median())
    print("Frais de port moyen :", df['freight_value'].mean())
    print("Frais de port médian :", df['freight_value'].median())

# Boxplot des montants pour visualiser les outliers
if 'price' in df.columns:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.boxplot(x=df['price'])
    plt.title("Distribution du prix des commandes")
    plt.xlabel("Prix")
    plt.show()

# Histogramme des montants
if 'price' in df.columns:
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title("Histogramme du prix des commandes")
    plt.xlabel("Prix")
    plt.ylabel("Nombre de commandes")
    plt.show()


# %% Section 6: Export dataset complet
os.makedirs(PROCESSED_DIR, exist_ok=True)
df.to_csv(os.path.join(PROCESSED_DIR, "olist_full_dataset.csv"), index=False)
print("\nDataset complet exporté dans :", PROCESSED_DIR)
