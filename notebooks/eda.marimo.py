# eda.marimo.py
# ---------------------------------
# EDA et préparation du dataset Olist pour ML / Power BI
# ---------------------------------

import pandas as pd
import os

# ---------------------------------
# 1. Chargement des données
# ---------------------------------

# Chemin absolu du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # remonte à la racine
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

print("Chemin RAW_DIR :", RAW_DIR)
print("Fichiers disponibles :", os.listdir(RAW_DIR))

os.makedirs(PROCESSED_DIR, exist_ok=True)


# Chargements des données 

orders = pd.read_csv(os.path.join(RAW_DIR, "olist_orders_dataset.csv"))
order_items = pd.read_csv(os.path.join(RAW_DIR, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(RAW_DIR, "olist_products_dataset.csv"))
customers = pd.read_csv(os.path.join(RAW_DIR, "olist_customers_dataset.csv"))
sellers = pd.read_csv(os.path.join(RAW_DIR, "olist_sellers_dataset.csv"))
payments = pd.read_csv(os.path.join(RAW_DIR, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(RAW_DIR, "olist_order_reviews_dataset.csv"))

# Jointures principales

df = orders.merge(order_items, on="order_id", how="left") \
           .merge(products, on="product_id", how="left") \
           .merge(customers, on="customer_id", how="left") \
           .merge(payments, on="order_id", how="left") \
           .merge(reviews, on="order_id", how="left")

# Aperçu rapide

print("Dataset final :")
print(df.head())
print("\nShape :", df.shape)

# Analyses simples

# Nombre de commandes par état
orders_per_state = df.groupby("customer_state")["order_id"].nunique().sort_values(ascending=False)
print("\nNombre de commandes par état :")
print(orders_per_state.head())

# Top 10 produits les plus vendus
top_products = df.groupby("product_id")["order_id"].count().sort_values(ascending=False).head(10)
print("\nTop 10 produits les plus vendus :")
print(top_products)

# Retard moyen de livraison (en jours)
df["shipping_delay_days"] = (pd.to_datetime(df["order_approved_at"]) - pd.to_datetime(df["order_purchase_timestamp"])).dt.days
print("\nRetard moyen de livraison (jours) :", df["shipping_delay_days"].mean())

# -----------------------------
# Export dataset final
# -----------------------------
df.to_csv(os.path.join(PROCESSED_DIR, "olist_full_dataset.csv"), index=False)
print("\nDataset complet exporté dans :", PROCESSED_DIR)


