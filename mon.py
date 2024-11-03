# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import streamlit as st


# Charger le fichier Parquet
sales_data = pd.read_parquet('fusion_data.parquet', engine='pyarrow')

# Assurez-vous que la colonne 'date' est au format datetime
sales_data['date'] = pd.to_datetime(sales_data['date'])
sales_data['year'] = sales_data['date'].dt.year  # Extraire l'année à partir de la date

# Interface utilisateur Streamlit
st.title("Prévisions des Ventes des magasins Favorita", anchor="tableau_de_bord")

# **Prévisions des ventes par Famille de Produits**

# Créer des caractéristiques temporelles pour le modèle XGBoost
sales_data['day'] = sales_data['date'].dt.day
sales_data['month'] = sales_data['date'].dt.month
sales_data['dayofweek'] = sales_data['date'].dt.dayofweek

# Créer des décalages pour les ventes précédentes (n jours)
for i in range(1, 8):
    sales_data[f'sales_lag_{i}'] = sales_data.groupby('family')['sales'].shift(i)

# Supprimer les lignes avec des valeurs manquantes
sales_data.dropna(inplace=True)

# Préparer une liste pour stocker les résultats des prévisions futures par famille
forecast_results_family = []

# Boucle sur chaque famille de produits pour faire des prévisions
for family in sales_data['family'].unique():
    family_data = sales_data[sales_data['family'] == family]

    if family_data.empty:
        continue  # Si aucune donnée pour cette famille, passez à la suivante

    # Diviser en ensembles d'entraînement et de test
    X = family_data[['day', 'month', 'year', 'dayofweek'] + [f'sales_lag_{i}' for i in range(1, 8)]]
    y = family_data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Préparer les données pour les prévisions sur 15 jours
    future_dates = pd.date_range(start=family_data['date'].max() + pd.Timedelta(days=1), periods=15)
    future_data = []

    for date in future_dates:
        day = date.day
        month = date.month
        year = date.year
        dayofweek = date.dayofweek
        sales_lags = [family_data['sales'].iloc[-i] if len(family_data) >= i else 0 for i in range(1, 8)]
        future_data.append([day, month, year, dayofweek] + sales_lags)

    # Prédictions pour les 15 prochains jours
    future_predictions = model.predict(future_data)

    # Créer un DataFrame pour les prévisions de cette famille
    family_forecast_df = pd.DataFrame({
        'date': future_dates,
        'Ventes Prévues': future_predictions,
        'Famille': family  # Ajout de la colonne famille
    })

    # Ajouter les prévisions au résultat final
    forecast_results_family.append(family_forecast_df)

# Concatenation des résultats pour chaque famille dans un DataFrame final
forecast_df_family = pd.concat(forecast_results_family, ignore_index=True)

# Calculer les ventes totales prévues pour chaque produit (famille)
total_sales_by_family = forecast_df_family.groupby('Famille')['Ventes Prévues'].sum().reset_index()

# Trier les produits par ordre décroissant pour avoir les prévisions les plus élevées en haut du tableau
total_sales_by_family = total_sales_by_family.sort_values(by='Ventes Prévues', ascending=False)

# Afficher le tableau trié avec les produits ayant les prévisions les plus élevées en première position
st.subheader("Produits avec les Prévisions de Ventes les Plus Élevées ")
st.dataframe(total_sales_by_family.style.format({'Ventes Prévues': '{:.2f}'}).background_gradient(cmap='Oranges', low=0, high=1))

# Graphique des Prévisions par Famille
top_families = total_sales_by_family.head(5)  # Obtenez les 5 familles avec les ventes prévues les plus élevées
top_forecast_df = forecast_df_family[forecast_df_family['Famille'].isin(top_families['Famille'])]

fig_family_forecast = px.line(
    top_forecast_df,
    x='date',
    y='Ventes Prévues',
    color='Famille',
    title='Tendances des Ventes Prévues par Famille de Produits',
    labels={'Ventes Prévues': 'Ventes Prévues', 'date': 'Date'}
)
st.plotly_chart(fig_family_forecast)

# Prévisions des ventes par Magasin

# Préparer une liste pour stocker les résultats des prévisions futures par magasin
forecast_results_store = []

# Boucle sur chaque magasin pour faire des prévisions
for store in sales_data['store_nbr'].unique():
    store_data = sales_data[sales_data['store_nbr'] == store]

    if store_data.empty:
        continue  # Si aucune donnée pour ce magasin, passez au suivant

    # Diviser en ensembles d'entraînement et de test
    X = store_data[['day', 'month', 'year', 'dayofweek'] + [f'sales_lag_{i}' for i in range(1, 8)]]
    y = store_data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Préparer les données pour les prévisions sur 15 jours
    future_dates = pd.date_range(start=store_data['date'].max() + pd.Timedelta(days=1), periods=15)
    future_data = []

    for date in future_dates:
        day = date.day
        month = date.month
        year = date.year
        dayofweek = date.dayofweek
        sales_lags = [store_data['sales'].iloc[-i] if len(store_data) >= i else 0 for i in range(1, 8)]
        future_data.append([day, month, year, dayofweek] + sales_lags)

    # Prédictions pour les 15 prochains jours
    future_predictions = model.predict(future_data)

    # Créer un DataFrame pour les prévisions de ce magasin
    store_forecast_df = pd.DataFrame({
        'date': future_dates,
        'Ventes Prévues': future_predictions,
        'Magasin': store  # Ajout de la colonne magasin
    })

    # Ajouter les prévisions au résultat final
    forecast_results_store.append(store_forecast_df)

# Concatenation des résultats pour chaque magasin dans un DataFrame final
forecast_df_store = pd.concat(forecast_results_store, ignore_index=True)

# Calculer les ventes totales prévues pour chaque magasin
total_sales_by_store = forecast_df_store.groupby('Magasin')['Ventes Prévues'].sum().reset_index()

# Trier les magasins par ordre décroissant pour avoir les prévisions les plus élevées en haut du tableau
total_sales_by_store = total_sales_by_store.sort_values(by='Ventes Prévues', ascending=False)

# Afficher le tableau trié avec les magasins ayant les prévisions les plus élevées en première position
st.subheader("Magasins avec les Prévisions de Ventes les Plus Élevées")
st.dataframe(total_sales_by_store.style.format({'Ventes Prévues': '{:.2f}'}).background_gradient(cmap='Blues', low=0, high=1))

# Graphique des Prévisions par Magasin
fig_store_forecast = px.line(
    forecast_df_store,
    x='date',
    y='Ventes Prévues',
    color='Magasin',
    title='Tendances des Ventes Prévues par Magasin',
    labels={'Ventes Prévues': 'Ventes Prévues', 'date': 'Date'}
)
st.plotly_chart(fig_store_forecast)
