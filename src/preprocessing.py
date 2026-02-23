import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def preprocess_data(input_path, output_path):
    print(f"Chargement des données depuis : {input_path}")
    df = pd.read_csv(input_path)

    # 1. Imputation de l'Age (Médiane car 30% de manquants)
    # On utilise la médiane car elle n'est pas influencée par les valeurs extrêmes
    if 'Age' in df.columns:
        imputer = SimpleImputer(strategy='median')
        df['Age'] = imputer.fit_transform(df[['Age']])

    # 2. Nettoyage de SupportTicketsCount (Anomalies 999, -1)
    # On garde uniquement les valeurs entre 0 et 20 (seuil réaliste)
    if 'SupportTicketsCount' in df.columns:
        df = df[(df["SupportTicketsCount"] >= 0) & (df["SupportTicketsCount"] <= 20)]

    # 3. Nettoyage de SatisfactionScore (Anomalies 99, -1)
    if 'SatisfactionScore' in df.columns:
        # Créer un indicateur pour les valeurs qui étaient fausses (utile pour le ML)
        df["Satisfaction_was_invalid"] = ((df["SatisfactionScore"] < 1) | (df["SatisfactionScore"] > 5)).astype(int)
        
        # Remplacer les valeurs hors limites par NaN puis imputer par la médiane (3.0)
        df.loc[(df["SatisfactionScore"] < 1) | (df["SatisfactionScore"] > 5), "SatisfactionScore"] = np.nan
        df["SatisfactionScore"] = df["SatisfactionScore"].fillna(df["SatisfactionScore"].median())

    # 4. Parsing des Dates (Gestion des 3 formats différents)
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'], 
            dayfirst=True, 
            errors='coerce'
        )
        # Extraire l'année et le mois (les modèles ne lisent pas les dates brutes)
        df['RegYear'] = df['RegistrationDate'].dt.year
        df['RegMonth'] = df['RegistrationDate'].dt.month
        # Remplir les quelques dates qui auraient échoué au parsing
        df['RegYear'] = df['RegYear'].fillna(df['RegYear'].median())
        df['RegMonth'] = df['RegMonth'].fillna(df['RegMonth'].median())

    # 5. Feature Engineering (Demandé dans le projet)
    # Calcul de la valeur monétaire par jour d'ancienneté
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # 6. Suppression des colonnes inutiles ou constantes
    cols_to_drop = ['NewsletterSubscribed', 'RegistrationDate', 'LastLoginIP']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing terminé. Fichier sauvegardé dans : {output_path}")
    return df

# --- Exécution ---
if __name__ == "__main__":
    raw_data_path = 'data/raw/data.csv'
    processed_data_path = 'data/processed/retail_customers_PREPROCESSED.csv'
    
    if os.path.exists(raw_data_path):
        preprocess_data(raw_data_path, processed_data_path)
    else:
        print(f"Erreur : Le fichier {raw_data_path} est introuvable.")