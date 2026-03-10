import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

# charger les données
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# charger les meilleurs paramètres
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# créer le modèle avec ces paramètres
model = RandomForestRegressor(**best_params)

# entraîner le modèle
model.fit(X_train, y_train.values.ravel())

# sauvegarder le modèle
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
