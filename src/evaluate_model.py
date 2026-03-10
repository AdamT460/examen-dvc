import pandas as pd
import pickle
import json

from sklearn.metrics import mean_squared_error, r2_score

# charger les données
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

# charger le modèle
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# prédictions
predictions = model.predict(X_test)

# métriques
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

scores = {
    "mse": mse,
    "r2": r2
}

# sauvegarder les métriques
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f)

# sauvegarder les prédictions
pred_df = pd.DataFrame({
    "y_true": y_test.values.ravel(),
    "y_pred": predictions
})

pred_df.to_csv("data/predictions.csv", index=False)
