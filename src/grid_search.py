import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# charger les données
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# modèle
model = RandomForestRegressor()

# grille de paramètres
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

# grid search
grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train.values.ravel())

# récupérer les meilleurs paramètres
best_params = grid.best_params_

# sauvegarde
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)
