import joblib
import pandas as pd
from catboost import CatBoostClassifier


if __name__ == "__main__":
    df = pd.read_csv('spaceship_transformed.csv')
    x, y = df.drop(["Transported"], axis=1), df.Transported
    model = CatBoostClassifier(verbose=False)
    model.fit(x, y)
    joblib.dump(model, "./model.joblib")
