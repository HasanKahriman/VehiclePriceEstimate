import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from analysis.feature_engineering import build_pipeline


def train_and_evaluate_model(df):
    df = df.copy()

    drop_cols = ['price', 'days_listed', 'sold', 'list_id']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42, max_iter=10000),
        "ElasticNet": ElasticNet(random_state=42, max_iter=10000),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    }
    pipelines = {name: build_pipeline(model) for name, model in models.items()}

    results = []

    for model_name, pipe in pipelines.items():
        # Model selection uses cross-validation on the training set only
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                                   scoring="neg_root_mean_squared_error").mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results.append({
            "Model": model_name,
            "CV_RMSE": cv_rmse,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        })

    df_results = pd.DataFrame(results)
    print("\n--- MODEL COMPARISON TABLE ---")
    print(f"{'Model':<25} | {'CV RMSE':<15} | {'MAE':<15} | {'RMSE':<15} | {'R2':<10}")
    print("-" * 92)

    for index, row in df_results.iterrows():
        print(f"{row['Model']:<25} | {row['CV_RMSE']:<15,.2f} | {row['MAE']:<15,.2f} | "
              f"{row['RMSE']:<15,.2f} | {row['R2']:<10,.4f}")

    best_name = df_results.loc[df_results["CV_RMSE"].idxmin(), "Model"]
    print(f"\nSelected model (lowest CV RMSE): {best_name}")

    return pipelines[best_name], df_results
