from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from analysis.feature_engineering import build_pipeline


def feature_importance(df):
    X = df.drop(columns=['price', 'days_listed', 'sold', 'list_id'], errors='ignore')
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_pipeline(LinearRegression())
    model.fit(X_train, y_train)

    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    return sorted(zip(X_test.columns, result.importances_mean), key=lambda x: x[1], reverse=True)


def print_final_report(df, hypothesis_results, reg_results, log_metrics):
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print("\n1. Number of vehicles in the dataset")
    print(len(df))

    print("\n2. Average price by brand (highest first)")
    print(df.groupby('brand')['price'].mean().sort_values(ascending=False).round(2))

    print("\n3. Features that affect price the most (permutation importance, Linear Regression)")
    for name, score in feature_importance(df)[:5]:
        print(f"{name:<15}: {score:.4f}")

    print("\n4. Correlation between age and mileage")
    print(round(df['age'].corr(df['mileage']), 4))

    print("\n5. Hypothesis test results")
    print("Transmission - price p-value:", hypothesis_results["transmission"]["p-value"])
    print("Age - price p-value         :", hypothesis_results["age"]["p-value"])
    print("Damage - price p-value      :", hypothesis_results["damage"]["p-value"])

    print("\n6-7. Regression model comparison")
    print(reg_results.round(4).to_string(index=False))

    print("\n8. Logistic regression performance")
    print("Accuracy:", round(log_metrics["accuracy"], 4))
    print("Recall  :", round(log_metrics["recall"], 4))
    print("F1 Score:", round(log_metrics["f1"], 4))

    print("\n9. False positives / false negatives")
    tn, fp, fn, tp = log_metrics["confusion_matrix"].ravel()
    print("False positives:", fp)
    print("False negatives:", fn)
    print("A false positive means the model says the car will sell, so the seller does not lower the price")
    print("and the listing waits for nothing. The cost is wasted time.")
    print("A false negative means the model says the car will not sell, so the seller may lower the price")
    print("unnecessarily. The cost is lost revenue.")

    print("\n10. Limitations")
    print("- The data is synthetic.")
    print("- Only 1500 listings and one train/test split.")
    print("- Sale status is observed only within 30 days, so cars sold later are labelled as not sold.")
