from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score

from analysis.feature_engineering import build_pipeline


def prediction_model(df):
    df = df.copy()

    drop_cols = ['days_listed', 'sold', 'list_id']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_pipeline(LogisticRegression(max_iter=1000), extra_numeric=["price"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall  : {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix")
    print(conf_matrix)
    print("\nClassification Report")
    print(class_report)

    metrics = {"accuracy": accuracy, "recall": recall, "f1": f1, "confusion_matrix": conf_matrix}
    return model, metrics
