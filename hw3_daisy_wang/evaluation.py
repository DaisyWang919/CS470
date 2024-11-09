import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data_path = 'data.csv'
data = pd.read_csv(data_path)

for col in data.select_dtypes(include=['object']).columns:
    if col != 'Has heart disease? (Prediction Target)':
        data[col] = data[col].astype('category').cat.codes

data['Has heart disease? (Prediction Target)'] = data['Has heart disease? (Prediction Target)'].map({'Yes': 1, 'No': 0})

X = data.drop(columns=['Has heart disease? (Prediction Target)', 'person ID'])
y = data['Has heart disease? (Prediction Target)']

model = DecisionTreeClassifier(max_depth=20, min_samples_split=2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

specificities = []
sensitivities = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    specificities.append(specificity)
    sensitivities.append(sensitivity)

print("Evaluation Results (Cross-Validation):")
print(f"Accuracy: {np.mean(cv_results['test_accuracy']):.2f} ± {np.std(cv_results['test_accuracy']):.2f}")
print(f"Precision: {np.mean(cv_results['test_precision']):.2f} ± {np.std(cv_results['test_precision']):.2f}")
print(f"Recall: {np.mean(cv_results['test_recall']):.2f} ± {np.std(cv_results['test_recall']):.2f}")
print(f"F1-Score: {np.mean(cv_results['test_f1']):.2f} ± {np.std(cv_results['test_f1']):.2f}")
print(f"Specificity: {np.mean(specificities):.2f} ± {np.std(specificities):.2f}")
print(f"Sensitivity: {np.mean(sensitivities):.2f} ± {np.std(sensitivities):.2f}")
