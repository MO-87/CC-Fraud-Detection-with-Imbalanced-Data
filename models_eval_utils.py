from sklearn.metrics import (f1_score,
                             precision_recall_curve,
                             auc, classification_report)


def model_eval(model, data, y_true, name):
    y_pred_proba = model.predict_proba(data)[:, 1]
    y_pred = model.predict(data)

    f1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    report = classification_report(y_true, y_pred)

    print(f"{name} F1-score : {f1:.3f} | {name} PR AUC : {pr_auc:.3f}")
    print("\n", report)
