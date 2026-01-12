from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Hàm đánh giá, trả về Accuracy, ROC_AUC,...
def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    except:
        y_proba = None
        roc = "N/A"

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print(f"\n---- {title} ----")
    print("Accuracy:", acc)
    print("ROC AUC :", roc)
    print("\nClassification Report:\n", cr)

    return {
        "name": title,
        "accuracy": acc,
        "roc_auc": roc,
        "cm": cm,
        "report": cr
    }

#Hàm vẽ Confusion Matrix
def plot_confusion_matrix(cm, title="Model"):

    # Tạo figure và axes rõ ràng
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Chọn màu dựa trên tên (Tùy chọn cho đẹp)
    cmap = "Blues" if "Baseline" in title else "Greens"
   
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=['No Treatment', 'Treatment'],
        yticklabels=['No Treatment', 'Treatment'],
        ax=ax 
    )

    ax.set_title(f"Confusion Matrix - {title}")
    ax.set_xlabel("Predicted Recommendation")
    ax.set_ylabel("Actual History")
    
    return fig 

#Hàm hiển thị Accuracy theo cột để so sánh
def plot_accuracy_comparison(models_dict, X_test, y_test):
    model_names = []
    accuracies = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_names.append(name)
        accuracies.append(acc)

    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, accuracies, color=colors)
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Comparison of Model Accuracy")

    # Hiển thị số liệu trên cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    return fig

#Hàm hiển thị ROC_AUC theo cột để so sánh
def plot_roc_comparison(models_dict, X_test, y_test):
    model_names = []
    roc_aucs = []

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
        else:
            
            continue

        model_names.append(name)
        roc_aucs.append(roc_auc)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, roc_aucs, color=colors)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC Score")
    ax.set_title("Comparison of Model ROC-AUC")

    # Hiển thị giá trị trên cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,height + 0.01,
            f"{height:.4f}",
            ha="center",va="bottom",fontweight="bold")

    return fig