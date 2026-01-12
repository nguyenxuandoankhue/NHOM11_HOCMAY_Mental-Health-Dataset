import pandas as pd
import eda
import os
from sklearn.model_selection import train_test_split
from preprocessing import load_data, preprocess_data
from XGBoost_VanPhuocNhatPhi import train_xgb_baseline, train_xgb_tuned
from LightGBM_VanPhuocNhatPhi import train_lgb_baseline, train_lgb_tuned
from RandomForest_DangAnhTuong import train_rf_baseline, train_rf_tuned
from DecisionTree_NguyenXuanDoanKhue import train_dt_baseline, train_dt_tuned
from evaluation import evaluate_model, plot_accuracy_comparison, plot_confusion_matrix, plot_roc_comparison

def main():
    print(">>> MAIN STARTED <<<")
    
    # 1. TẠO THƯ MỤC LƯU KẾT QUẢ 
    if not os.path.exists("results"):
        os.makedirs("results")
        print("Đã tạo thư mục 'results' để lưu báo cáo.")

    # 2. LOAD & PREPROCESS
    DATA_PATH = "data/Mental_Health_dataset.csv" 
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "data/survey.csv"
        
    print(f"Loading data from: {DATA_PATH}")
    try:
        df = load_data(DATA_PATH)
        
        print(">>> Đang chạy phân tích EDA (Vẽ biểu đồ)...")
        try:
            eda.plot_and_save_gender_vs_treatment(df)
        except Exception as e_eda:
            print(f"Không thể chạy EDA: {e_eda}")
            
        df = preprocess_data(df)    
    except Exception as e:
        print(f"Lỗi load data: {e}")
        return


    # 3. SPLIT DATA
    TARGET_COL = "Treatment"
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. TRAIN & EVALUATE
    trained_models = {}
    results_list = []

    with open("results/training_log.txt", "w", encoding="utf-8") as f:
        f.write("=== KẾT QUẢ HUẤN LUYỆN MODEL ===\n\n")

        # ===== Decision Tree =====
        # Decision Tree Baseline
        dt_base = train_dt_baseline(X_train, y_train)
        res_dt_base = evaluate_model(dt_base, X_test, y_test, "Decision Tree (Baseline)")
        results_list.append(res_dt_base)
        trained_models["DT Base"] = dt_base
        f.write("Decision Tree (Baseline)\n")
        f.write(f"Accuracy : {res_dt_base['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_dt_base['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_dt_base['report'])
        f.write("\n\n")
        # Decision Tree Tunning
        dt_tuned = train_dt_tuned(X_train, y_train)
        res_dt_tuned = evaluate_model(dt_tuned, X_test, y_test, "Decision Tree (Tuned)")
        results_list.append(res_dt_tuned)
        trained_models["DT Tuned"] = dt_tuned
        f.write("Decision Tree (Tuned)\n")
        f.write(f"Accuracy : {res_dt_tuned['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_dt_tuned['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_dt_tuned['report'])
        f.write("\n\n")

        # ===== Random Forest =====
        #Random Forest Baseline
        rf_base = train_rf_baseline(X_train, y_train)
        res_rf_base = evaluate_model(rf_base, X_test, y_test, "Random Forest (Baseline)")
        results_list.append(res_rf_base)
        trained_models["RF Base"] = rf_base
        f.write("Random Forest (Baseline)\n")
        f.write(f"Accuracy : {res_rf_base['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_rf_base['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_rf_base['report'])
        f.write("\n\n")
        #Random Forest Tunning
        rf_tuned = train_rf_tuned(X_train, y_train)
        res_rf_tuned = evaluate_model(rf_tuned, X_test, y_test, "Random Forest (Tuned)")
        results_list.append(res_rf_tuned)
        trained_models["RF Tuned"] = rf_tuned
        f.write("Random Forest (Tuned)\n")
        f.write(f"Accuracy : {res_rf_tuned['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_rf_tuned['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_rf_tuned['report'])
        f.write("\n\n")

        # ===== LightGBM =====
        #LightGBM Baseline
        lgb_base = train_lgb_baseline(X_train, y_train)
        res_lgb_base = evaluate_model(lgb_base, X_test, y_test, "LightGBM (Baseline)")
        results_list.append(res_lgb_base)
        trained_models["LGB Base"] = lgb_base
        f.write("LightGBM (Baseline)\n")
        f.write(f"Accuracy : {res_lgb_base['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_lgb_base['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_lgb_base['report'])
        f.write("\n\n")
        #LightGBM Tunning
        lgb_tuned = train_lgb_tuned(X_train, y_train)
        res_lgb_tuned = evaluate_model(lgb_tuned, X_test, y_test, "LightGBM (Tuned)")
        results_list.append(res_lgb_tuned)
        trained_models["LGB Tuned"] = lgb_tuned
        f.write("LightGBM (Tuned)\n")
        f.write(f"Accuracy : {res_lgb_tuned['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_lgb_tuned['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_lgb_tuned['report'])
        f.write("\n\n")

        # ===== XGBoost =====
        #XGBoost Baseline
        xgb_base = train_xgb_baseline(X_train, y_train)
        res_xgb_base = evaluate_model(xgb_base, X_test, y_test, "XGBoost (Baseline)")
        results_list.append(res_xgb_base)
        trained_models["XGB Base"] = xgb_base
        f.write("XGBoost (Baseline)\n")
        f.write(f"Accuracy : {res_xgb_base['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_xgb_base['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_xgb_base['report'])
        f.write("\n\n")
        #XGBoost Tunning
        xgb_tuned = train_xgb_tuned(X_train, y_train)
        res_xgb_tuned = evaluate_model(xgb_tuned, X_test, y_test, "XGBoost (Tuned)")
        results_list.append(res_xgb_tuned)
        trained_models["XGB Tuned"] = xgb_tuned
        f.write("XGBoost (Tuned)\n")
        f.write(f"Accuracy : {res_xgb_tuned['accuracy']:.4f}\n")
        f.write(f"ROC AUC  : {res_xgb_tuned['roc_auc']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(res_xgb_tuned['report'])
        f.write("\n")

    print("\nĐã ghi lại kết quả số vào file: results/training_log.txt")

    # 5. VẼ BIỂU ĐỒ VÀ LƯU RA FILE ẢNH
    print("\nĐang vẽ và lưu biểu đồ...")

    # Lưu Confusion Matrix
    for res in results_list:
        fig = plot_confusion_matrix(res['cm'], title=res['name'])
        safe_name = res['name'].replace(" ", "_")
        save_path = f"results/cm_{safe_name}.png"
        fig.savefig(save_path)
        print(f"   -> Đã lưu: {save_path}")

    # Lưu biểu đồ so sánh Accuracy
    fig_acc = plot_accuracy_comparison(trained_models, X_test, y_test)
    fig_acc.savefig("results/model_acr_comparison.png")
    print("   -> Đã lưu: results/model_acr_comparison.png")

    fig_acc = plot_roc_comparison(trained_models, X_test, y_test)
    fig_acc.savefig("results/model_roc_comparison.png")
    print("   -> Đã lưu: results/mode_roc_comparison.png")

    print("\nHOÀN TẤT TOÀN BỘ! Hãy mở thư mục 'results' để lấy dữ liệu báo cáo.")

if __name__ == "__main__":
    main()