from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

# Hàm huấn luyện mô hình LightGBM với bộ tham số mặc định (Baseline)
def train_lgb_baseline(X_train, y_train):
    print("--- Training LightGBM Default ---")
    model = LGBMClassifier(
        objective="binary",
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model

# Hàm huấn luyện mô hình LightGBM với tinh chỉnh siêu tham số bằng RandomizedSearchCV
def train_lgb_tuned(X_train, y_train):
    print("--- Training LightGBM Tuned (RandomizedSearchCV) ---")
    
    #Tạo lưới tham số
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 3, 5, 7, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }
    #Tìm bộ tham số tốt nhất
    search = RandomizedSearchCV(
        LGBMClassifier(
            objective="binary",
            random_state=42,
            verbosity=-1
        ),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print("LightGBM best params:", search.best_params_)
    best_model = search.best_estimator_

    # ===== PHẦN QUAN TRỌNG: LƯU MODEL =====
    # Tạo thư mục models nếu chưa có
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Lưu file với tên riêng biệt
    save_path = 'models/lightgbm_phi.pkl'
    joblib.dump(best_model, save_path)
    print(f"Đã lưu model LightGBM tại: {save_path}")
    
    return best_model