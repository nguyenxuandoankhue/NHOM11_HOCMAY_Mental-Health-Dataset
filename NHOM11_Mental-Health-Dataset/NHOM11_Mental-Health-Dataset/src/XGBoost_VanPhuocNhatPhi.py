from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

#Hàm huấn luyện mô hình XGBoost với tham số mặc định
def train_xgb_baseline(X_train, y_train):
    print("--- Training XGBoost Default ---")
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

#Hàm huấn luyện mô hình XGBoost với tinh chỉnh siêu tham số
def train_xgb_tuned(X_train, y_train):
    print("--- Training XGBoost Tuned (RandomizedSearchCV) ---")
    
    #Lưới tham số
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.8]
    }
    #Tìm bộ tham số tốt nhất
    search = RandomizedSearchCV(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print("XGBoost best params:", search.best_params_)
    best_model = search.best_estimator_

    # ===== PHẦN QUAN TRỌNG: LƯU MODEL =====
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Đặt tên file (Mình đặt là xgboost_phi.pkl cho đồng bộ)
    save_path = 'models/xgboost_phi.pkl'
    joblib.dump(best_model, save_path)
    print(f"Đã lưu model XGBoost tại: {save_path}")
    
    return best_model