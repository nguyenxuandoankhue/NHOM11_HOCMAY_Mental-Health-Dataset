from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
#Huấn luyện mô hình Random Forest với bộ tham số mặc định
def train_rf_baseline(X_train, y_train):
    print("-Training Random Forest Baseline ")
    model = RandomForestClassifier(
        n_estimators=100,       
        max_depth=None,         
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

#Huấn luyện mô hình Random Forest với tinh chỉnh siêu tham số Và LƯU MODEL ra file .pkl
def train_rf_tuned(X_train, y_train):
    print("--- Training Random Forest Tuned")
    
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        param_distributions=param_dist,
        n_iter=5,          
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("Random Forest best params:", search.best_params_)
    
    best_model = search.best_estimator_

    #LƯU MODEL 
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Đặt tên file và lưu file pkl
    save_path = 'models/random_forest_tuong.pkl'
    joblib.dump(best_model, save_path)
    print(f" ---Đã lưu model Random Forest tại: {save_path}")

    return best_model