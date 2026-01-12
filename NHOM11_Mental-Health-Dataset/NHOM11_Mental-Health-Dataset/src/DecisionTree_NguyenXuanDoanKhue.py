from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import pandas as pd

def train_dt_baseline(X_train, y_train):
    """
    Huấn luyện mô hình Decision Tree với tham số mặc định (Baseline)
    
    Công dụng:
    - Tạo một mô hình DT đơn giản làm điểm xuất phát (baseline).
    - Giới hạn max_depth=3 để cây ngắn, dễ giải thích và tránh overfit.
    - Sử dụng random_state=42 để kết quả huấn luyện có thể lặp lại.
    
    Tham số:
    - X_train: Tập features huấn luyện
    - y_train: Nhãn huấn luyện (Treatment: Yes/No)
    
    Trả về:
    - Mô hình DT đã huấn luyện (baseline)
    """
    print("--- Training Decision Tree Baseline ---")
    # Khởi tạo mô hình Decision Tree với độ sâu nhỏ = 3
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # feature_importances_ cho biết mức độ đóng góp của mỗi feature
    # vào việc giảm impurity (Gini / Entropy) trong quá trình chia nhánh
    importances = pd.Series(
        model.feature_importances_,          # Giá trị importance từ mô hình
        index=X_train.columns                # Tên các cột features
    )
    # Các feature có importance > 0 là các feature
    # thực sự được Decision Tree sử dụng để ra quyết định
    important_features = importances[importances > 0]
    print("Số features có tác động:", important_features.shape[0])
    print("\n=== Features có ảnh hưởng ===")
    print(important_features.sort_values(ascending=False))
    # Các feature có importance = 0:
    # - Không được sử dụng trong bất kỳ node nào
    # - Có thể là feature dư thừa hoặc gây nhiễu
    non_important_features = importances[importances == 0]
    print("Số features không ảnh hưởng:", non_important_features.shape[0])
    # LƯU MODEL + THÔNG TIN FEATURE
    if not os.path.exists('models'):
        os.makedirs('models')
    # Lưu cả model và metadata để:
    # - Phân tích lại feature sau
    # - So sánh với mô hình tuned
    save_path = 'models/decision_tree_khue_baseline.pkl'  
    save_data = {
        'tên':'===== DECISION TREE BASE =====',
        
        'so_luong_features_tham_gia':important_features.shape[0],
        'important_features': important_features.index.tolist(),  
        'so_luong_features_khong_tham_gia':non_important_features.shape[0],
        'zero_features': non_important_features.index.tolist(),
       
    }
    joblib.dump(save_data, save_path)
    print(f"✅ Đã lưu model Decision Tree Baseline + features quan trọng tại: {save_path}")
    return model

# ================================================
# Hàm huấn luyện Decision Tree Tuned (RandomizedSearchCV)
# ================================================
def train_dt_tuned(X_train, y_train):
    """
 
    Huấn luyện Decision Tree với tinh chỉnh siêu tham số bằng RandomizedSearchCV
    Và lưu mô hình tốt nhất ra file .pkl
    
    Công dụng:
    - Tìm kiếm ngẫu nhiên các siêu tham số tốt nhất để cải thiện hiệu suất mô hình.
    - Sử dụng RandomizedSearchCV thay vì GridSearchCV để tiết kiệm thời gian (thử 100 tổ hợp).
    - Tối ưu theo ROC AUC (phù hợp cho bài toán phân loại cân bằng nhưng cần chú trọng khả năng phân biệt Yes/No).
    - Lưu mô hình tốt nhất vào thư mục 'models' để sử dụng lại sau (deploy hoặc so sánh).
    
    Tham số:
    - X_train: Tập features huấn luyện
    - y_train: Nhãn huấn luyện
    
    Trả về:
    - Mô hình DT tốt nhất sau tuning (best_estimator_)
    
    Và LƯU MODEL ra file .pkl
    """
    print("--- Training Decision Tree Tuned (RandomizedSearchCV) ---")
   
  # Khoảng tham số để tìm kiếm ngẫu nhiên (rộng nhưng thực tế)
    param_dist = {
        "max_depth": [None, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70],              # Độ sâu cây
        "min_samples_split": [2, 5, 10, 20, 50, 100, 110, 120, 150, 200],         # Số mẫu tối thiểu để split node
        "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100, 110, 120, 150, 200],       # Số mẫu tối thiểu ở lá
        "criterion": ["gini", "entropy"]                                           # Đo lường impurity
    }
# Khởi tạo RandomizedSearchCV
    search = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),   # Mô hình cơ sở
        param_distributions=param_dist,            # Khoảng tham số
        n_iter=100,                                # Thử ngẫu nhiên 100 tổ hợp (có thể tăng nếu máy mạnh)
        cv=5,                                      # 5-fold cross-validation
        scoring="roc_auc",                         # Tiêu chí đánh giá: AUC tốt cho bài toán này
        random_state=42,                           # Cố định kết quả random
        n_jobs=-1,                                 # Chạy song song trên tất cả lõi CPU
       
    )
    # Thực hiện tìm kiếm và huấn luyện
    print("Đang tuning... (có thể mất vài phút đến vài chục phút tùy máy)")
    search.fit(X_train, y_train)
    # Lấy mô hình tốt nhất
    print("Decision Tree best params:", search.best_params_)
   
    best_model = search.best_estimator_
     # PHÂN TÍCH FEATURE IMPORTANCE SAU TUNING
    importances_tuned = pd.Series(
        best_model.feature_importances_,
        index=X_train.columns
    )
    important_features = importances_tuned[importances_tuned > 0]
    print("Số features thực sự được DT sử dụng:", important_features.shape[0])
    zero_features = importances_tuned[importances_tuned == 0]
    print("Số features không ảnh hưởng:", zero_features.shape[0])
    print("\n=== Features không ảnh hưởng ===")
    print(zero_features.index.tolist())
   
    # ===== PHẦN QUAN TRỌNG: LƯU MODEL =====
    # Tạo thư mục models nếu chưa có
   
    if not os.path.exists('models'):
        os.makedirs('models')
       
    # Lưu file với tên riêng biệt để không bị trùng với model khác
    # Bạn Khue làm Decision Tree thì đặt tên file là decision_tree.pkl
   
    save_result = 'models/decision_tree_khue_tuned.pkl'
    # kết quả chi tiết
    save_path='models/decision_tree_khue.pkl'  
    joblib.dump(best_model, save_path)
    print(f"✅ Đã lưu thông số tại: {save_path}")
    save_data = {
        'tên':'===== DECISION TREE TUNING =====',
        
        'số lượng features  tham gia vào':important_features.shape[0],
        'important_features': important_features.index.tolist(),  
        'số lượng features không  tham gia vào':zero_features.shape[0],
        'zero_features': zero_features.index.tolist(),
       
    }
   
    joblib.dump(save_data, save_result)
    print(f"✅ Đã lưu model Decision Tree tại: {save_result}")
   
    return best_model