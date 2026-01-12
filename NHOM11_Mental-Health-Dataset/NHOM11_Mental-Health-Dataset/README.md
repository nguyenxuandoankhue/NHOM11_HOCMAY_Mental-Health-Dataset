# PHÂN TÍCH VÀ DỰ ĐOÁN SỨC KHỎE TÂM THẦN (NHÓM 11)

## 1. Giới thiệu
Dự án ứng dụng Machine Learning (Decision Tree, Random Forest, XGBoost, LightGBM) để dự đoán nguy cơ sức khỏe tâm thần của nhân viên, tích hợp Web App hỗ trợ doanh nghiệp sàng lọc.

## 2. Cài đặt môi trường
Yêu cầu: Python 3.8 trở lên.
Mở Terminal tại thư mục gốc dự án và chạy lệnh:

pip install -r requirements.txt

## 3. Hướng dẫn chạy chương trình

### Cách 1: Chạy Web App (Demo)
Dùng lệnh sau để mở giao diện web:

python -m streamlit run src/app.py


### Cách 2: Huấn luyện lại mô hình (Train Model)
Để huấn luyện lại và sinh ra các file model (.pkl) mới nhất cùng báo cáo biểu đồ:

python src/main.py

Kết quả huấn luyện sẽ được lưu trong thư mục `results/`.

## 4. Cấu trúc thư mục
- `data/`: Chứa file dữ liệu csv.
- `src/`: Mã nguồn (Xử lý dữ liệu, Train model, Web App).
- `models/`: Chứa các file model đã huấn luyện (.pkl).
- `results/`: Chứa báo cáo kết quả và biểu đồ.
