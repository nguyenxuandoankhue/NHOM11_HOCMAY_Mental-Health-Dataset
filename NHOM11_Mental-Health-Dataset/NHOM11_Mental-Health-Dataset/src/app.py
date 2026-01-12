import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

#1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="AI Sàng Lọc Tâm Lý - Group 11",
    page_icon="🧠",
    layout="wide"
)

# CSS 1 tí
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# 2. HÀM LOAD MODEL
@st.cache_resource
def load_models():
    models = {}
    # Đường dẫn linh hoạt
    if os.path.exists("models"):
        model_dir = "models"
    elif os.path.exists("../models"):
        model_dir = "../models"
    else:
        st.error("❌ Không tìm thấy thư mục models! Hãy chạy main.py trước.")
        return {}

    model_files = {
        "Random Forest": "random_forest_tuong.pkl",
        "Decision Tree": "decision_tree_khue.pkl", 
        "LightGBM": "lightgbm_phi.pkl",
        "XGBoost": "xgboost_phi.pkl"
    }
    
    count = 0
    for name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                count += 1
            except Exception as e:
                pass # Bỏ qua nếu lỗi load 1 model
    
    if count == 0:
        st.error("⚠️ Không load được model nào. Vui lòng kiểm tra lại thư mục models.")
        return {}
        
    return models

loaded_models = load_models()

# 3. HÀM XỬ LÝ DỮ LIỆU
def preprocess_input(input_df, model):
    df = input_df.copy()
    
    # Map Binary
    binary_map = {'Yes': 1, 'No': 0}
    cols_to_map = ['FamilyHistory', 'SelfEmployed', 'CopingStruggles']
    for col in cols_to_map:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
            
    # Map Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
        
    # Map Ordinal
    days_map = {'Go out Every day': 0, '1-14 days': 1, '15-30 days': 2, '31-60 days': 3, 'More than 2 months': 4}
    level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    care_map = {'No': 0, 'Not sure': 1, 'Yes': 2}
    
    if 'DaysIndoors' in df.columns:
        df['DaysIndoors'] = df['DaysIndoors'].map(days_map)
    if 'MoodSwings' in df.columns:
        df['MoodSwings'] = df['MoodSwings'].map(level_map)
    if 'CareOptions' in df.columns:
        df['CareOptions'] = df['CareOptions'].map(care_map)

    # Align Columns 
    if hasattr(model, "feature_names_in_"):
        model_cols = model.feature_names_in_
        
        df_final = pd.DataFrame(columns=model_cols)
        for col in df.columns:
            if col in df_final.columns:
                df_final[col] = df[col]
                
        df_final = df_final.fillna(0)
        return df_final
    else:
        return df

# 4. GIAO DIỆN CHÍNH
st.title("🩺 Hệ Thống Sàng Lọc Sức Khỏe Tâm Thần (AI)")
st.markdown("---")

if not loaded_models:
    st.stop()

# TẠO TABS
tab1, tab2 = st.tabs(["👤 Người Dùng nhập Input", "👤 Người Dùng nhập File"])


# TAB 1: Người Dùng nhập Input
with tab1:
    st.subheader("📝 Kiểm tra sức khỏe tâm thần ")
    st.info("Hệ thống sử dụng 4 Mô hình AI  để đưa ra kết quả chính xác nhất.")
    
    with st.form("personal_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Tuổi", 18, 80, 25)
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            family = st.selectbox("Tiền sử gia đình?", ["No", "Yes"])
        with c2:
            days = st.selectbox("Thời gian ở trong nhà", ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months'])
            mood = st.selectbox("Thay đổi tâm trạng", ["Low", "Medium", "High"])
            care = st.selectbox("Biết về điều trị?", ["No", "Not sure", "Yes"])
        with c3:
            self_emp = st.selectbox("Làm tự do?", ["No", "Yes"])
            coping = st.selectbox("Khó khăn khi gặp áp lực?", ["No", "Yes"])

        submitted = st.form_submit_button("🚀 PHÂN TÍCH NGAY")

    if submitted:
        # Chuẩn bị dữ liệu
        raw_df = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'FamilyHistory': [family],
            'CareOptions': [care], 'DaysIndoors': [days], 'MoodSwings': [mood],
            'SelfEmployed': [self_emp], 'CopingStruggles': [coping]
        })
        
        # Chạy dự đoán với 4 model
        model_results = {}
        model_probs = {}
        
        for name, model in loaded_models.items():
            try:
                processed = preprocess_input(raw_df, model)
                pred = model.predict(processed)[0]
                model_results[name] = pred
                
                # Lấy xác suất nếu có
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(processed)[0][1]
                else:
                    prob = float(pred)
                model_probs[name] = prob
            except:
                model_results[name] = 0
                model_probs[name] = 0.0

        # Tổng hợp kết quả (Voting)
        avg_risk = np.mean(list(model_probs.values()))
        vote_count = sum(list(model_results.values()))
        
        st.divider()
        col_res, col_chart = st.columns([1, 1])
        
        with col_res:
            st.markdown("### 🏁 Kết Luận Của Hội Đồng AI")
            if vote_count >= 2: # Nếu >= 2 model báo bệnh
                st.error(f"🚨 NGUY CƠ CAO (Độ tin cậy: {avg_risk*100:.1f}%)")
                st.write(f"⚠️ Có **{vote_count}/4** mô hình cảnh báo bạn nên đi khám.")
                st.markdown("**Lời khuyên:** Bạn nên sắp xếp thời gian gặp bác sĩ tâm lý để được tư vấn.")
            else:
                st.success(f"✅ BÌNH THƯỜNG (Độ tin cậy: {(1-avg_risk)*100:.1f}%)")
                st.write(f"👍 Chỉ có **{vote_count}/4** mô hình cảnh báo.")
                st.markdown("**Lời khuyên:** Hãy duy trì lối sống lành mạnh hiện tại.")
                
            # Chi tiết
            with st.expander("Xem chi tiết từng mô hình"):
                st.dataframe(pd.DataFrame({
                    "Mô hình": list(model_results.keys()),
                    "Dự đoán": ["Nguy cơ" if x==1 else "An toàn" for x in model_results.values()],
                    "Xác suất rủi ro tâm lí": [f"{x*100:.1f}%" for x in model_probs.values()]
                }))

        with col_chart:
            st.markdown("### 📊 So Sánh Độ Tin Cậy")
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#ff4b4b' if x > 0.5 else '#00cc96' for x in model_probs.values()]
            bars = ax.bar(model_probs.keys(), model_probs.values(), color=colors)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Xác suất rủi ro")
            plt.xticks(rotation=15)
            
            # Số trên cột
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)

# TAB 2: Người dùng nhập File (UPLOAD FILE )
with tab2:
    st.subheader("📂 Phân Tích Hàng Loạt ")
    st.info("Tính năng nâng cao: Sử dụng cả 4 mô hình AI để rà soát danh sách người cần kiểm tra.")
    
    uploaded_file = st.file_uploader("Tải lên file CSV danh sách người", type=["csv"])
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Đã tải lên: {len(df_upload)} người")
        
        if st.button("🚀 Chạy Quét Rủi Ro (Hội Đồng AI)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Tạo các cột chứa kết quả dự đoán của từng model
            for idx, (name, model) in enumerate(loaded_models.items()):
                status_text.text(f"Đang chạy mô hình {name}...")
                processed = preprocess_input(df_upload, model)
                df_upload[f'Pred_{name}'] = model.predict(processed)
                progress_bar.progress((idx + 1) * 20)
                
            status_text.text("Đang tổng hợp kết quả (Voting)...")
            
            # Tính tổng phiếu bầu (Voting)
            # Cộng các cột dự đoán lại: Pred_RF + Pred_XGB ...
            pred_cols = [f'Pred_{name}' for name in loaded_models.keys()]
            df_upload['Total_Votes'] = df_upload[pred_cols].sum(axis=1)
            
            # Quy tắc: Nếu >= 2 phiếu bầu (trên 4) -> Nên đi khám (Cảnh giác cao)
            # Hoặc >= 3 phiếu (Nếu muốn chặt chẽ hơn). Ở đây chọn >=2 để an toàn (sàng lọc mà).
            df_upload['Final_Result'] = df_upload['Total_Votes'].apply(lambda x: 1 if x >= 2 else 0)
            df_upload['Khuyen_Nghi'] = df_upload['Final_Result'].map({1: '⚠️ NÊN ĐI KHÁM', 0: '✅ Bình thường'})
            
            progress_bar.progress(100)
            status_text.text("✅ Hoàn tất!")
            
            # --- HIỂN THỊ KẾT QUẢ ---
            st.divider()
            
            # 1. Thống kê tổng
            risk_count = len(df_upload[df_upload['Final_Result'] == 1])
            safe_count = len(df_upload) - risk_count
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Tổng nhân viên", len(df_upload))
            kpi2.metric("Nguy cơ cao", risk_count, delta_color="inverse")
            kpi3.metric("Bình thường", safe_count)
            
            # 2. Biểu đồ SO SÁNH CÁC MÔ HÌNH (Giải quyết yêu cầu của cô)
            st.subheader("📊 So Sánh Sự Nhạy Cảm Của Các Mô Hình")
            st.write("Biểu đồ này cho thấy mỗi mô hình phát hiện bao nhiêu người có nguy cơ.")
            
            # Tính tổng số ca bệnh mà mỗi model phát hiện
            model_counts = {}
            for name in loaded_models.keys():
                count = df_upload[f'Pred_{name}'].sum()
                model_counts[name] = count
                
            # Vẽ biểu đồ
            fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
            bars = ax_comp.bar(model_counts.keys(), model_counts.values(), color='#4e79a7')
            ax_comp.set_ylabel("Số người phát hiện nguy cơ")
            
            # Số liệu trên cột
            for bar in bars:
                height = bar.get_height()
                ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig_comp)
    

            # 3. Danh sách chi tiết
            st.subheader("📋 Danh sách những người cần lưu ý")
            
            # Chọn cột hiển thị
            cols_show = ['Total_Votes', 'Khuyen_Nghi']
            # Thêm tên tuổi nếu có
            info_cols = [c for c in ['Name', 'EmployeeID', 'Full Name', 'Age', 'Gender'] if c in df_upload.columns]
            cols_show = info_cols + cols_show
            
            risk_df = df_upload[df_upload['Final_Result'] == 1].sort_values(by='Total_Votes', ascending=False)
            st.dataframe(risk_df[cols_show])
            
            # 4. Download
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải Báo Cáo Chi Tiết (.csv)",
                data=csv,
                file_name='bao_cao_suckhoe_tonghop.csv',
                mime='text/csv'
            )