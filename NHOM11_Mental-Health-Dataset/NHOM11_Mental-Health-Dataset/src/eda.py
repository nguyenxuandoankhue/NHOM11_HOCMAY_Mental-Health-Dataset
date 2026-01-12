import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# HÀM HỖ TRỢ HIỂN THỊ TRÊN TERMINAL
def run_eda(df):
    print("===== DATA INFO")
    df.info()
    print("\n===== SAMPLE DATA")
    print(df.head(10))
    print("\n===== DESCRIPTIVE STATISTICS")
    print(df.describe())

#Kiểm tra missing values
def check_missing(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())
#Kiểm tra trùng cột dữ liệu 
def check_duplicate_columns(df):
    col1 = 'SocialWeakness'
    col2 = 'SocialWeakness.1'
    
    if col1 in df.columns and col2 in df.columns:
        print(f"\nCheck duplicate between {col1} and {col2}:")
        print((df[col1] == df[col2]).value_counts())
    else:
        print(f"\n[Info] Không tìm thấy cột {col2} (có thể đã được xử lý xóa bỏ).")

def plot_and_save_gender_vs_treatment(df):
    if not os.path.exists('results'):
        os.makedirs('results')

    if 'Treatment' in df.columns and 'Gender' in df.columns:
        plt.figure(figsize=(8, 5))
        # Vẽ biểu đồ
        sns.countplot(x="Gender", hue="Treatment", data=df, palette="Set2")
        plt.title("Gender Distribution by Treatment History")
        plt.ylabel("Count")
        
        # LƯU ẢNH 
        save_path = 'results/EDA_Gender_Analysis.png'
        plt.savefig(save_path)
        plt.close() 
        
        print(f"✅ [EDA] Đã lưu biểu đồ phân tích tại: {save_path}")
    else:
        print("⚠️ [EDA] Dữ liệu thiếu cột Treatment hoặc Gender, không vẽ được.")