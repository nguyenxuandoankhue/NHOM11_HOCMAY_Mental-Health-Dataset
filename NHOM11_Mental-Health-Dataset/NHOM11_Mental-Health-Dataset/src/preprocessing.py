import pandas as pd
import numpy as np

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(df):
    df = df.copy()

    # ===== 0. LOẠI BỎ CỘT KHÔNG CẦN THIẾT =====
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
        
    if 'SocialWeakness.1' in df.columns:
        df = df.drop(columns=['SocialWeakness.1'])

    # ===== 1. XỬ LÝ MISSING VALUES =====
    if 'SelfEmployed' in df.columns:
        mode_val = df['SelfEmployed'].mode()[0] if not df['SelfEmployed'].empty else 'No'
        df['SelfEmployed'] = df['SelfEmployed'].fillna(mode_val)

    # ===== 2. MÃ HÓA DỮ LIỆU (MAPPING) =====
    # 2.1 Binary mapping 
    binary_cols = ['SelfEmployed', 'FamilyHistory', 'CopingStruggles']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

    # Xử lý riêng cột Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # 2.2 Ordinal encoding (Thứ tự)
    days_map = {'Go out Every day': 0, '1-14 days': 1, '15-30 days': 2, '31-60 days': 3, 'More than 2 months': 4}
    level_map = {'Low': 0, 'Medium': 1, 'High': 2}
    care_map = {'No': 0, 'Not sure': 1, 'Yes': 2}

    if 'DaysIndoors' in df.columns:
        df['DaysIndoors'] = df['DaysIndoors'].map(days_map)
    if 'MoodSwings' in df.columns:
        df['MoodSwings'] = df['MoodSwings'].map(level_map)
    if 'CareOptions' in df.columns:
        df['CareOptions'] = df['CareOptions'].map(care_map)

    # ===== 3. XỬ LÝ BIẾN MỤC TIÊU =====
    if 'Treatment' in df.columns:
        df['Treatment'] = df['Treatment'].map({'No': 0, 'Yes': 1})

    # ===== 4. ONE-HOT ENCODING =====
    
    multi_cols = ['MentalHealthHistory', 'HabitsChange', 'IncreasingStress', 
                  'SocialWeakness', 'WorkInterest', 'MentalHealthInterview']
    nominal_cols = ['Country', 'Occupation']
    
    # Chỉ lấy những cột thực sự có trong df hiện tại
    cols_to_dummy = [col for col in multi_cols + nominal_cols if col in df.columns]
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)

    return df