import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load('best_depression_model.joblib')
scaler = joblib.load('scaler.joblib')
top_features = joblib.load('top_features.joblib')

st.title("Student Depression Prediction App")
st.write("Aplikasi ini memprediksi status depresi mahasiswa berdasarkan faktor akademik dan gaya hidup.")

# Form for user input
with st.form("prediction_form"):
    st.subheader("Input Data Mahasiswa")
    
    # Dynamic inputs based on top features used in the model
    inputs = {}
    inputs['Have you ever had suicidal thoughts ?'] = st.selectbox("Pernah terpikir untuk bunuh diri?", ["No", "Yes"])
    inputs['Academic Pressure'] = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
    inputs['Financial Stress'] = st.slider("Tekanan Finansial (1-5)", 1, 5, 3)
    inputs['CGPA'] = st.number_input("CGPA", 0.0, 10.0, 7.5)
    inputs['Age'] = st.number_input("Usia", 18, 60, 21)
    inputs['City'] = st.text_input("Kota (Contoh: Srinagar, Jaipur)", "Srinagar")
    inputs['Work/Study Hours'] = st.number_input("Jam Belajar/Kerja per Hari", 1, 12, 6)
    inputs['Degree'] = st.text_input("Gelar (Contoh: B.Tech, BSc)", "B.Tech")
    inputs['Study Satisfaction'] = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
    inputs['Dietary Habits'] = st.selectbox("Pola Makan", ["Healthy", "Moderate", "Unhealthy"])
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # 1. Buat DataFrame awal dari input
    df_input = pd.DataFrame([inputs])

    # 2. Manual Mapping (Harus SAMA dengan LabelEncoder di Notebook kamu)
    # Sesuaikan angka di bawah ini dengan urutan alfabet/label di dataset asli
    mapping_suicidal = {"No": 0, "Yes": 1}
    mapping_diet = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
    
    df_input['Have you ever had suicidal thoughts ?'] = df_input['Have you ever had suicidal thoughts ?'].map(mapping_suicidal)
    df_input['Dietary Habits'] = df_input['Dietary Habits'].map(mapping_diet)

    # Untuk City dan Degree, karena opsinya banyak, kita gunakan cara aman:
    # Mengubah string menjadi hash angka sederhana atau nilai default jika tidak ada encoder-nya
    df_input['City'] = df_input['City'].apply(lambda x: len(x)) # Ini hanya placeholder
    df_input['Degree'] = df_input['Degree'].apply(lambda x: len(x)) # Ini hanya placeholder
    
    # 3. Menyamakan Struktur dengan 16 Fitur Asli
    # Kita buat dataframe kosong dengan 16 kolom (semua nol), lalu timpa dengan input user
    full_features = [
        'Gender', 'Age', 'City', 'Working Professional or Student', 'Academic Pressure', 
        'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 
        'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness'
    ]
    
    df_final = pd.DataFrame(0, index=[0], columns=full_features)
    
    # Update nilai yang ada dari input user
    for col in inputs.keys():
        if col in df_final.columns:
            df_final[col] = df_input[col]

    # 4. Scaling
    # Scaler WAJIB menerima 16 fitur sesuai saat fit() dulu
    data_scaled = scaler.transform(df_final)
    
    # Buat dataframe hasil scaling agar bisa difilter berdasarkan top_features
    df_scaled = pd.DataFrame(data_scaled, columns=full_features)

    # 5. Prediksi menggunakan top_features yang sudah di-scaling
    prediction = model.predict(df_scaled[top_features])
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Hasil Prediksi: Menunjukkan Gejala Depresi")
        st.write("Tetap semangat! Jangan ragu untuk bercerita kepada orang terdekat atau profesional.")
    else:
        st.success("### Hasil Prediksi: Tidak Menunjukkan Gejala Depresi")
        st.write("Kesehatan mentalmu terlihat baik. Pertahankan pola hidup sehat ya!")
