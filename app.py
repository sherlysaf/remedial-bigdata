import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="Remedial Big Data", layout="wide")

# =========================================================
# 2. SYARAT VALIDASI WAJIB: WATERMARK IDENTITAS MAHASISWA
# =========================================================
st.sidebar.title("📌 Identitas Mahasiswa")
st.sidebar.markdown("""
> **Remedial Big Data**  
> **Nama:** [Sherly safitri]  
> **NIM:** [20220040219]  
""")
st.sidebar.markdown("---")

# Judul Utama Dashboard
st.title("📊 Customer Churn Dashboard & Prediction System")
st.write("Sistem Analisis Big Data dan Prediksi Churn berbasis Machine Learning.")

# 3. Load Data dari CSV hasil olahan PySpark Colab
@st.cache_data
def load_data():
    try:
        # Membaca data_cleaned.csv dari folder yang sama
        return pd.read_csv("data_cleaned.csv")
    except Exception as e:
        # Fallback jika data_cleaned.csv tidak ditemukan di folder
        st.warning("Berkas 'data_cleaned.csv' tidak ditemukan. Menggunakan sampel data dummy.")
        return pd.DataFrame({
            "gender": ["Female", "Male", "Female", "Male"] * 50,
            "age": [30, 45, 22, 38] * 50,
            "tenure": [5, 12, 2, 8] * 50,
            "monthly_charges": [100.0, 500.0, 50.0, 300.0] * 50,
            "churn": [0, 1, 0, 1] * 50
        })

df = load_data()

# 4. Membuat Menu Tab (Dashboard vs Form Prediksi)
tab1, tab2 = st.tabs(["📈 Dashboard Visualisasi", "🤖 Form Prediksi Real-Time"])

# ---------------------------------------------------------
# TAB 1: DASHBOARD VISUALISASI DATA
# ---------------------------------------------------------
with tab1:
    st.header("Visualisasi Data Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            df, 
            x="age", 
            color="churn", 
            title="Distribusi Umur Berdasarkan Status Churn",
            barmode="overlay"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.box(
            df, 
            x="churn", 
            y="monthly_charges", 
            title="Perbandingan Biaya Bulanan vs Status Churn",
            color="churn"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: FORM PREDIKSI REAL-TIME
# ---------------------------------------------------------
with tab2:
    st.header("Form Prediksi Churn Real-Time")
    st.write("Isi formulir di bawah ini untuk menguji prediksi model secara interaktif:")
    
    # Latih model ML di backend Streamlit berdasarkan data CSV
    X = df[["age", "tenure", "monthly_charges"]]
    y = df["churn"]
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    
    # Form Input Interaktif
    col_a, col_b = st.columns(2)
    with col_a:
        age_input = st.number_input("Umur Pelanggan (Tahun)", min_value=18, max_value=100, value=30)
        tenure_input = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12)
    with col_b:
        charges_input = st.number_input("Biaya Bulanan ($)", min_value=0.0, max_value=2000.0, value=150.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Eksekusi Prediksi
    if st.button("🚀 Jalankan Prediksi", type="primary"):
        prediction = clf.predict([[age_input, tenure_input, charges_input]])
        
        if prediction[0] == 1:
            st.error("⚠️ **Hasil Prediksi:** Pelanggan berpotensi **CHURN (Berhenti Berlangganan)**!")
        else:
            st.success("✅ **Hasil Prediksi:** Pelanggan diprediksi **SETIA (Tetap Aktif)**.")
