import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load model terbaik
model_path = "best_model_xgboost.pkl"
model = pickle.load(open(model_path, "rb"))

print("✅ Model berhasil dimuat kembali!")


def run():
    st.title("Loan Approval Classification Dataset")
    st.markdown("Masukkan data calon nasabah untuk memprediksi kelayakan pinjaman.")

    with st.form(key="Calon_Nasabah"):
        st.header("Input Data Calon Nasabah")

        name = st.text_input("Masukkan Nama", placeholder="cth: Rajib Kurniawan")
        age = st.number_input("Usia", min_value=18, max_value=100, value=28)
        income = st.number_input("Pendapatan Bulanan", min_value=0, value=52000)
        credit_score = st.number_input("Credit Score (BI Checking)", min_value=300, max_value=900, value=710)
        credit_history = st.number_input("Lama Riwayat Kredit (Tahun)", min_value=0, value=4)
        loan_interest = st.number_input("Bunga Pinjaman (%)", min_value=0.0, value=11.2)
        loan_ratio = st.number_input("Loan to Income Ratio", min_value=0.0, max_value=1.0, value=0.15)
        loan_amount = st.number_input("Jumlah Pinjaman (USD)", min_value=0, value=8000)

        st.write("___")

        gender = st.selectbox("Gender", ["male", "female"])
        education = st.selectbox("Tingkat Pendidikan", ["Bachelor", "Master", "PhD"])
        home_status = st.selectbox("Status Kepemilikan Rumah", ["RENT", "MORTGAGE", "OWN"])
        loan_purpose = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        experience_years = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=40, value=5)
        default_payment = st.selectbox("Riwayat Gagal Bayar Sebelumnya", ["No", "Yes"])

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        # Buat DataFrame dari input

        data_inf = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Education_Level": education,
            "Income": income,
            "Experience_Years": experience_years,
            "Home_Status": home_status,
            "Loan_Amount": loan_amount,
            "Loan_Purpose": loan_purpose,
            "Loan_Interest_Rate": loan_interest,
            "Loan_to_Income_Ratio": loan_ratio,
            "Credit_History_Length": credit_history,
            "Credit_Score": credit_score,
            "Default_On_Payment": default_payment
        }])

        # Tambahkan fitur turunan
        def categorize_experience(years):
            if years <= 2:
                return "Entry Level"
            elif years <= 5:
                return "Junior"
            elif years <= 10:
                return "Senior"
            elif years <= 20:
                return "Expert"
            else:
                return "Veteran"

        data_inf["Experience_Category"] = data_inf["Experience_Years"].apply(categorize_experience)

        # Prediksi
        pred = model.predict(data_inf)
        proba = model.predict_proba(data_inf)[:, 1]

        data_inf["Loan_Status_Prediction"] = pred
        data_inf["Loan_Approval_Probability"] = proba

        # Output

        st.subheader("📊 Hasil Prediksi")
        st.dataframe(data_inf)

        if pred[0] == 1:
            st.success(f"✅ Pengajuan pinjaman **DISETUJUI** dengan probabilitas {proba[0]:.2%}")
        else:
            st.error(f"❌ Pengajuan pinjaman **DITOLAK** dengan probabilitas {proba[0]:.2%}")

if __name__ == "__main__":
    run()
