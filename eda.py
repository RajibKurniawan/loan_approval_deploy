import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def run():
    # element title
    st.title("Loan Approval Classification Dataset")

    # element image
    st.image(
        "https://d307bj69336vgo.cloudfront.net/wp-content/w3-webp/uploads/2024/12/Loan-Approval-Classification-Dataset.pngw3.webp",
        caption="",
    )

    # header
    st.markdown("## Latar Belakang")
    # markdown
    st.markdown(
        """
                Peningkatan jumlah pengajuan pinjaman pada lembaga keuangan dan fintech saat ini menuntut adanya 
                sistem yang dapat membantu proses pengambilan keputusan kredit secara lebih cepat dan objektif. 
                Setiap pengajuan pinjaman harus melalui analisis kelayakan agar lembaga tidak mengalami risiko gagal bayar (default). 
                Oleh karena itu, diperlukan model machine learning yang dapat memprediksi apakah calon peminjam layak mendapatkan persetujuan pinjaman atau tidak, 
                berdasarkan atribut-atribut seperti pendapatan, status pernikahan, pekerjaan, jumlah pinjaman, dan sebagainya."""
    )

    st.header("Dataset")
    st.markdown("Loan Approval Classification Dataset")

    # load data dengan pandas
    data = pd.read_csv(
        "loan_data.csv")

    # rename column
    data.rename(
        columns={
            "person_age": "Age",
            "person_gender": "Gender",
            "person_education": "Education_Level",
            "person_income": "Income",
            "person_emp_exp": "Experience_Years",
            "person_home_ownership": "Home_Status",
            "loan_amnt": "Loan_Amount",
            "loan_intent": "Loan_Purpose",
            "loan_int_rate": "Loan_Interest_Rate",
            "loan_percent_income": "Loan_to_Income_Ratio",
            "cb_person_cred_hist_length": "Credit_History_Length",
            "credit_score": "Credit_Score",
            "previous_loan_defaults_on_file": "Default_On_Payment",
            "loan_status": "Loan_Status",
        },
        inplace=True,
    )
    data

    # tampilkan datafram
    st.dataframe(data)

    # EDA
    st.header("Exploratory Data Analyst")

    # 1. Distribusi
    st.subheader("1. Distribusi Target (Loan_Status)")

    # Visualisasi
    fig = plt.figure(figsize=(5, 5))
    labels = ["Ditolak (0)", "Disetujui (1)"]
    sizes = data["Loan_Status"].value_counts()
    colors = ["#ff9999", "#66b3ff"]
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=(0.05, 0),
    )
    plt.title("Proporsi Status Pinjaman")
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)

    # insight
    st.markdown(
        "Dari pie chart di atas terlihat rasio antara pinjaman yang disetujui dan ditolak. Proporsinya dapat membantu mengetahui apakah data seimbang (balanced) atau tidak seimbang (imbalanced). mungkin dikarenakan tidak lembaga pemberi pinjaman memiliki kebijakan konservatif dalam menyetujui pinjaman, yang umumnya dipengaruhi oleh faktor-faktor seperti pendapatan rendah, skor kredit rendah, dan rasio pinjaman terhadap pendapatan yang tinggi."
    )

    # 2. Rata-rata Pendapatan per Status Pinjaman
    st.subheader("2. Rata-rata Pendapatan per Status Pinjaman")

    # Visualisasi
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(data=data, x="Loan_Status", y="Income", palette="viridis", estimator="mean")
    plt.title("Rata-rata Pendapatan per Status Pinjaman")
    plt.xlabel("Loan Status (0 = Ditolak, 1 = Disetujui)")
    plt.ylabel("Rata-rata Pendapatan")
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Rata-rata pendapatan pemohon yang tidak disetujui lebih tinggi dibanding yang disetujui. Hal ini menegaskan bahwa income Pemohon berisiko tidak mampu mencicil adalah faktor penting dalam kelayakan pinjaman."
    )

    # 3. Usia Pemohon berdasarkan Status Pinjaman
    st.subheader("3. Usia Pemohon berdasarkan Status Pinjaman")

    # Visualisasi
    fig = plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=data[data["Loan_Status"] == 0]["Age"], label="Ditolak", shade=True, color="red")
    sns.kdeplot(
        data=data[data["Loan_Status"] == 1]["Age"],
        label="Disetujui",
        shade=True,
        color="blue",)
    plt.title("Distribusi Usia Pemohon berdasarkan Status Pinjaman")
    plt.xlabel("Usia Pemohon")
    plt.legend()
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Pemohon dengan usia antara 21–35 tahun tampak memiliki peluang lebih tinggi disetujui dibanding usia lebih muda. Distribusi ini membantu memahami profil umur tipikal peminjam yang berhasil."
    )

    # 4. Status Kepemilikan Rumah dan Status Pinjaman
    st.subheader("4. Status Kepemilikan Rumah dan Status Pinjaman")

    # Visualisasi
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="Home_Status", hue="Loan_Status", palette="rocket")
    plt.title("Status Kepemilikan Rumah dan Status Pinjaman")
    plt.xlabel("Status Kepemilikan Rumah")
    plt.ylabel("Jumlah Pemohon")
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Pemohon yang memiliki rumah sendiri (OWN atau MORTGAGE) lebih sering disetujui dibandingkan penyewa (RENT). Kepemilikan aset fisik memperkuat kepercayaan nantinya ke lembaga kredit. dan dari hasil visualisasi diatas masih banyak yang tidak disetujui pembiayaan dikarenakan beberapa faktor yang kurang mendukung untuk pengajuan kredit"
    )

    # 5.Hubungan Antara Rasio Pinjaman terhadap Pendapatan
    st.subheader("5. Hubungan Antara Rasio Pinjaman terhadap Pendapatan")

    # Visualisasi
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="Loan_to_Income_Ratio",
        y="Credit_Score",
        hue="Loan_Status",
        palette="Set1",
        alpha=0.6,
    )
    plt.title("Rasio Pinjaman terhadap Pendapatan vs Skor Kredit")
    plt.xlabel("Loan to Income Ratio")
    plt.ylabel("Credit Score")
    plt.legend(title="Loan Status", labels=["Ditolak (0)", "Disetujui (1)"])
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Pemohon dengan rasio pinjaman terhadap pendapatan yang rendah (<0.1) dan skor kredit tinggi lebih sering disetujui. Grafik ini memperlihatkan interaksi antara dua fitur numerik terhadap target. jadi semakin besar pinjaman bisa saja salah satu faktor penolakan"
    )

    # 6. Status Kepemilikan Rumah dan Status Pinjaman
    st.subheader("6. Lama Pekerjaan dengan Status Pinjaman")

    # Kategorisasi berdasarkan pengalaman
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
    data["Experience_Category"] = data["Experience_Years"].apply(categorize_experience)

    # countplot chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Pie chart untuk yang Disetujui
    approved = (
        data[data["Loan_Status"] == 1]["Experience_Category"].value_counts(normalize=True)
        * 100
    )
    axes[0].pie(
        approved,
        labels=approved.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Blues"),
        wedgeprops={"edgecolor": "white"},
    )
    axes[0].set_title("Disetujui (Loan_Status = 1)")

    # Pie chart untuk yang Ditolak
    rejected = (
        data[data["Loan_Status"] == 0]["Experience_Category"].value_counts(normalize=True)
        * 100
    )
    axes[1].pie(
        rejected,
        labels=rejected.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Reds"),
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Ditolak (Loan_Status = 0)")

    plt.suptitle(
        "Perbandingan Proporsi Lama Pengalaman Kerja Berdasarkan Status Pinjaman",
        fontsize=14,
    )
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Berdasarkan hasil analisis, lama pengalaman kerja tidak menunjukkan perbedaan proporsi yang signifikan antara pemohon yang disetujui dan ditolak. Hal ini mengindikasikan bahwa faktor lain seperti skor kredit, pendapatan, dan rasio pinjaman terhadap pendapatan memiliki pengaruh yang lebih besar terhadap keputusan persetujuan pinjaman."
    )

    # 7. Hubungan Riwayat GagaL Bayar dengan Status Pinjaman
    st.subheader("7. Hubungan Riwayat GagaL Bayar dengan Status Pinjaman")

    # Visualisasi
    fig = plt.figure(figsize=(6,4))
    sns.countplot(data=data, x="Default_On_Payment", hue="Loan_Status", palette="Set2")
    plt.title("Hubungan Riwayat Gagal Bayar dengan Status Pinjaman")
    plt.xlabel("Pernah Gagal Bayar Sebelumnya")
    plt.ylabel("Jumlah Pemohon")
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Pemohon yang memiliki riwayat gagal bayar sebelumnya (Yes) jauh lebih sering ditolak. karena ini logis karena catatan buruk di masa lalu mungkin ada kendala macet atau gagal bayar yang berbulan akan memengaruhi kepercayaan lembaga keuangan."
    )

    # 8. Hubungan Riwayat GagaL Bayar dengan Status Pinjaman
    st.subheader("8. Hubungan Riwayat GagaL Bayar dengan Status Pinjaman")

    # Visualisasi
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(data=data, x="Loan_Status", y="Credit_Score", palette="cool")
    plt.title("Hubungan Skor Kredit dengan Status Pinjaman")
    plt.xlabel("Loan Status (0 = Ditolak, 1 = Disetujui)")
    plt.ylabel("Skor Kredit")
    plt.show()
    # menampilkan matplotlib chart
    st.pyplot(fig)
    # insight
    st.markdown(
        "Pemohon dengan skor kredit tinggi hampir selalu memiliki peluang disetujui lebih besar. Ini adalah salah satu variabel paling penting dalam analisis risiko kredit. dan mungkin ada yang kredit skor tinggi tetapi ada faktor yang lain kurang mendukung jadi disini antara status Ditoal dan Disetujui tidak jauh beda."
    )

    # run
if __name__ == "__main__":
    run()
