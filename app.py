import streamlit as st
import pickle
import re
import string
import numpy as np

# ======================================
# LOAD MODEL & TF-IDF
# ======================================
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model_svm.pkl", "rb"))

# ======================================
# TEXT PREPROCESSING
# ======================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ======================================
# CONFIDENCE SCORE (SVM)
# ======================================
def svm_confidence(model, X):
    decision = model.decision_function(X)
    confidence = 1 / (1 + np.exp(-decision))  # sigmoid normalization
    return confidence[0]

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Spam SMS PayLater Detector",
    page_icon="📩",
    layout="centered"
)

# ======================================
# HEADER
# ======================================
st.markdown("""
<h1 style='text-align:center;'>📩 Spam SMS PayLater Detector</h1>
<p style='text-align:center; color:gray;'>
Deteksi SMS spam PayLater menggunakan <b>TF-IDF</b> dan <b>Support Vector Machine</b>
</p>
<hr>
""", unsafe_allow_html=True)

# ======================================
# ABOUT
# ======================================
with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    Aplikasi ini bertujuan untuk mendeteksi SMS spam yang memanfaatkan
    **ancaman tagihan PayLater palsu**, seperti denda, jatuh tempo,
    dan pemblokiran akun.
    """)

# ======================================
# SAMPLE SMS
# ======================================
st.subheader("💡 Contoh SMS (Klik untuk uji cepat)")

contoh_sms = [
    "Tagihan PayLater Anda belum dibayar. Denda akan dikenakan hari ini.",
    "Akun PayLater Anda diblokir sementara. Segera lakukan verifikasi.",
    "Kesempatan terakhir! Bayar tagihan PayLater sekarang.",
    "Promo PayLater 0% bunga khusus hari ini.",
    "Pembayaran PayLater Anda berhasil. Terima kasih.",
    "Ingatkan teman Anda untuk menggunakan PayLater dan dapatkan bonus!",
    "Tagihan PayLater Anda sudah lunas. Nikmati kemudahan berbelanja lagi.",
    "Verifikasi identitas Anda untuk keamanan akun PayLater.",
    "Dapatkan cashback hingga 50% dengan menggunakan PayLater di merchant pilihan.",
    "Perbarui informasi pembayaran PayLater Anda untuk menghindari gangguan layanan."
]

for i, sms in enumerate(contoh_sms, 1):
    if st.button(f"Contoh {i}"):
        st.session_state["sms_input"] = sms

# ======================================
# INPUT TEXT
# ======================================
st.subheader("✍️ Masukkan Teks SMS")

sms_input = st.text_area(
    "",
    value=st.session_state.get("sms_input", ""),
    height=150,
    placeholder="Contoh: Tagihan PayLater Anda belum dibayar."
)

# ======================================
# DETECT BUTTON
# ======================================
detect_btn = st.button("🔍 Deteksi Sekarang", use_container_width=True)

# ======================================
# RESULT
# ======================================
if detect_btn:
    if sms_input.strip() == "":
        st.warning("⚠️ Teks SMS tidak boleh kosong.")
    else:
        cleaned_text = clean_text(sms_input)
        vectorized_text = tfidf.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        confidence = svm_confidence(model, vectorized_text)

        st.markdown("---")
        st.subheader("📊 Hasil Deteksi")

        if prediction == "spam":
            st.error("🚨 **SPAM**")
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            st.write("""
            Pesan ini memiliki pola **ancaman, urgensi, dan penipuan PayLater palsu**.
            Disarankan **tidak mengklik tautan atau membalas pesan**.
            """)
        else:
            st.success("✅ **HAM (Bukan Spam)**")
            st.metric("Confidence Score", f"{(1-confidence)*100:.2f}%")
            st.write("""
            Pesan ini tidak terindikasi sebagai spam PayLater.
            Tetap waspada dan pastikan sumber pesan resmi.
            """)

# ======================================
# FOOTER
# ======================================
st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:13px;'>
kelompok 1 UAS NLP — Spam SMS PayLater Detection
</p>
""", unsafe_allow_html=True)
