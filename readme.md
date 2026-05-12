# ☕ Cafe Recommendation System (Content-Based Filtering)

Sistem rekomendasi kafe berbasis web yang dibangun menggunakan **Python**, **Streamlit**, dan algoritma **Content-Based Filtering**. Aplikasi ini memungkinkan pengguna untuk membandingkan beberapa kafe sekaligus dan mendapatkan rekomendasi tempat serupa berdasarkan kemiripan ulasan (TF-IDF & Cosine Similarity).

## 🚀 Fitur Utama

* **Multi-Input Comparison**: Masukkan satu atau lebih kafe favorit untuk melihat perbandingan rating secara *side-by-side*.
* **Smart Recommendation**: Menghasilkan rekomendasi berdasarkan "User Profile" (rata-rata vektor fitur) jika pengguna memasukkan lebih dari satu kafe.
* **Cafe Statistics**: Ringkasan distribusi rating bintang (5.0, 4.0+, dan <4.0) untuk melihat kualitas keseluruhan database.
* **Interactive Table**: Jelajahi seluruh database kafe dengan fitur pencarian, pengurutan, dan filter yang interaktif.
* **Match Scoring**: Menampilkan persentase kecocokan (*similarity score*) untuk setiap rekomendasi.

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman**: Python 3.x
* **Framework Web**: [Streamlit](https://streamlit.io/)
* **Analisis Data**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
* **Model Persistence**: Pickle / Joblib

## 📦 Instalasi

1. **Clone Repository**
   ```bash
   git clone [https://github.com/Sintasitinuriah/Final_Project_STBI.git](https://github.com/Sintasitinuriah/Final_Project_STBI.git)
   cd Final_Project_STBI