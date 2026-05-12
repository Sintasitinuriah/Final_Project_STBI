import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DEFINISI CLASS ---
class ContentBasedFilteringKafe:
    def __init__(self, tfidf_matrix, sim_matrix, dokumen_kafe):
        self.tfidf_matrix = tfidf_matrix
        self.sim_matrix   = sim_matrix
        self.df_kafe      = dokumen_kafe.copy().reset_index(drop=True)
        self.nama_ke_idx  = pd.Series(self.df_kafe.index, index=self.df_kafe['nama_kafe'])

    def rekomendasikan(self, list_kafe, top_n=5):
        idx_pilihan = [self.nama_ke_idx[n] for n in list_kafe if n in self.nama_ke_idx]
        if not idx_pilihan: return pd.DataFrame()

        vektor_profil = np.average(self.tfidf_matrix[idx_pilihan].toarray(), axis=0)
        skor = cosine_similarity(vektor_profil.reshape(1, -1), self.tfidf_matrix).flatten()
        
        skor[idx_pilihan] = -1.0
        top_idx = np.argsort(skor)[::-1][:top_n]
        
        return pd.DataFrame({
            'Nama Kafe'        : self.df_kafe.loc[top_idx, 'nama_kafe'].values,
            'Similarity Score' : skor[top_idx].round(4),
            'Rating'           : self.df_kafe.loc[top_idx, 'rata_rating'].values,
            'Ulasan'           : self.df_kafe.loc[top_idx, 'jumlah_ulasan'].values,
        })

# --- 2. LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        with open('model_kafe.pkl', 'rb') as f:
            data = pickle.load(f)
        return ContentBasedFilteringKafe(data['tfidf_matrix'], data['sim_matrix'], data['df_kafe'])
    except:
        return None

# --- 3. UI UTAMA ---
def main():
    st.set_page_config(page_title="Cafe Finder Hub", layout="wide")
    st.title("☕ Cafe Recommendation System")
    st.markdown("---")

    model = load_data()
    if not model:
        st.error("Model tidak ditemukan.")
        return

    # --- SIDEBAR ---
    st.sidebar.header("🔍 Cari & Bandingkan")
    jml_input = st.sidebar.number_input("Jumlah kafe acuan:", min_value=1, max_value=10, value=1)
    
    list_kafe_db = model.df_kafe['nama_kafe'].tolist()
    kafe_dipilih = []

    for i in range(int(jml_input)):
        pilihan = st.sidebar.selectbox(f"Kafe {i+1}:", list_kafe_db, key=f"kafe_{i}")
        kafe_dipilih.append(pilihan)

    top_n = st.sidebar.slider("Jumlah Rekomendasi:", 1, 12, 6)
    btn_rekomendasi = st.sidebar.button("Dapatkan Rekomendasi ✨")

    # --- BAGIAN HASIL REKOMENDASI ---
    if btn_rekomendasi:
        st.success(f"Daftar Rekomendasi Berdasarkan {len(set(kafe_dipilih))} Pilihanmu:")
        hasil = model.rekomendasikan(list(set(kafe_dipilih)), top_n=top_n)
        
        if not hasil.empty:
            cols_rek = st.columns(3)
            for idx, row in hasil.iterrows():
                with cols_rek[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"#### {row['Nama Kafe']}")
                        st.write(f"⭐ **Rating:** {row['Rating']}")
                        st.caption(f"Match: {int(row['Similarity Score']*100)}% | {row['Ulasan']} ulasan")
        st.divider()

    # --- BAGIAN JELAJAH KAFE (TABEL & RINGKASAN) ---
    st.subheader("🏙️ Jelajahi Semua Kafe")
    
    # 1. Ringkasan Bintang (Metrics)
    total_kafe = len(model.df_kafe)
    bintang_5 = len(model.df_kafe[model.df_kafe['rata_rating'] >= 5.0])
    bintang_4_keatas = len(model.df_kafe[(model.df_kafe['rata_rating'] >= 4.0) & (model.df_kafe['rata_rating'] < 5.0)])
    kurang_dari_4 = len(model.df_kafe[model.df_kafe['rata_rating'] < 4.0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Kafe", total_kafe)
    m2.metric("⭐⭐⭐⭐⭐ (5.0)", bintang_5)
    m3.metric("⭐⭐⭐⭐ (4.0 - 4.9)", bintang_4_keatas)
    m4.metric("⭐ (< 4.0)", kurang_dari_4)

    st.write("Gunakan tabel di bawah untuk melihat detail atau mencari nama kafe:")

    # 2. Tampilan Tabel
    # Kita rapikan kolom agar enak dibaca
    df_display = model.df_kafe[['nama_kafe', 'rata_rating', 'jumlah_ulasan']].copy()
    df_display.columns = ['Nama Kafe', 'Rating', 'Total Ulasan']
    
    # Menampilkan tabel dengan fitur pencarian dan sorting bawaan Streamlit
    st.dataframe(
        df_display.sort_values(by='Rating', ascending=False), 
        use_container_width=True, 
        hide_index=True
    )

if __name__ == "__main__":
    main()