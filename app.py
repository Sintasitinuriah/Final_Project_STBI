import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DEFINISI CLASS (SUDAH DIOPTIMALKAN UNTUK VIBES) ---
class ContentBasedFilteringKafe:
    def __init__(self, tfidf_matrix, sim_matrix, dokumen_kafe, tfidf_vectorizer=None):
        self.tfidf_matrix = tfidf_matrix
        self.sim_matrix   = sim_matrix
        self.df_kafe      = dokumen_kafe.copy().reset_index(drop=True)
        self.vectorizer   = tfidf_vectorizer
        self.nama_ke_idx  = pd.Series(self.df_kafe.index, index=self.df_kafe['nama_kafe'])
        
        # Inisialisasi parameter IMDb
        self.C = self.df_kafe['rata_rating'].mean()
        self.m = 5
        self._hitung_skor_kualitas_global()

    def _hitung_skor_kualitas_global(self):
        """Menghitung skor kualitas berbobot (IMDb) untuk setiap kafe"""
        v = self.df_kafe['jumlah_ulasan']
        R = self.df_kafe['rata_rating']
        
        # Rumus IMDb
        skor_kualitas = (v / (v + self.m) * R) + (self.m / (v + self.m) * self.C)
        self.df_kafe['skor_kualitas_norm'] = skor_kualitas / 5.0

    def _get_top_keywords(self, row_idx, top_k=4):
        """
        Mengambil kata kunci dominan/vibes dengan SMART FALLBACK MECHANISM.
        Jika vectorizer kosong, data diekstrak dari kolom tag hasil olahan dataset.
        """
        # LAYER 1: Ekstraksi Berbasis Pembobotan TF-IDF (Jika Vectorizer Tersedia)
        if self.vectorizer is not None:
            try:
                row_vector = self.tfidf_matrix[row_idx].toarray().flatten()
                # Hanya pilih kata yang memiliki bobot TF-IDF > 0
                non_zero_indices = np.where(row_vector > 0)[0]
                if len(non_zero_indices) > 0:
                    top_indices = non_zero_indices[np.argsort(row_vector[non_zero_indices])[::-1][:top_k]]
                    feature_names = self.vectorizer.get_feature_names_out()
                    top_words = [feature_names[i].title() for i in top_indices]
                    if top_words:
                        return ", ".join(top_words)
            except Exception:
                pass # Jika ada error kalkulasi, otomatis lari ke Layer 2
                
        # LAYER 2: Fallback Ekstraksi Langsung dari Kolom Deskripsi/Tag Vibe Karakteristik
        row_data = self.df_kafe.iloc[row_idx]
        extracted_tags = []
        
        kolom_vibe_sumber = ['tag_kebisingan', 'tag_makanan', 'tag_suasana', 'tag_layanan']
        for col in kolom_vibe_sumber:
            if col in self.df_kafe.columns and pd.notna(row_data[col]):
                val_str = str(row_data[col]).strip()
                if val_str:
                    # Pecah kalimat berdasarkan spasi/koma, ambil kata unik berkarakter > 3 huruf
                    words = [w.strip().title() for w in val_str.replace(',', ' ').split() if len(w.strip()) > 3]
                    extracted_tags.extend(words)
                    
        # Hilangkan duplikasi kata tanpa merusak urutan asli
        unique_tags = list(dict.fromkeys(extracted_tags))[:top_k]
        if unique_tags:
            return ", ".join(unique_tags)
            
        # LAYER 3: Fallback Terakhir dari Dokumen Gabungan
        if 'dokumen_gabungan' in self.df_kafe.columns:
            words = [w.title() for w in str(row_data['dokumen_gabungan']).split() if len(w) > 4]
            unique_words = list(dict.fromkeys(words))[:top_k]
            if unique_words:
                return ", ".join(unique_words)
                
        return "Cozy, Cafe Jogja" # Default jika kosong sama sekali

    def rekomendasikan(self, list_kafe, top_n=5, alpha=0.7):
        """Fungsi rekomendasi dengan Hybrid Reranking (Teks + Rating)"""
        idx_pilihan = [self.nama_ke_idx[n] for n in list_kafe if n in self.nama_ke_idx]
        if not idx_pilihan: return pd.DataFrame()

        # 1. Hitung profil teks pengguna & kemiripannya
        vektor_profil = np.average(self.tfidf_matrix[idx_pilihan].toarray(), axis=0)
        skor_teks = cosine_similarity(vektor_profil.reshape(1, -1), self.tfidf_matrix).flatten()
        
        # 2. Hitung Skor Akhir (Hybrid: alpha% Teks + beta% Rating Kualitas)
        beta = 1.0 - alpha
        skor_kualitas = self.df_kafe['skor_kualitas_norm'].values
        skor_akhir = (alpha * skor_teks) + (beta * skor_kualitas)
        
        # 3. Blokir kafe yang sudah dipilih agar tidak direkomendasikan ulang
        skor_akhir[idx_pilihan] = -1.0
        
        # 4. Ambil Top N
        top_idx = np.argsort(skor_akhir)[::-1][:top_n]
        
        # 5. Ekstrak kata kunci untuk kafe-kafe yang direkomendasikan
        keywords_list = [self._get_top_keywords(i, top_k=4) for i in top_idx]
        
        return pd.DataFrame({
            'Nama Kafe'        : self.df_kafe.loc[top_idx, 'nama_kafe'].values,
            'Text Sim Score'   : skor_teks[top_idx].round(4),
            'Final Score'      : skor_akhir[top_idx].round(4),
            'Rating'           : self.df_kafe.loc[top_idx, 'rata_rating'].values,
            'Ulasan'           : self.df_kafe.loc[top_idx, 'jumlah_ulasan'].values,
            'Keywords'         : keywords_list
        })


# --- 2. LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        # Menangani fleksibilitas nama file pkl proyekmu
        try:
            with open('model_kafe.pkl', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            with open('model_kafe_jogja.pkl', 'rb') as f:
                data = pickle.load(f)
            
        # Ambil dataframe secara fleksibel berdasarkan key yang tersimpan di .pkl
        df_kafe_key = data.get('df_kafe', data.get('dokumen_kafe', None))
        vectorizer = data.get('vectorizer', None)
        
        if df_kafe_key is None:
            st.error("Format dataframe di dalam file pickle tidak dikenali.")
            return None
        
        return ContentBasedFilteringKafe(
            tfidf_matrix=data['tfidf_matrix'], 
            sim_matrix=data['sim_matrix'], 
            dokumen_kafe=df_kafe_key,
            tfidf_vectorizer=vectorizer
        )
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- 3. UI UTAMA ---
def main():
    st.set_page_config(page_title="Cafe Finder Hub", layout="wide")
    st.title("☕ Smart Cafe Recommendation System")
    st.markdown("Sistem rekomendasi berbasis **Content-Based Filtering** dengan analisis teks ulasan (NLP) dan pembobotan kualitas kafe.")
    st.markdown("---")

    model = load_data()
    if not model:
        return

    # --- SIDEBAR ---
    st.sidebar.header("🔍 Cari & Bandingkan")
    jml_input = st.sidebar.number_input("Jumlah kafe acuan:", min_value=1, max_value=5, value=1)
    
    list_kafe_db = model.df_kafe['nama_kafe'].tolist()
    kafe_dipilih = []

    for i in range(int(jml_input)):
        pilihan = st.sidebar.selectbox(f"Kafe Acuan {i+1}:", list_kafe_db, key=f"kafe_{i}")
        kafe_dipilih.append(pilihan)

    # Pengaturan Alpha & Beta (Teks vs Rating)
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Pengaturan Bobot Algoritma")
    alpha_val = st.sidebar.slider("Fokus Kemiripan Ulasan (%)", min_value=0, max_value=100, value=70, step=10)
    alpha = alpha_val / 100.0
    st.sidebar.caption(f"*Sisa {100 - alpha_val}% akan fokus pada rating & kualitas tempat.*")
    
    top_n = st.sidebar.slider("Jumlah Rekomendasi:", 1, 12, 6)
    btn_rekomendasi = st.sidebar.button("Dapatkan Rekomendasi ✨")

    # --- BAGIAN HASIL REKOMENDASI ---
    if btn_rekomendasi:
        kafe_unik = list(set(kafe_dipilih))
        
        # Ekstrak kata kunci dari kafe acuan untuk ditampilkan ke user
        kata_kunci_acuan = []
        for kafe in kafe_unik:
            idx = model.nama_ke_idx[kafe]
            kata_kunci_acuan.append(model._get_top_keywords(idx, top_k=3))
            
        st.info(f"**Menganalisis karakteristik dari:** {', '.join(kafe_unik)}\n\n**Topik/Vibe Dominan Acuan:** *{', '.join(set(kata_kunci_acuan))}*")
        
        hasil = model.rekomendasikan(kafe_unik, top_n=top_n, alpha=alpha)
        
        if not hasil.empty:
            cols_rek = st.columns(3)
            for idx, row in hasil.iterrows():
                with cols_rek[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"#### {row['Nama Kafe']}")
                        st.write(f"⭐ **{row['Rating']}** ({row['Ulasan']} ulasan)")
                        
                        # Menampilkan Vibes dengan gaya italic tebal yang rapi
                        st.markdown(f"🏷️ **Vibe:** *{row['Keywords']}*")
                        
                        # Keamanan Progress Bar (Clipped antara 0.0 - 1.0 agar UI Streamlit tidak crash)
                        progress_score = float(np.clip(row['Final Score'], 0.0, 1.0))
                        st.progress(progress_score)
                        st.caption(f"Match Score: {int(progress_score*100)}% | Teks Sim: {int(row['Text Sim Score']*100)}%")
        else:
            st.warning("Tidak ada rekomendasi yang ditemukan.")
            
        st.divider()

    # --- BAGIAN JELAJAH KAFE (TABEL & RINGKASAN) ---
    st.subheader("🏙️ Jelajahi Semua Kafe")
    
    total_kafe = len(model.df_kafe)
    bintang_5 = len(model.df_kafe[model.df_kafe['rata_rating'] >= 4.5])
    bintang_4_keatas = len(model.df_kafe[(model.df_kafe['rata_rating'] >= 4.0) & (model.df_kafe['rata_rating'] < 4.5)])
    kurang_dari_4 = len(model.df_kafe[model.df_kafe['rata_rating'] < 4.0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Kafe", total_kafe)
    m2.metric("⭐⭐⭐⭐⭐ (≥ 4.5)", bintang_5)
    m3.metric("⭐⭐⭐⭐ (4.0 - 4.4)", bintang_4_keatas)
    m4.metric("⭐ (< 4.0)", kurang_dari_4)

    st.write("Gunakan tabel di bawah untuk melihat detail atau mencari nama kafe:")

    df_display = model.df_kafe[['nama_kafe', 'rata_rating', 'jumlah_ulasan']].copy()
    df_display.columns = ['Nama Kafe', 'Rating', 'Total Ulasan']
    
    st.dataframe(
        df_display.sort_values(by='Total Ulasan', ascending=False), 
        use_container_width=True, 
        hide_index=True
    )

if __name__ == "__main__":
    main()