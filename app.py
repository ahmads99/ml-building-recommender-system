import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/processed_movie_data.csv')
    return df

# Fungsi untuk menghitung skor popularitas
def weighted_rating(x, m, C):
    v = x['numVotes']
    R = x['averageRating']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Fungsi untuk mendapatkan rekomendasi berdasarkan konten
def get_recommendations(title, cosine_sim, df, top_n=5):
    idx = df[df['primaryTitle'] == title].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['primaryTitle', 'startYear', 'genres', 'averageRating', 'numVotes']]

# Fungsi untuk memuat CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Fungsi utama aplikasi
def main():
    # Muat CSS
    load_css('style.css')

    # Muat data
    df = load_data()


    # Hitung skor popularitas
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(0.90)
    df['weighted_score'] = df.apply(weighted_rating, args=(m, C), axis=1)

    # Preprocessing untuk TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['genres'].fillna('').str.replace(',', ' '))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


    # Judul aplikasi
    st.title("🎬 Sistem Rekomendasi Film")
    st.markdown("Temukan film favoritmu dan dapatkan rekomendasi yang sesuai dengan selera kamu! 🍿")

    # Filter berdasarkan genre dan tahun
    st.sidebar.header("🔍 Filter")
    genres = sorted(set(genre for sublist in df['genres'].dropna().str.split(',') for genre in sublist))
    selected_genre = st.sidebar.selectbox("Pilih Genre", ["Semua"] + genres)
    years = sorted(df['startYear'].dropna().unique(), reverse=True)  # reverse=True untuk urutan terbalik
    years = [int(year) for year in years]  # Ubah menjadi integer
    selected_year = st.sidebar.selectbox("Pilih Tahun Rilis", ["Semua"] + years)

    filtered_df = df.copy()
    if selected_genre != "Semua":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, na=False)]
    if selected_year != "Semua":
        filtered_df = filtered_df[filtered_df['startYear'] == selected_year]

    # Menampilkan film populer dengan teks di tengah
    st.markdown('<h2 class="center-text">🔥 Film Terpopuler</h2>', unsafe_allow_html=True)
    st.markdown('<p class="center-text">Berikut adalah film-film dengan skor popularitas tertinggi:</p>', unsafe_allow_html=True)


    popular_movies = filtered_df.sort_values('weighted_score', ascending=False).head(10)

    # Menampilkan film populer dalam 3 kolom per baris dengan ukuran card yang sama
    num_columns = 3
    columns = st.columns(num_columns)

    # Loop untuk menampilkan movie card
    for idx, (_, row) in enumerate(popular_movies.iterrows()):
        col_idx = idx % num_columns  # Menentukan kolom berdasarkan index
        with columns[col_idx]:
            st.markdown(
                f"""
                <div class="movie-card" style="
                    border: 1px solid #ccc;
                    padding: 10px;
                    margin: 10px;
                    border-radius: 5px;
                    background-color: #f9f9f9;
                    height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                ">
                    <h4>🎞️ {row['primaryTitle']}</h4>
                    <p><b>Tahun:</b> {int(row['startYear'])} | <b>Genre:</b> {row['genres']}</p>
                    <p><b>Rating:</b> ⭐ {row['averageRating']:.1f} ({int(row['numVotes'])} suara)</p>
                    <p><b>Skor Popularitas:</b> 🔥 {row['weighted_score']:.2f}</p>
                </div>
                """, unsafe_allow_html=True
            )



   # Film Favorit - Rekomendasi
    st.header("🎯 Rekomendasi Berdasarkan Film Favoritmu")
    st.markdown("Cari dan pilih film favoritmu, kami siap rekomendasikan yang seru!")

    movie_list = df['primaryTitle'].dropna().unique()
    selected_movie = st.selectbox(
        "🔎 Ketik atau Pilih Film",
        sorted(movie_list),  # Urutkan abjad biar gampang cari
        index=None,
        placeholder="Cari judul film..."
    )

    if st.button("🎬 Tampilkan Rekomendasi"):
        recommendations = get_recommendations(selected_movie, cosine_sim, df)
        if recommendations is None:
            st.error("Film tidak ditemukan! 😔")
        else:
            st.success(f"Rekomendasi terbaik untuk '{selected_movie}':")
            for _, row in recommendations.iterrows():
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <h4>🎥 {row['primaryTitle']}</h4>
                        <p><b>Tahun:</b> {int(row['startYear'])} | <b>Genre:</b> {row['genres']}</p>
                        <p><b>Rating:</b> ⭐ {row['averageRating']:.1f} ({int(row['numVotes'])} suara)</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
