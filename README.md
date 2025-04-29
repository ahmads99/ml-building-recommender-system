# Dashboard Visualisasi 
[Lihat dashboard](https://lookerstudio.google.com/reporting/0ef1f331-f668-4c85-8e53-0b732f5734c5)
![image](https://github.com/user-attachments/assets/53196075-9a39-4d74-ac8b-7f7121f7682e)


# 🎬 Sistem Rekomendasi Film

## Ringkasan
Proyek ini adalah bagian dari submission **Machine Learning with Python: Building Recommender System**. Sistem rekomendasi film ini mengimplementasikan dua pendekatan utama: **Popularity-Based Recommender** dan **Content-Based Filtering**. Sistem ini menggunakan dataset IMDb (title.basics.tsv dan title.ratings.tsv) untuk merekomendasikan film berdasarkan popularitas (rating tertimbang) dan kesamaan genre. Fokusnya adalah film yang dirilis antara 2010 hingga 2019, menyesuaikan dengan tren film modern.

## Fitur
- **Popularity-Based Recommender**: Menyajikan rekomendasi film terpopuler berdasarkan formula rating tertimbang IMDb, yang menggabungkan `averageRating` dan `numVotes`.
- **Content-Based Filtering**: Rekomendasi film berdasarkan kesamaan genre dengan menggunakan teknik **TF-IDF** dan **cosine similarity**.
- **Analisis Data**: Memberikan insight tentang tren film, seperti distribusi tahun rilis, durasi genre, dan perbandingan konten dewasa vs non-dewasa.
- **Visualisasi**: Menampilkan scatter plot dan boxplot untuk menggambarkan distribusi rating, popularitas, dan variasi durasi berdasarkan genre.

## Dataset
- **Sumber**: Dataset IMDb yang disediakan oleh DQLab (akses publik).
- **File**:
  - `title.basics.tsv`: Metadata film (tconst, titleType, primaryTitle, startYear, runtimeMinutes, genres, dll.)
  - `title.ratings.tsv`: Rating film (tconst, averageRating, numVotes)

## Preprocessing
- Menangani nilai hilang (NaN) dengan mengisi median pada `startYear` dan `runtimeMinutes` atau "Unknown" pada `primaryTitle` dan `genres`.
- Menghapus kolom `endYear` karena 99.2% nilainya hilang.
- Mengonversi `startYear` dan `runtimeMinutes` ke tipe numerik.
- Memfilter hanya film (titleType == 'movie') yang dirilis antara tahun 2010 hingga 2019.

## Persyaratan
Untuk menjalankan proyek ini, pastikan kamu telah menginstal library Python berikut:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
