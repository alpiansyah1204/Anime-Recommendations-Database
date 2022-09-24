# Laporan Proyek Machine Learning - Rizqi Alpiansyah
 
## Project Overview
 
Anime adalah animasi komputer buatan tangan yang berasal dari Jepang yang telah menarik banyak pengikut di seluruh dunia. Budaya Jepang telah lama didominasi oleh anime dan manga, dengan gelombang pengakuan yang serupa antar generasi. Lebih beberapa tahun terakhir, pengakuan untuk anime dan manga rekan sketsanya telah tumbuh secara signifikan di Inggris Raya dan karena itu Barat. Karena kemampuannya yang khas untuk tumbuh bersama pemirsa, anime tidak diragukan lagi salah satu alasannya itu telah bertahan dalam ujian waktu dan tumbuh dalam kualitas di seluruh dunia. Meskipun anime sangat disukai di Inggris dan Di negara-negara barat, masih sedikit orang yang tidak memiliki rencana tentang apa itu anime. Makalah ini bertujuan untuk mengembangkan rekomendasi atau sistem pilihan yang menawarkan saran kepada mereka yang baru mengenal dunia anime dengan menggunakan KNN dan SVD algoritma.

Referensi: [Movie Recommendation System](https://www.irjet.net/archives/V8/i5/IRJET-V8I5679.pdf)
 
## Business Understanding
 
### Problem Statements
- Berdasarkan data film yang ada, bagaimana sistem dapat merekomendasikan film lain yang mirip dengan film tersebut?
- Berdasarkan data rating yang diberikan pengguna, bagaimana sistem dapat merekomendasikan film lain yang mungkin disukai pengguna dan belum pernah ditonton oleh pengguna?
 
### Goals
- Menghasilkan rekomendasi film untuk pengguna dengan teknik content-based filtering.
- Menghasilkan rekomendasi film yang sesuai dengan preferensi pengguna dan belum pernah ditonton pengguna dengan teknik collaborative filtering.
 
### Solution Statements
Saya menggunakan dua teknik / pendekatan sistem rekomendasi, yaitu content-based filtering dan collaborative filtering. Untuk content-based filtering, metrik yang saya gunakan adalah cosine similarity, sedangkan untuk collaborative filtering, metrik yang saya gunakan adalah RMSE.
 
## Data Understanding
 
Dataset yang saya gunakan adalah dataset film dan rating dari MovieLens. Dataset ini terdiri dari 20.000.263 rating dan 465.564 tag pada 27.278 data film oleh 138.493 user. Dataset ini terdiri dari total 6 file terpisah, tag.csv, rating.csv, movie.csv, link.csv, genome_scores.csv, dan genome_tags.csv. Untuk project ini, saya hanya menggunakan 2 file, yaitu movie.csv dan rating.csv.
 
Sumber: [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).
 
**Deskripsi Data**:  
- movies: merupakan informasi mengenai film.
- ratings: merupakan rating film yang diberikan oleh user.
 
**Exploratory Data Analysis**  
- Menampilkan data movies  
  ![Movies](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/movies.png?raw=True)
 
  movies terdiri dari 3 kolom, yaitu movieId, title, dan genres.
- Visualisasi genres  
  ![Genres](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/movies_genres.png?raw=True)
 
  Terdapat 20 genre film. Drama adalah genre film terbanyak, disusul oleh Comedy, Thriller, Romance, Action dan seterusnya.
- Menampilkan data ratings  
  ![Ratings](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/ratings.png?raw=True)
 
  ratings terdiri dari 4 kolom, yaitu userId, movieId, rating, dan timestamp.
- Cek data ratings  
  ![Ratings Desc](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/ratings_describe.png?raw=True)
 
  Dari output diatas, dapat dilihat bahwa nilai rating terendah adalah 0.5, sedangkan nilai rating tertinggi adalah 5.0.
 
## Data Preparation
 
- Menggabungkan dataframe movies dan ratings. Tujuannya adalah agar kita dapat mengetahui film yang pernah ditonton user dan memfilter film yang belum pernah dinilai user.
  ```python
  df_fix = pd.merge(movies, ratings, on='movieId', how='left')
  ```
- Drop kolom timestamp. Data ini di-drop karena tidak dibutuhkan untuk membuat rekomendasi film.
  ```python
  df_fix.drop('timestamp', axis=1, inplace=True)
  ```
- Menangani missing value.
  ```python
  df_fix.dropna(inplace=True)
  ```  
  ![Missing Value](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/missing_value.png?raw=True)
 
  Terdapat 534 baris missing value pada kolom userId dan rating. Artinya, ada 534 film yang belum pernah dinilai oleh user. Data ini tidak dapat digunakan untuk membuat rekomendasi film. Oleh karena itu, dengan pertimbangan bahwa jumlah data ini tidak banyak dibandingkan dengan total data film yang ada, data ini akan di-drop.
  ```python
  df_fix.dropna(inplace=True)
  ```
 
## Modeling
 
### Model Content Based Filtering
Proses:
 
- Untuk model content based filtering, data yang dibutuhkan hanya data film saja. Oleh karena itu, saya membuat dataframe baru yang hanya berisi data movieId, title, dan genres.
- Menghapus data yang duplikat karena tidak diperlukan untuk model ini.
- Satu film dapat dikategorikan ke dalam banyak genre. Genre-genre ini perlu direpresentasikan dalam bentuk matriks untuk memudahkan perhitungan kemiripan film yang satu dengan yang lain.  
  ![Genres Matrix](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_genres_matrix.png?raw=True)
 
  Dari output diatas, dapat dilihat bahwa film dengan id 0 dikategorikan film dengan genre Adventure, Animation, Children, dan Comedy.
- Menghitung cosine similarity untuk mengetahui tingkat kemiripan antar film.  
  ![Cosine Similarity](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_cosine_similarity.png?raw=True)
 
  Dari output diatas, dapat dilihat bahwa film Gaudi Afternoon (2001) 0.5% mirip dengan film Heaven Can Wait (1978) tapi tidak mirip sama sekali dengan film Strait-Jacket (1964).
- Mendapatkan top-20 rekomendasi film.  
  Film pertama  
  ![Film 1](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_film_1.png?raw=True)  
  Rekomendasi  
  ![Rec Film 1](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_rec_film_1.png?raw=True)
 
  Film kedua  
  ![Film 2](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_film_2.png?raw=True)  
  Rekomendasi  
  ![Rec Film 2](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cb_rec_film_2.png?raw=True)
 
**Kelebihan Content Based Filtering:**
- Model tidak butuh data dari banyak user, karena rekomendasi spesifik untuk satu user.
- Model dapat memberikan rekomendasi yang mirip dengan preferensi user.
 
**Kekurangan Content Based Filtering:**
- Model tidak dapat merekomendasikan hal yang baru untuk user.
 
### Model Collaborative Filtering
Proses:
 
- Melakukan shuffling data agar distribusi data menjadi random dan menghindari overfitting.
- Encoding data userId dan movieId. Hal ini dilakukan untuk memudahkan identifikasi data user dan film yang ada.
- Split data, 90% untuk training, dan 10% untuk validation. Hal ini dilakukan untuk menguji keakuratan model yang telah dilatih.  
- Membuat arsitektur model.  
  ![Arsitektur Model](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cl_model.png?raw=True)
- Training model menggunakan binary crossentropy loss function, adam optimizer dan metrik RMSE.  
  ![Training Model](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cl_training.png?raw=True)
 
  Dari hasil training selama 3 epochs, diperoleh nilai error RMSE 0.1557, dan 0.1748 untuk data validasi.
 
  ![Visualisasi Metrik](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cl_visualisasi_metrik.png?raw=True)
- Mendapatkan top-20 rekomendasi film.  
  Film yang pernah ditonton dan diberi rating tinggi oleh user 17759.  
  ![Watched Movie](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cl_watched.png?raw=True)  
  Rekomendasi untuk user 17759.  
  ![Rec for User](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/cl_rec.png?raw=True)
 
**Kelebihan Collaborative Filtering:**  
- Model dapat merekomendasikan hal baru untuk di-explore oleh user.
- Model dapat memberikan rekomendasi kepada user berdasarkan preferensi user lain yang mungkin mirip.
 
**Kekurangan Collaborative Filtering:**  
- Model membutuhkan data banyak user.
 
## Evaluation
 
Metrik yang saya gunakan untuk model content based filtering adalah cosine similarity, sedangkan untuk model collaborative filtering, metrik yang saya gunakan adalah root mean squared error (RMSE).
 
**Cosine Similarity**:  
Cosine Similarity diperoleh dari mengukur sudut cos antara dua vektor yang diproyeksikan dalam ruang multidimensi.  
Rumus Cosine Similarity:  
![Rumus Cosine Sim](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/rumus_cosine_sim.png?raw=True)
 
**Root Mean Squared Error**:  
Root Mean Squared Error atau RMSE diperoleh dari menghitung akar dari jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.  
Rumus RMSE:  
![Rumus RMSE](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S2_Recommendation_System/image/rumus_rmse.png?raw=True)
 
### Model Content Based Filtering
Model dapat memberikan rekomendasi yang cukup baik dengan skor cosine similarity rata-rata diatas 0.9 untuk film pertama dan diatas 0.7 untuk film kedua. Perbedaan skor ini dapat disebabkan karena data yang tidak seimbang. Namun, secara keseluruhan, model dapat memberikan rekomendasi film yang mirip dengan film yang diinputkan.
 
### Model Collaborative Filtering
Dari proses training yang dilakukan selama 3 epochs, diperoleh nilai error 0.1557, dan 0.1748 untuk data validasi. Model dapat memberikan rekomendasi yang cukup baik. Terdapat beberapa film dengan genre yang mirip dengan film yang pernah ditonton user, namun juga ada beberapa film dengan genre baru yang belum pernah ditonton user.
 
## Kesimpulan
Dari hasil rekomendasi yang diberikan kedua model tersebut, menurut saya kedua model sudah dapat memberikan rekomendasi sesuai dengan yang diharapkan. Namun, untuk mencapai hasil yang lebih baik lagi, masih banyak hal yang harus dilakukan, terutama memperbanyak dataset untuk meningkatkan sebaran data dan meningkatkan performa model collaborative filtering.    
