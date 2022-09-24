# Laporan Proyek Machine Learning - Rizqi Alpiansyah
 
## Project Overview
 
Anime adalah animasi komputer buatan tangan yang berasal dari Jepang yang telah menarik banyak pengikut di seluruh dunia. Budaya Jepang telah lama didominasi oleh anime dan manga, dengan gelombang pengakuan yang serupa antar generasi. Lebih beberapa tahun terakhir, pengakuan untuk anime dan manga rekan sketsanya telah tumbuh secara signifikan di Inggris Raya dan karena itu Barat. Karena kemampuannya yang khas untuk tumbuh bersama pemirsa, anime tidak diragukan lagi salah satu alasannya itu telah bertahan dalam ujian waktu dan tumbuh dalam kualitas di seluruh dunia. Meskipun anime sangat disukai di Inggris dan Di negara-negara barat, masih sedikit orang yang tidak memiliki rencana tentang apa itu anime. proyek ini bertujuan untuk mengembangkan rekomendasi atau sistem pilihan yang menawarkan saran kepada mereka yang baru mengenal dunia anime dengan menggunakan KNN dan SVD algoritma.

Referensi: [ANIME RECOMMENDATION SYSTEM]([https://www.irjet.net/archives/V8/i5/IRJET-V8I5679.pdf](https://ijcrt.org/papers/IJCRT2201084.pdf))
 
## Business Understanding
 
### Problem Statements
- Berdasarkan data anime yang ada, bagaimana sistem dapat merekomendasikan anime lain yang mirip dengan anime tersebut?
- Berdasarkan data rating yang diberikan pengguna, bagaimana sistem dapat merekomendasikan anime lain yang mungkin disukai pengguna dan belum pernah ditonton oleh pengguna?
 
### Goals
- Menghasilkan rekomendasi anime untuk pengguna dengan teknik content-based filtering.
- Menghasilkan rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton pengguna dengan teknik collaborative filtering.
 
### Solution Statements
Dalam proyek kali ini saya menggunakan dua teknik pendektan sistem rekomendasi,  yaitu content-based filtering dan collaborative filtering. 
- Menggunakan Content Based Filtering tujuan menggunakan content based filtering yaitu untuk merekomendasikan anime yang mirip dengan anime yang disukai pengguna lain dimasa lalu. hasil dari rekomendasi ini bersifat subjektif karena hanya melihat dari history pengguna terdahulu. model ini dibuat dengan TF-IDF Vectorizer dan Cosine Similarity.
- Menggunakan Collaborative Based Filtering tujuan menggunakan Collaborative Based Filtering yaitu untuk merekomendasikan anime berdasarkan pendapat komunitas pengguna. model ini tidak memerlukan atribut untuk setiap itemnya pada sistem berbasis konten. model ini akan dibuat menggunakan RecommenderNet. 

 
## Data Understanding
 
Dataset yang saya gunaka adalah dataset Anime Recommendations Database dari COOPERUNION. dataset ini berisi 12,300 data pada dataframe anime.csv. Selain itu ada juga Dataframe rating.csv berisikan 7.813.737 data. 

Sumber: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).
 
**Deskripsi Data**:  
- **anime.csv**
  - anime_id - myanimelist.net's unique id identifying an anime.
  - name - full name of anime.
  - genre - comma separated list of genres for this anime.
  - type - movie, TV, OVA, etc.
  - episodes - how many episodes in this show. (1 if movie).
  - rating - average rating out of 10 for this anime.
  - members - number of community members that are in this anime's "group".
- **rating.csv**
  - user_id - non identifiable randomly generated user id.
  - anime_id - the anime that this user has rated.
  - rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
 
 
**Exploratory Data Analysis**  
- Menampilkan head data anime  

  ![data anime](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/animeheadfirst.png?raw=True)
  
- menampilakn jumlah genre anime pada dataset

  ![genre](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/genre.png?raw=True)
  
- Menampilkan head data rating  

  ![data rating](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/ratingheadfitrst.png?raw=True)
  
- pada proyek kali ini hanya menggunakan 5000 data dengan cara menambahkan code 

  ``` df_rating = df_rating.drop(range(20000, 7813737))  ``` 
  
- melihat deskripsi pada dataset rating

  ![rating deskripsi](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/ratingdeskripsi.png?raw=True)
  
- mengubah nilai rating -1 ke dalam NaN

``` df_rating["rating"].replace({-1: np.nan}, inplace=True)  ``` 

- menghapus baris atau kolom jika bernilai NA.

``` df_rating = df_rating.dropna(axis = 0, how ='any')  ``` 
 
- menghitung jumlah data null pada df_anime 

 ``` df_anime.isnull().sum() ``` 
 
 - mengatasi data kosong dengan cara menghapusnya
 
 ```df_anime =  df_anime.dropna()``` 
 
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
