# Laporan Proyek Machine Learning - Rizqi Alpiansyah
 
## Project Overview
 
Anime adalah animasi komputer buatan tangan yang berasal dari Jepang yang telah menarik banyak pengikut di seluruh dunia. Budaya Jepang telah lama didominasi oleh anime dan manga, dengan gelombang pengakuan yang serupa antar generasi. Lebih beberapa tahun terakhir, pengakuan untuk anime dan *manga* rekan sketsanya telah tumbuh secara signifikan di Inggris Raya dan karena itu Barat. Karena kemampuannya yang khas untuk tumbuh bersama pemirsa, anime tidak diragukan lagi salah satu alasannya itu telah bertahan dalam ujian waktu dan tumbuh dalam kualitas di seluruh dunia. Meskipun anime sangat disukai di Inggris dan Di negara-negara barat, masih sedikit orang yang tidak memiliki rencana tentang apa itu anime. proyek ini bertujuan untuk mengembangkan rekomendasi atau sistem pilihan yang menawarkan saran kepada mereka yang baru mengenal dunia anime dengan menggunakan *KNN* dan *SVD algoritma*.

Referensi: [ANIME RECOMMENDATION SYSTEM]([https://www.irjet.net/archives/V8/i5/IRJET-V8I5679.pdf](https://ijcrt.org/papers/IJCRT2201084.pdf))
 
## Business Understanding
 
### Problem Statements
- Berdasarkan data anime yang ada, bagaimana sistem dapat merekomendasikan anime lain yang mirip dengan anime tersebut?
- Berdasarkan data *rating* yang diberikan pengguna, bagaimana sistem dapat merekomendasikan anime lain yang mungkin disukai pengguna dan belum pernah ditonton oleh pengguna?
 
### Goals
- Menghasilkan rekomendasi anime untuk pengguna dengan teknik *content-based filtering*.
- Menghasilkan rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton pengguna dengan teknik *collaborative filtering*.
 
### Solution Statements
Dalam proyek kali ini saya menggunakan dua teknik pendekatan sistem rekomendasi,  yaitu *content-based filtering* dan *collaborative filtering*. 
-  Tujuan menggunakan *content based filtering* yaitu untuk merekomendasikan anime yang mirip dengan anime yang disukai pengguna lain di masa lalu. hasil dari rekomendasi ini bersifat subjektif karena hanya melihat dari history pengguna terdahulu. model ini dibuat dengan *TF-IDF Vectorizer* dan *Cosine Similarity*.
-  Tujuan menggunakan *Collaborative Based Filtering* yaitu untuk merekomendasikan anime berdasarkan pendapat komunitas pengguna. model ini tidak memerlukan atribut untuk setiap itemnya pada sistem berbasis konten. model ini akan dibuat menggunakan *RecommenderNet*. 

 
## Data Understanding
 
*Dataset* yang saya gunakan adalah *dataset* Anime *Recommendations Database* dari *COOPERUNION*. *dataset* ini berisi 12,300 data pada *dataframe anime.csv*. Selain itu ada juga *Dataframe rating.csv* berisikan 7.813.737 data. 

Sumber: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).
 
**Deskripsi Data**:  

- ***anime.csv**
  - *anime_id - myanimelist.net's unique id identifying an anime.
  - *name - full name of anime.
  - *genre - comma separated list of genres for this anime.
  - *type - movie, TV, OVA, etc.
  - *episodes - how many episodes in this show. (1 if movie).
  - *rating - average rating out of 10 for this anime.
  - *members - number of community members that are in this anime's "group".
- ***rating.csv**
  - *user_id - non identifiable randomly generated user id.
  - *anime_id - the anime that this user has rated.
  - *rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).
 
 
**Exploratory Data Analysis**  
- Menampilkan *head data anime  

  ![data anime](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/animeheadfirst.png?raw=True)
  
- menampilkan jumlah *genre* anime pada *dataset

  ![genre](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/genre.png?raw=True)
  pada gambar di atas bisa dilihat bahwa total *genre* anime pada dataset ada 83 *genre* dari total 12,300 
- Menampilkan head data rating  

  ![data rating](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/ratingheadfitrst.png?raw=True)
  
  
- melihat deskripsi pada dataset rating

  ![rating deskripsi](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/ratingdeskripsi.png?raw=True)

## Data Preparation
tujuan dari data preparation untuk memastikan bahwa data mentah yang sedang disiapkan untuk diproses dan dianalisis akurat dan konsisten sehingga hasil rekomendasi dan analitik akan valid.
### Data Cleaning 
*Data cleaning* yaitu proses mempersiapkan data untuk analisis dengan menghapus atau memodifikasi data yang tidak benar, tidak lengkap, tidak relevan, di duplikat, atau diformat dengan tidak benar. pada program ini saya menghapus data tersebut. data yang kosong dapat ditemukan pada *genre* dan *rating*
### Data Transform
*Data Transform* adalah teknik mengubah data dari satu format ke format lainnya. Transformasi Data dapat dibagi menjadi langkah-langkah berikut. Masing-masing langkah ini akan diterapkan berdasarkan kompleksitas transformasi. teknik ini digunakan pada proyek ini seperti : 
- melakukan persiapan data untuk menyandikan (*encode*) fitur *‘user_id’* dan *‘anime_id’* ke dalam indeks *integer*.
- -memetakan *user_id* dan *anime_id* ke *dataframe* yang berkaitan.
- Mengecek beberapa hal dalam data seperti jumlah *user*, jumlah anime, kemudian mengubah nilai *rating* menjadi *float.*
 ### Feature Engineering
 Membagi Data untuk Training dan Validasi untuk Collaborative Based Filtering. Hal ini dilakukan agar model kita menghindari masalah seperti overfitting dan underfitting.
## Modeling
 
### Model Content Based Filtering
Pada content *Based Filtering*, saya menggunakan *TF-IDF Vectorizer* untuk membangun sistem rekomendasi berdasarkan genre anime. Alasannya adalah untuk menemukan representasi fitur penting dari setiap genre anime. Lalu, saya ubah vektor *tf-idf* dalam bentuk matriks dengan fungsi *todense().* Setelah itu, saya menghitung derajat kesamaan (*similarity degree*) antara anime dengan teknik *cosine similarity*. 
Proses:
 
- *TF-IDF Vectorizer*
digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap kategori genre.
output yang dihasilkan yaitu (12017, 47). Nilai 12017 merupakan ukuran data dan 47 merupakan matrik kategori genre. 

- menghasilkan vektor *tf-idf* dalam bentuk matriks, kita menggunakan fungsi *todense()*
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/matrik1.png?raw=True)

- Membuat *dataframe* untuk melihat *tf-idf matrix*
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/matrik1.png?raw=True)

- matriks *tf-idf* untuk beberapa anime dan kategori anime  
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/tdifmatrik.png?raw=True)

- matriks kesamaan setiap anime dengan menampilkan nama restoran dalam 5 *sampel* kolom (axis = 1) dan 20 *sampel* baris (axis=0).
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/testsample.png?raw=True)


**Kelebihan Content Based Filtering:**
- Model tidak butuh data dari banyak user, karena rekomendasi spesifik untuk satu user.
- Model dapat memberikan rekomendasi yang mirip dengan preferensi user.
 
**Kekurangan Content Based Filtering:**
- Model tidak dapat merekomendasikan hal yang baru untuk user.
 
pada pengujian Model *Content Based* rekomendasi yang diberikan cukup baik 

```python
df_anime[df_anime.name.eq('Boku no Hero Academia')]
```
|anime_id	| name	| genre|	type|	episodes|	rating|	members
| ------- | :---------------------------------: | :----------------------------------------------: | :-----: | :-----: | :-----: | :-----: |
|31964|	Boku no Hero Academia	|Action, Comedy, School, Shounen, Super Power	|TV|	13	|8.36|	282002|
- hasil uji coba 
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/result%20bokunohero.png?raw=True)

 
### Model Collaborative Filtering
Proses:
 
- Menyandikan (*encode*) fitur *user_id* dan *anime_id* ke dalam indeks *integer*.
- Memetakan *user_id* dan *anime_id* ke dataframe yang berkaitan lalu melakukan cek beberapa hal dalam data seperti jumlah *user*, jumlah anime, dan mengubah nilai *rating* menjadi *float*
*output* yang didapat "*Number of User: 234, Number of Anime: 2666, Min Rating: 1.0, Max Rating: 10.0"
- Membagi data menjadi data training dan validasi dengan komposisi 80:20. lalu memetakan (*mapping*) data *user* dan anime menjadi satu *value*
- membuat fungsi Rekomendasi
- Lalu pada pembuatan model, saya menggunakan *RecommenderNet*. Setelah itu saya *me-compile* model ini menggunakan *Binary Crossentropy* untuk menghitung *loss function*, *Adam (Adaptive Moment Estimation)* sebagai *optimizer*, dan *root mean squared error (RMSE)* sebagai *metrics evaluation*. Lalu, saya melatih model dengan *batch size* = 8, dan *epoch* = 100. Untuk mendapatkan rekomendasi, saya membuat fungsi untuk mendapatkan anime yang belum ditonton oleh *user* tersebut dengan mencocokkan *anime_id* yang berada di *anime.csv* dan *rating.csv* . 

**Kelebihan Collaborative Filtering:**  
- Model dapat merekomendasikan hal baru untuk di-explore oleh *user*.
- Model dapat memberikan rekomendasi kepada user berdasarkan preferensi user lain yang mungkin mirip.
 
**Kekurangan Collaborative Filtering:**  
- Model membutuhkan data banyak user.
 
 hasil yang didapat dari model yang sudah dibuat mendapatkan hasil seperti berikut 

![hasil  ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/top%2010%20anime.png?raw=True) 
 
 
## Evaluation
 
Metrik yang saya gunakan untuk model *content based filtering* adalah *cosine similarity*, sedangkan untuk model *collaborative filtering*, metrik yang saya gunakan adalah *root mean squared error (RMSE)*.
 
**Cosine Similarity**:  
*Cosine Similarity* diperoleh dari mengukur sudut cos antara dua vektor yang diproyeksikan dalam ruang multidimensi.  
Rumus Cosine Similarity: 

![Cosine  ](https://i0.wp.com/hendroprasetyo.com/wp-content/uploads/2020/04/image-3.png?resize=407%2C110&ssl=1?raw=True) 
 

**Root Mean Squared Error**:  
Root Mean Squared Error atau RMSE diperoleh dari menghitung akar dari jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.  
Rumus RMSE:  

![rmse  ](https://1.bp.blogspot.com/-AodtifmdR1U/X-NOXo0avGI/AAAAAAAACmI/_jvy7eLB72UB00dW_buPYZCa9ST2yx8XACNcBGAsYHQ/s453/rumus%2Brmse.jpg?raw=True) 
 
 
### Model Content Based Filtering
Pada Content Based Filtering, saya mencoba mengevaluasi model saya dengan memakai matrix precision. Maksud dari precision di sini adalah, berapa banyak genre yang sesuai dengan anime yang dipilih / jumlah rekomendasi. Saya membuat sebuah if loop yang akan membuat variabel 'a' bertambah satu jika genre sama persis dengan anime yang dipilih. Kodenya bisa dilihat seperti ini:
presisi dari model yang dibuat 
```python
a = 0

for row in result.itertuples():
  if (row.genre == 'Action', 'Comedy', 'School', 'Shounen', 'Super Power'):
    a += 1

precision = (a/5)*100
print("presisi darin model yang dibuat {}%".format(precision))
```
output "presisi darin model yang dibuat 100.0%"


### Model Collaborative Filtering
Dari proses training yang dilakukan selama 100 epochs, diperoleh nilai error 0.1129 , dan 0.1518 untuk data validasi. Model dapat memberikan rekomendasi yang cukup baik. Terdapat beberapa anime dengan genre yang mirip dengan anime yang pernah ditonton user, namun juga ada beberapa anime dengan genre baru yang belum pernah ditonton user.
visualisasi metrik yang didapat dari model yang dilatih 

![visuak ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/visualisasi.png?raw=True) 


 
## Kesimpulan
Dari hasil rekomendasi yang diberikan kedua model tersebut, menurut saya kedua model sudah dapat memberikan rekomendasi sesuai dengan yang diharapkan. Namun, untuk mencapai hasil yang lebih baik lagi.


