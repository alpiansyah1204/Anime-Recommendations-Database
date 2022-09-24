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
  
  
- melihat deskripsi pada dataset rating

  ![rating deskripsi](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/ratingdeskripsi.png?raw=True)

## Data Preparation
untuk memastikan bahwa data mentah yang sedang disiapkan untuk diproses dan dianalisis akurat dan konsisten sehingga hasil rekomendasi dan analitik akan valid.

- pada proyek kali ini hanya menggunakan 5000 data dengan cara menambahkan code 

``` python
df_rating = df_rating.drop(range(20000, 7813737)) 
``` 

- mengubah nilai rating -1 ke dalam NaN

``` python
df_rating["rating"].replace({-1: np.nan}, inplace=True)  
``` 

- menghapus baris atau kolom jika bernilai NA.

```python
df_rating = df_rating.dropna(axis = 0, how ='any') 
``` 

- menghitung jumlah data null pada df_anime 

```python
df_anime.isnull().sum()
``` 
 terdapat 62 data kososng pada genre dan 230 pada rating. data ini akan kita hapus agar dapat kita gunakan utnuk membuat rekomendasi anime data. 
 - mengatasi data kosong dengan cara menghapusnya
 
```python
df_anime =  df_anime.dropna()
``` 
 
## Modeling
 
### Model Content Based Filtering
Proses:
 
- TF-IDF Vectorizer
digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap kategori masakan genre.
```python

tf = TfidfVectorizer()
tf.fit(df_anime['genre']) 
```

- fit dan transformasi ke dalam bentuk matriks. 
```python

tfidf_matrix = tf.fit_transform(df_anime['genre']) 
tfidf_matrix.shape
```
output yang dihasilkan yaitu (12017, 47). Nilai 12017 merupakan ukuran data dan 47 merupakan matrik kategori genre. 

- menghasilkan vektor tf-idf dalam bentuk matriks, kita menggunakan fungsi todense()
```python
tfidf_matrix.todense()
```
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/matrik1.png?raw=True)

- Membuat dataframe untuk melihat tf-idf matrix
```python
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=df_anime.name
).sample(22, axis=1).sample(10, axis=0)
```
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/matrik1.png?raw=True)

- matriks tf-idf untuk beberapa anime dan kategori anime  
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/tdifmatrik.png?raw=True)

- matriks kesamaan setiap anime dengan menampilkan nama restoran dalam 5 sampel kolom (axis = 1) dan 20 sampel baris (axis=0).
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/testsample.png?raw=True)


**Kelebihan Content Based Filtering:**
- Model tidak butuh data dari banyak user, karena rekomendasi spesifik untuk satu user.
- Model dapat memberikan rekomendasi yang mirip dengan preferensi user.
 
**Kekurangan Content Based Filtering:**
- Model tidak dapat merekomendasikan hal yang baru untuk user.
 
### Model Collaborative Filtering
Proses:
 
- Menyandikan (encode) fitur user_id dan anime_id ke dalam indeks integer.
- Memetakan user_id dan anime_id ke dataframe yang berkaitan lalu melakukan cek beberapa hal dalam data seperti jumlah user, jumlah anime, dan mengubah nilai rating menjadi float
output yang didapat "Number of User: 234, Number of Anime: 2666, Min Rating: 1.0, Max Rating: 10.0"
- Membagi data menjadi data training dan validasi dengan komposisi 80:20. lalu memetakan (mapping) data user dan anime menjadi satu value
- membuat fungsi Rekomendasi
```python
class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.anime_embedding = layers.Embedding(
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.anime_bias = layers.Embedding(num_anime, 1)
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    anime_vector = self.anime_embedding(inputs[:, 1])
    anime_bias = self.anime_bias(inputs[:, 1])
 
    dot_user_anime = tf.tensordot(user_vector, anime_vector, 2) 
 
    x = dot_user_anime + user_bias + anime_bias
    
    return tf.nn.sigmoid(x)
```
- Lalu pada pembuatan model, saya menggunakan RecommenderNet. Setelah itu saya me-compile model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Lalu, saya melatih model dengan batch size = 8, dan epoch = 100. Untuk mendapatkan rekomendasi, saya membuat fungsi untuk mendapatkan anime yang belum ditonton oleh user tersebut dengan menyocokkan anime_id yang berada di anime.csv dan rating.csv . 


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
 
**Root Mean Squared Error**:  
Root Mean Squared Error atau RMSE diperoleh dari menghitung akar dari jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.  
Rumus RMSE:  
 
### Model Content Based Filtering
pada pengujian Model COntent Based rekomendasi yang diberikan cukup baik 

```python
df_anime[df_anime.name.eq('Boku no Hero Academia')]
```
|anime_id	| name	| genre|	type|	episodes|	rating|	members
| ------- | :---------------------------------: | :----------------------------------------------: | :-----: | :-----: | :-----: | :-----: |
|31964|	Boku no Hero Academia	|Action, Comedy, School, Shounen, Super Power	|TV|	13	|8.36|	282002|
- hasil uji coba 
![tfidf_matrix ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/result%20bokunohero.png?raw=True)

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
lalu hasil yang didapat 
![hasil  ](https://github.com/alpiansyah1204/ML-Terapan2/blob/main/images/top%2010%20anime.png?raw=True) 
 
## Kesimpulan
Dari hasil rekomendasi yang diberikan kedua model tersebut, menurut saya kedua model sudah dapat memberikan rekomendasi sesuai dengan yang diharapkan. Namun, untuk mencapai hasil yang lebih baik lagi.
