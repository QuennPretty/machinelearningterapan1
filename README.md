# Laporan Proyek Machine Learning - Pretty  N Simanjuntak

## Domain Proyek
Proyek ini menggunakan machine learning untuk mengklasifikasikan kinerja siswa menjadi baik atau kurang. Data yang digunakan mencakup nilai ujian, jenis kelamin, pendidikan orang tua, dan faktor lainnya. Tujuan utamanya adalah untuk membantu institusi pendidikan (seperti sekolah) dalam mengenali siswa yang mungkin membutuhkan bantuan atau intervensi lebih awal. Dengan identifikasi dini, diharapkan prestasi siswa dapat ditingkatkan dan kesenjangan pencapaian dapat dikurangi. Masalah ini penting untuk diselesaikan karena performa akademis siswa memiliki dampak jangka panjang terhadap peluang pendidikan dan karir mereka di masa depan. Pendekatan machine learning dipilih karena kemampuannya untuk menganalisis pola kompleks dari berbagai faktor yang mungkin mempengaruhi kinerja siswa, yang mungkin sulit diidentifikasi melalui analisis manual.

### Referensi:
(Bagian ini dapat diisi dengan referensi dari literatur atau riset terkait jika ada. Notebook yang diberikan tidak menyertakan referensi eksternal, namun untuk laporan formal, sumber seperti Google Scholar dapat digunakan untuk mencari studi relevan mengenai prediksi performa siswa).

Nurmalitasari, Awang Long, Z., & Mohd Noor, M. F. (2023). Factors Influencing Dropout Students in Higher Education. Education Research International, 2023, Article ID 7704142. https://doi.org/10.1155/2023/7704142

Latar belakang sosial ekonomi dan demografis terbukti sangat memengaruhi prestasi akademik siswa (OECD, 2019; Sirin, 2005).



## Business Understanding
Pada bagian ini, dijelaskan proses klarifikasi masalah terkait klasifikasi performa siswa.

### Problem Statements
Berdasarkan latar belakang dan tujuan proyek yang dijelaskan dalam notebook:

Pernyataan Masalah 1: Institusi pendidikan memerlukan cara yang efektif untuk mengidentifikasi siswa yang berpotensi memiliki performa akademis kurang secara dini.

Pernyataan Masalah 2: Terdapat berbagai faktor demografis dan terkait persiapan belajar yang mungkin mempengaruhi performa siswa, namun pola pengaruhnya belum tentu mudah terlihat tanpa analisis data mendalam.

Pernyataan Masalah 3: Bagaimana membangun model prediktif yang akurat untuk mengklasifikasikan performa siswa (baik atau kurang) berdasarkan data yang tersedia?

## Goals
Tujuan dari proyek ini berdasarkan pernyataan masalah adalah:

Jawaban Pernyataan Masalah 1 & 3: Mengembangkan model klasifikasi machine learning yang mampu memprediksi apakah seorang siswa akan memiliki performa "baik" (rata-rata skor >= 70) atau "kurang" (rata-rata skor < 70) dengan akurasi dan F1-score yang tinggi.

Jawaban Pernyataan Masalah 2: Mengidentifikasi fitur-fitur (seperti skor ujian, jenis kelamin, tingkat pendidikan orang tua, dll.) yang paling berpengaruh dalam menentukan klasifikasi performa siswa.

### Solution statements
Untuk mencapai tujuan tersebut, pendekatan solusi yang diimplementasikan dan dievaluasi dalam notebook adalah:

Solusi 1: Mengembangkan model baseline menggunakan algoritma Random Forest Classifier dengan parameter default untuk mendapatkan gambaran awal performa.

Solusi 2: Melakukan optimasi pada model Random Forest Classifier menggunakan hyperparameter tuning dengan GridSearchCV untuk meningkatkan performanya.

Solusi 3: Mengembangkan dan mengevaluasi model klasifikasi lain sebagai perbandingan, yaitu Logistic Regression dan Support Vector Machine (SVM).

Solusi yang diberikan akan diukur menggunakan metrik evaluasi klasifikasi seperti Akurasi, Presisi, Recall, dan F1-Score, dengan F1-Score sebagai metrik utama karena potensi ketidakseimbangan kelas.

## Data Understanding
Data yang digunakan dalam proyek ini adalah "StudentsPerformance.csv". Dataset ini berisi informasi mengenai performa siswa dalam ujian matematika, membaca, dan menulis, serta beberapa atribut demografis dan latar belakang lainnya.

Dataset ini terdiri dari 1000 baris data siswa dengan 8 kolom awal.
Sumber Data:https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

Variabel-variabel pada dataset StudentsPerformance.csv adalah sebagai berikut:

| Variabel                      | Deskripsi                                                                 | Contoh Nilai                        |
| :---------------------------- | :------------------------------------------------------------------------ | :---------------------------------- |
| `gender`                      | Jenis kelamin siswa                                                       | 'female', 'male'                    |
| `race/ethnicity`              | Kelompok ras/etnis siswa                                                  | 'group A', 'group B', dst.          |
| `parental level of education` | Tingkat pendidikan orang tua                                              | "bachelor's degree", "some college" |
| `lunch`                       | Jenis makan siang yang diterima siswa                                     | 'standard', 'free/reduced'          |
| `test preparation course`     | Apakah siswa menyelesaikan kursus persiapan ujian atau tidak             | 'none', 'completed'                 |
| `math score`                  | Skor ujian matematika siswa                                               | (Numerik)                           |
| `reading score`               | Skor ujian membaca siswa                                                  | (Numerik)                           |
| `writing score`               | Skor ujian menulis siswa                                                  | (Numerik)                           |

Dari analisis awal (df.info()), diketahui bahwa tidak ada nilai yang hilang (missing values) pada dataset dan tipe data untuk setiap kolom sudah sesuai. Pengecekan duplikasi (df.duplicated().sum()) juga menunjukkan tidak ada baris data yang duplikat.

## Feature Engineering
Dilakukan penambahan dua fitur baru untuk membantu analisis dan pemodelan:

average_score: Dihitung sebagai rata-rata dari math score, reading score, dan writing score. Fitur ini merepresentasikan performa akademis keseluruhan siswa.

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

performance: Variabel target biner yang dibuat berdasarkan average_score. Siswa dengan average_score >= 70 diklasifikasikan sebagai performa baik (1), dan di bawah 70 sebagai performa kurang (0).

df['performance'] = (df['average_score'] >= 70).astype(int)

Distribusi kelas performance menunjukkan:

Performa Kurang (0): 541 siswa

Performa Baik (1): 459 siswa
Ini menunjukkan adanya sedikit ketidakseimbangan kelas yang perlu dipertimbangkan saat evaluasi model.

## Data Preparation
Tahapan persiapan data yang dilakukan secara berurutan adalah sebagai berikut:

Label Encoding untuk Fitur Kategorikal:

Penjelasan Proses: Fitur-fitur kategorikal seperti gender, race/ethnicity, parental level of education, lunch, dan test preparation course diubah menjadi representasi numerik menggunakan LabelEncoder dari sklearn.preprocessing.

Alasan: Sebagian besar algoritma machine learning memerlukan input dalam bentuk numerik. Label Encoding mengubah setiap kategori unik dalam sebuah fitur menjadi angka integer.

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

### Pembagian Data (Train-Test Split):

Penjelasan Proses: Dataset dibagi menjadi dua set: data latih (training set) dan data uji (testing set). Fitur (X) terdiri dari kolom-kolom kategorikal yang telah di-encode, skor-skor numerik (math score, reading score, writing score), dan fitur baru average_score. Variabel target (y) adalah kolom performance.

Alasan: Pembagian data ini penting untuk melatih model pada satu subset data dan kemudian mengevaluasi kinerjanya pada subset data lain yang belum pernah dilihat sebelumnya oleh model. Ini membantu mengukur kemampuan generalisasi model dan menghindari overfitting. Data dibagi dengan rasio 80% untuk pelatihan dan 20% untuk pengujian. Penggunaan random_state=42 memastikan hasil pembagian data konsisten dan dapat direproduksi.

from sklearn.model_selection import train_test_split

# Fitur X termasuk 'average_score'
X = df[categorical_cols + ['math score', 'reading score', 'writing score', 'average_score']]
y = df['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Bentuk data setelah pembagian:

X_train shape: (800, 9)

X_test shape: (200, 9)

y_train shape: (800,)

y_test shape: (200,)

(Catatan: Penggunaan average_score sebagai fitur input (X) untuk memprediksi performance yang didefinisikan langsung dari average_score berpotensi menyebabkan data leakage dan hasil model yang terlalu optimis. Dalam praktik nyata, fitur yang secara langsung menentukan target biasanya tidak disertakan sebagai prediktor).

## Modeling
Tahapan ini membahas model machine learning yang digunakan untuk klasifikasi performa siswa. Tiga algoritma utama dieksplorasi.

Random Forest Classifier (Baseline & Tuned)

Penjelasan: Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan (decision trees) dan menggabungkan hasilnya (mayoritas suara untuk klasifikasi).

Parameter (Baseline): Menggunakan parameter default dari sklearn.ensemble.RandomForestClassifier dengan random_state=42 untuk reproduktifitas.

Parameter (Tuned): Dilakukan hyperparameter tuning menggunakan GridSearchCV untuk mencari kombinasi parameter terbaik. Parameter yang diuji:

n_estimators: [50, 100, 150]

max_depth: [None, 10, 20]

min_samples_split: [2, 5, 10]
GridSearchCV menggunakan validasi silang 5-fold (cv=5) dan metrik scoring='f1'.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#### Baseline
rf_model_baseline = RandomForestClassifier(random_state=42)
rf_model_baseline.fit(X_train, y_train)

#### Tuned
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

Kelebihan: Cenderung memiliki akurasi tinggi, tahan terhadap overfitting (dibandingkan satu pohon keputusan), dan dapat menangani berbagai tipe data serta memberikan informasi feature importance.

Kekurangan: Bisa menjadi "kotak hitam" (sulit diinterpretasi cara kerjanya secara detail) dan membutuhkan lebih banyak sumber daya komputasi dibandingkan model linear.

## Logistic Regression

Penjelasan: Model linear yang menggunakan fungsi logistik untuk memprediksi probabilitas suatu kelas.

Parameter: Menggunakan random_state=42, solver='liblinear', dan max_iter=200.

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
lr_model.fit(X_train, y_train)

Kelebihan: Sederhana, cepat dilatih, dan hasilnya mudah diinterpretasi.

Kekurangan: Mengasumsikan hubungan linear antara fitur dan log-odds target, mungkin kurang akurat untuk masalah non-linear kompleks.

Support Vector Machine (SVM)

Penjelasan: Algoritma yang mencari hyperplane optimal untuk memisahkan data ke dalam kelas-kelas yang berbeda.

Parameter: Menggunakan random_state=42 dan probability=True (untuk memungkinkan perolehan probabilitas prediksi).

from sklearn.svm import SVC
svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, y_train)

Kelebihan: Efektif dalam ruang dimensi tinggi dan untuk kasus di mana jumlah dimensi lebih besar dari jumlah sampel. Fleksibel dengan penggunaan kernel yang berbeda.

Kekurangan: Kurang efisien untuk dataset besar, dan pemilihan parameter kernel yang tepat bisa menjadi tantangan.

Pemilihan Model Terbaik
Model terbaik dipilih berdasarkan F1-Score tertinggi pada data uji. F1-Score dipilih karena memberikan keseimbangan antara presisi dan recall, yang penting mengingat adanya sedikit ketidakseimbangan kelas. Berdasarkan hasil evaluasi, Random Forest Classifier (baik versi baseline maupun yang sudah di-tuning) menunjukkan F1-Score tertinggi.

## Evaluation
Metrik evaluasi yang digunakan untuk menilai kinerja model klasifikasi adalah:

Accuracy: Rasio prediksi yang benar terhadap total prediksi.

Formula: (TP+TN)/(TP+TN+FP+FN)

Precision: Dari semua yang diprediksi sebagai kelas positif, berapa yang benar-benar positif.

Formula: TP/(TP+FP)

Recall (Sensitivity): Dari semua yang sebenarnya positif, berapa yang berhasil diprediksi sebagai positif.

Formula: TP/(TP+FN)

F1-Score: Rata-rata harmonik dari Precision dan Recall.

Formula: 2×(Precision×Recall)/(Precision+Recall)

Classification Report: Ringkasan Precision, Recall, F1-Score untuk setiap kelas.

Confusion Matrix: Tabel yang menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN).

Hasil Proyek Berdasarkan Metrik Evaluasi
Berikut adalah ringkasan perbandingan kinerja model pada data uji:

| Model                      | Accuracy | F1-Score |
| :------------------------- | :------: | :------: |
| Random Forest (Baseline)   | 1.0000   | 1.0000   |
| Random Forest (Tuned)      | 1.0000   | 1.0000   |
| Logistic Regression        | 0.9300   | 0.9231   |
| SVM                        | 0.9900   | 0.9888   |



Analisis Hasil:

Random Forest Classifier (Baseline dan Tuned) mencapai performa sempurna (Accuracy 1.0000 dan F1-Score 1.0000) pada data uji. Parameter terbaik yang ditemukan untuk Random Forest Tuned adalah {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}.

SVM juga menunjukkan performa yang sangat baik dengan Accuracy 0.9900 dan F1-Score 0.9888.

Logistic Regression memiliki performa yang baik namun sedikit di bawah Random Forest dan SVM, dengan Accuracy 0.9300 dan F1-Score 0.9231.

Berdasarkan F1-Score tertinggi, Random Forest Classifier (Tuned) dipilih sebagai model terbaik.
Catatan: Skor sempurna seringkali mengindikasikan adanya masalah seperti data leakage, terutama karena average_score (yang langsung menentukan performance) digunakan sebagai fitur. Dalam skenario nyata, ini perlu investigasi lebih lanjut.

Visualisasi Confusion Matrix (untuk Tuned Random Forest):
Confusion matrix untuk model Random Forest yang di-tuning menunjukkan:

[[110   0]
 [  0  90]]

Ini berarti:

True Negatives (TN): 110 (siswa dengan performa kurang diprediksi benar)

False Positives (FP): 0 (siswa dengan performa kurang diprediksi baik)

False Negatives (FN): 0 (siswa dengan performa baik diprediksi kurang)

True Positives (TP): 90 (siswa dengan performa baik diprediksi benar)

[Visualisasi Confusion Matrix untuk Tuned Random Forest akan ditampilkan di sini jika notebook dieksekusi]

Visualisasi Feature Importances (untuk Tuned Random Forest):
Analisis feature importances dari model Random Forest terbaik menunjukkan fitur-fitur yang paling berpengaruh:

average_score: Kontribusi paling signifikan.

writing score

reading score

math score
Fitur-fitur kategorikal lainnya memiliki kontribusi yang jauh lebih kecil.

                       Feature  Importance
8                average_score    0.394031
7                writing score    0.288240
6                reading score    0.181349
5                   math score    0.121525
4      test preparation course    0.006159
0                       gender    0.003006
3                        lunch    0.002779
1               race/ethnicity    0.002147
2  parental level of education    0.000763
```[Visualisasi Feature Importances untuk Tuned Random Forest akan ditampilkan di sini jika notebook dieksekusi]`

Kesimpulan dari *feature importance* ini (dalam konteks notebook) adalah bahwa skor rata-rata dan skor individu mata pelajaran adalah prediktor terkuat untuk performa siswa secara keseluruhan.


