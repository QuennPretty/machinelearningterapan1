# Laporan Proyek Machine Learning - Pretty N Simanjuntak

## Domain Proyek
Proyek ini menggunakan machine learning untuk mengklasifikasikan kinerja siswa menjadi baik atau kurang. Data yang digunakan mencakup nilai ujian, jenis kelamin, pendidikan orang tua, dan faktor lainnya. Tujuan utamanya adalah untuk membantu institusi pendidikan dalam mengenali siswa yang mungkin membutuhkan bantuan atau intervensi lebih awal untuk meningkatkan prestasi mereka.

Masalah ini penting untuk diselesaikan karena performa akademis siswa memiliki dampak jangka panjang terhadap peluang pendidikan dan karir mereka di masa depan. Dengan mengidentifikasi faktor-faktor kunci yang memengaruhi kinerja, institusi dapat merancang program dukungan yang lebih efektif. Pendekatan machine learning dipilih karena kemampuannya untuk menganalisis pola kompleks dari berbagai faktor yang mungkin mempengaruhi kinerja siswa, yang sulit diidentifikasi melalui analisis manual.

### Referensi:
- Nurmalitasari, Awang Long, Z., & Mohd Noor, M. F. (2023). Factors Influencing Dropout Students in Higher Education. *Education Research International, 2023*, Article ID 7704142. https://doi.org/10.1155/2023/7704142
- Sirin, S. R. (2005). Socioeconomic Status and Academic Achievement: A Meta-Analytic Review of Research. *Review of Educational Research, 75*(3), 417–453.

## Business Understanding

Pada bagian ini, dijelaskan proses klarifikasi masalah terkait klasifikasi performa siswa.

### Problem Statements
- **Pernyataan Masalah 1:** Institusi pendidikan memerlukan cara yang efektif untuk mengidentifikasi siswa yang berpotensi memiliki performa akademis kurang secara dini.
- **Pernyataan Masalah 2:** Terdapat berbagai faktor demografis dan terkait persiapan belajar yang mungkin mempengaruhi performa siswa, namun pola pengaruhnya belum tentu mudah terlihat tanpa analisis data mendalam.
- **Pernyataan Masalah 3:** Bagaimana membangun model prediktif yang akurat untuk mengklasifikasikan performa siswa (baik atau kurang) berdasarkan data yang tersedia?

### Goals
- **Jawaban Pernyataan Masalah 1 & 3:** Mengembangkan dan mengevaluasi beberapa model klasifikasi *machine learning* untuk menemukan model terbaik yang mampu memprediksi secara akurat apakah seorang siswa akan memiliki performa "baik" atau "kurang".
- **Jawaban Pernyataan Masalah 2:** Mengidentifikasi fitur-fitur (seperti skor ujian, jenis kelamin, dll.) yang paling berpengaruh dalam menentukan klasifikasi performa siswa melalui analisis *feature importance* dari model terbaik.
- **Goal Tambahan:** Menetapkan target keberhasilan dengan F1-Score sebagai metrik utama untuk memastikan model dapat diandalkan sekalipun terdapat sedikit ketidakseimbangan kelas.

### Solution statements
- **Solusi 1:** Mengembangkan model baseline menggunakan algoritma Random Forest Classifier dengan parameter default untuk mendapatkan gambaran awal performa.
- **Solusi 2:** Melakukan optimasi pada model Random Forest Classifier menggunakan hyperparameter tuning dengan `GridSearchCV` untuk meningkatkan performanya.
- **Solusi 3:** Mengembangkan dan mengevaluasi model klasifikasi lain sebagai perbandingan, yaitu Logistic Regression dan Support Vector Machine (SVM), untuk memastikan solusi yang dipilih adalah yang paling optimal.

## Data Understanding
Proyek ini menggunakan dataset "StudentsPerformance.csv" yang berisi informasi performa siswa dalam tiga mata pelajaran beserta beberapa atribut demografis.

- **Sumber Data**: [Kaggle: Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Analisis Awal**: Dataset terdiri dari 1000 baris dan 8 kolom. Pengecekan awal menunjukkan tidak ada nilai yang hilang (*missing values*) maupun data duplikat, sehingga data siap untuk tahap persiapan selanjutnya.

### Variabel-variabel pada Dataset
| Variabel | Deskripsi | Tipe Data |
| :--- | :--- | :--- |
| `gender` | Jenis kelamin siswa | Kategorikal |
| `race/ethnicity` | Kelompok ras/etnis siswa | Kategorikal |
| `parental level of education` | Tingkat pendidikan orang tua | Kategorikal |
| `lunch` | Jenis makan siang yang diterima siswa | Kategorikal |
| `test preparation course` | Status penyelesaian kursus persiapan ujian | Kategorikal |
| `math score` | Skor ujian matematika | Numerik |
| `reading score` | Skor ujian membaca | Numerik |
| `writing score` | Skor ujian menulis | Numerik |

## Data Preparation
Tahapan persiapan data dilakukan secara berurutan untuk memastikan data siap digunakan untuk pemodelan.

### 1. Feature Engineering
- **Proses**: Tahap pertama adalah membuat fitur baru untuk mendefinisikan variabel target. Fitur `average_score` dibuat dengan menghitung rata-rata dari tiga skor ujian. Kemudian, fitur target `performance` dibuat dari `average_score`, di mana nilai 1 menandakan performa baik (skor >= 70) dan 0 untuk performa kurang.
- **Alasan**: Pembuatan `average_score` bertujuan untuk mendapatkan representasi tunggal dari kinerja akademis siswa, yang kemudian digunakan untuk membuat target klasifikasi biner yang jelas.
    ```python
    # Membuat fitur rata-rata skor
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    # Membuat variabel target
    df['performance'] = (df['average_score'] >= 70).astype(int)
    ```

### 2. Label Encoding
- **Proses**: Fitur-fitur kategorikal seperti `gender`, `race/ethnicity`, dll., diubah menjadi representasi numerik menggunakan `LabelEncoder`.
- **Alasan**: Sebagian besar algoritma machine learning memerlukan input dalam bentuk numerik. Label Encoding adalah teknik sederhana untuk melakukan transformasi ini.
    ```python
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    ```

### 3. Pembagian Data (Train-Test Split)
- **Proses**: Dataset dibagi menjadi data latih (80%) dan data uji (20%).
- **Alasan**: Pembagian data ini penting untuk melatih model pada satu subset dan mengevaluasi kinerjanya pada subset lain yang belum pernah dilihatnya. Ini membantu mengukur kemampuan generalisasi model. `random_state=42` digunakan agar proses pembagian dapat direproduksi.
    ```python
    from sklearn.model_selection import train_test_split
    X = df[categorical_cols + ['math score', 'reading score', 'writing score', 'average_score']]
    y = df['performance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

## Modeling
Tahapan ini membahas tiga model machine learning yang digunakan untuk klasifikasi performa siswa.

### 1. Random Forest Classifier
- **Penjelasan**: Random Forest adalah metode *ensemble* yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk prediksi yang lebih akurat dan stabil.
- **Kelebihan**: Cenderung memiliki akurasi tinggi, tahan terhadap *overfitting*, dan dapat memberikan informasi *feature importance*.
- **Kekurangan**: Bisa menjadi "kotak hitam" (sulit diinterpretasi) dan membutuhkan lebih banyak sumber daya komputasi.
- **Proses Improvement**: Model dioptimalkan menggunakan *hyperparameter tuning* dengan `GridSearchCV`. Parameter terbaik yang ditemukan adalah `n_estimators=50`, `max_depth=None`, dan `min_samples_split=2`.

### 2. Logistic Regression
- **Penjelasan**: Model linear yang menggunakan fungsi logistik untuk memprediksi probabilitas suatu kelas.
- **Kelebihan**: Sederhana, cepat dilatih, dan hasilnya mudah diinterpretasi.
- **Kekurangan**: Kurang akurat untuk masalah non-linear yang kompleks.

### 3. Support Vector Machine (SVM)
- **Penjelasan**: Algoritma yang mencari *hyperplane* optimal untuk memisahkan data ke dalam kelas-kelas yang berbeda.
- **Kelebihan**: Efektif dalam ruang dimensi tinggi dan fleksibel dengan penggunaan *kernel*.
- **Kekurangan**: Kurang efisien untuk dataset besar dan pemilihan parameter *kernel* yang tepat bisa menjadi tantangan.

### Pemilihan Model Terbaik
Model terbaik dipilih berdasarkan **F1-Score** tertinggi pada data uji. F1-Score dipilih karena memberikan keseimbangan antara *precision* dan *recall*, yang penting mengingat adanya sedikit ketidakseimbangan kelas. Berdasarkan hasil evaluasi, **Random Forest Classifier (Tuned)** dipilih sebagai model terbaik.

## Evaluation
Metrik evaluasi yang digunakan untuk menilai kinerja model klasifikasi adalah:

- **Accuracy**: Rasio prediksi yang benar terhadap total prediksi.
  - *Formula*: $$ \frac{TP+TN}{TP+TN+FP+FN} $$
- **Precision**: Dari semua yang diprediksi sebagai kelas positif, berapa yang benar-benar positif.
  - *Formula*: $$ \frac{TP}{TP+FP} $$
- **Recall**: Dari semua yang sebenarnya positif, berapa yang berhasil diprediksi sebagai positif.
  - *Formula*: $$ \frac{TP}{TP+FN} $$
- **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Metrik ini baik digunakan untuk kasus kelas yang tidak seimbang.
  - *Formula*: $$ 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

### Hasil Proyek Berdasarkan Metrik Evaluasi
Berikut adalah ringkasan perbandingan kinerja model pada data uji, sesuai dengan eksekusi di notebook:

| Model | Accuracy | F1-Score |
| :--- | :------: | :------: |
| Random Forest (Baseline) | 1.0000 | 1.0000 |
| Random Forest (Tuned) | 1.0000 | 1.0000 |
| Logistic Regression | 0.9300 | 0.9231 |
| SVM | 0.9900 | 0.9888 |

### Analisis Hasil
- **Performa Model**: Random Forest Classifier (Baseline dan Tuned) mencapai performa sempurna, sementara SVM juga menunjukkan hasil yang sangat tinggi.
- **Analisis Kritis (Kesesuaian Fungsi)**: Performa sempurna (Akurasi dan F1-Score 1.0000) pada model Random Forest adalah indikasi kuat adanya **kebocoran data (data leakage)**. Hal ini terjadi karena fitur `average_score` dimasukkan sebagai prediktor (X) untuk target `performance`, padahal `performance` itu sendiri didefinisikan secara langsung dari `average_score`. Akibatnya, model tidak benar-benar "belajar", melainkan hanya menemukan hubungan langsung tersebut. Dalam skenario praktis, skor ini tidak realistis dan model ini tidak akan dapat digeneralisasi pada data baru.
- **Analisis Tambahan**: *Confusion matrix* dan visualisasi *feature importance* dari notebook juga mengonfirmasi adanya *data leakage*. `average_score` menjadi fitur dengan kontribusi paling signifikan, yang logis karena merupakan dasar dari variabel target.
