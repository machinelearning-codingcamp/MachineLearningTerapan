# Tabel Model Machine Learning

## Tahap Penyusunan Model

Semua model machine learning dalam proyek ini mengikuti tahap penyusunan yang sama, yaitu:
1. Pengumpulan dan preprocessing data
2. Memisahkan fitur (X) dan target (y)
3. Membagi data menjadi training dan testing
4. Inisialisasi model
5. Melatih model dengan metode fit()
6. Memprediksi nilai target pada data testing
7. Mengevaluasi model dengan metrik MSE, RMSE, dan R²

Yang membedakan masing-masing model adalah cara kerja algoritma di dalamnya, terutama pada saat proses pelatihan (fit) dan bagaimana model mempelajari pola dari data.

## Cara Kerja Model

| Model | Cara Kerja |
|-------|------------|
| Linear Regression | Mencari hubungan linear antara variabel independen (fitur) dan variabel dependen (target) dengan meminimalkan jumlah kuadrat dari selisih antara nilai aktual dan prediksi (residual). |
| LARS (Least Angle Regression) | Menggabungkan konsep forward selection dan regularisasi. Dimulai dari koefisien nol dan secara bertahap meningkatkan koefisien dalam arah yang memiliki korelasi tertinggi dengan residual saat ini. Algoritma secara iteratif memilih fitur yang paling berkorelasi dengan residual dan memperbarui koefisien secara bertahap hingga semua fitur terpilih atau kriteria berhenti tercapai. |
| Gradient Boosting Regressor | Membangun model secara sekuensial (biasanya decision tree), di mana setiap model baru mencoba memperbaiki kesalahan model sebelumnya. Mengoptimalkan fungsi loss dengan mengikuti arah gradien negatif. Proses dimulai dengan prediksi konstan, kemudian secara iteratif menghitung residual dari model saat ini, melatih weak learner baru untuk memprediksi residual tersebut, dan menambahkannya ke model dengan bobot tertentu (learning rate). Model final adalah kombinasi dari semua weak learner. |
| Random Forest Regressor | Metode ensemble yang terdiri dari banyak decision tree. Setiap tree dilatih pada subset acak dari data pelatihan (bootstrap sampling) dan menggunakan subset acak dari fitur. Pada setiap node, algoritma memilih subset acak dari fitur untuk mencari split terbaik dan memperluas tree hingga kriteria berhenti tercapai. Prediksi akhir adalah rata-rata dari prediksi semua tree. |
| Ridge Regression | Variasi dari Linear Regression yang menambahkan regularisasi L2. Meminimalkan jumlah kuadrat residual plus nilai alpha dikali jumlah kuadrat koefisien untuk mengurangi overfitting dan menangani multikolinearitas. Model mencari koefisien yang meminimalkan formula: RSS (Residual Sum of Squares) + alpha * (koefisien²). |

## Parameter Default Model

### Linear Regression
| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| fit_intercept | True | Menghitung nilai intercept dalam model |
| copy_X | True | Membuat salinan data X |
| n_jobs | None | Menggunakan 1 prosesor untuk komputasi |
| positive | False | Tidak membatasi koefisien untuk positif |

### LARS (Least Angle Regression)
| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| fit_intercept | True | Menghitung nilai intercept dalam model |
| verbose | False | Tidak menampilkan output tambahan saat fitting |
| normalize | True (deprecated) | Akan dihapus di versi mendatang |
| precompute | 'auto' | Menentukan otomatis apakah precompute bermanfaat |
| n_nonzero_coefs | 500 | Jumlah maksimum koefisien non-zero |
| eps | 2.22e-16 | Presisi mesin untuk stopping condition |
| copy_X | True | Membuat salinan data X |

### Gradient Boosting Regressor
| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| loss | 'squared_error' | Fungsi loss yang dioptimalkan |
| learning_rate | 0.1 | Tingkat pembelajaran untuk kontribusi setiap tree |
| n_estimators | 100 | Jumlah boosting stages (trees) |
| subsample | 1.0 | Fraksi sampel untuk fitting setiap tree |
| criterion | 'friedman_mse' | Fungsi untuk mengukur kualitas split |
| min_samples_split | 2 | Jumlah minimal sampel untuk internal node split |
| min_samples_leaf | 1 | Jumlah minimal sampel untuk leaf node |
| min_weight_fraction_leaf | 0.0 | Fraksi bobot minimal di leaf node |
| max_depth | 3 | Kedalaman maksimum setiap tree |
| min_impurity_decrease | 0.0 | Threshold untuk node split |
| max_features | None | Jumlah fitur untuk mencari split terbaik |
| alpha | 0.9 | Parameter untuk quantile loss dan huber loss |
| max_leaf_nodes | None | Jumlah maksimum leaf nodes |
| warm_start | False | Tidak menggunakan solusi sebelumnya |
| validation_fraction | 0.1 | Fraksi data training untuk validasi |
| n_iter_no_change | None | Iterasi tanpa peningkatan untuk early stopping |
| tol | 0.0001 | Toleransi untuk early stopping |
| ccp_alpha | 0.0 | Parameter kompleksitas untuk pruning |

### Random Forest Regressor
| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| n_estimators | 100 | Jumlah trees dalam forest |
| criterion | 'squared_error' | Fungsi untuk mengukur kualitas split |
| max_depth | None | Kedalaman maksimum tree (None = expand sampai leaf murni) |
| min_samples_split | 2 | Jumlah minimal sampel untuk internal node split |
| min_samples_leaf | 1 | Jumlah minimal sampel untuk leaf node |
| min_weight_fraction_leaf | 0.0 | Fraksi bobot minimal di leaf node |
| max_features | 1.0 atau 'auto' | Jumlah fitur untuk mencari split terbaik |
| max_leaf_nodes | None | Jumlah maksimum leaf nodes |
| min_impurity_decrease | 0.0 | Threshold untuk node split |
| bootstrap | True | Menggunakan bootstrap samples |
| oob_score | False | Tidak menggunakan out-of-bag samples untuk estimasi |
| n_jobs | None | Jumlah jobs untuk fitting dan prediksi (None = 1) |
| random_state | None | Seed untuk random number generator |
| verbose | 0 | Kontrol verbosity output |
| warm_start | False | Tidak menggunakan solusi sebelumnya |
| ccp_alpha | 0.0 | Parameter kompleksitas untuk pruning |

### Ridge Regression
| Parameter | Nilai Default | Keterangan |
|-----------|---------------|------------|
| alpha | 1.0 | Konstanta regularisasi yang menentukan kekuatan regularisasi |
| fit_intercept | True | Menghitung nilai intercept dalam model |
| normalize | False (deprecated) | Akan dihapus di versi mendatang |
| copy_X | True | Membuat salinan data X |
| max_iter | None | Jumlah maksimum iterasi untuk solver |
| tol | 0.001 | Toleransi untuk solusi |
| solver | 'auto' | Solver komputasi yang digunakan |
| positive | False | Koefisien tidak dibatasi positif |
| random_state | None | Seed untuk random number generator |