# Spam Detection — Yelp Review

Sistem deteksi spam pada ulasan Yelp menggunakan Machine Learning berbasis teks.

**Revalina Fidiya Anugrah-H1D023011**

---

## Dataset

- **Sumber:** [Kaggle — Yelp Review with Sentiments and Features](https://www.kaggle.com/datasets/naveedhn/yelp-review-with-sentiments-and-features/data)
- **Jumlah data:** ~355.000 ulasan restoran dan hotel
- **Label:** `1` = Spam, `0` = Not Spam
- **Lisensi:** CC BY 4.0
- **Fitur yang digunakan:** kolom `Review` (teks) sebagai input, `Spam(1) and Not Spam(0)` sebagai label

---

## Instalasi

```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib openpyxl
```

---

## Cara Menjalankan

1. Letakkan `dataset_yelp.xlsx` di path yang sesuai dalam `reviewyelp.py`
2. Jalankan script:

```bash
python reviewyelp.py
```

Output yang dihasilkan:
- Visualisasi distribusi label, panjang teks, dan confusion matrix
- `dataset_clean.csv` — data hasil preprocessing
- `model.pkl` dan `vectorizer.pkl` — model siap pakai

---

## Pipeline Machine Learning

```
Raw Data → Preprocessing → TF-IDF → SMOTE → Model → Evaluasi
```

| Tahap | Detail |
|---|---|
| Preprocessing | Lowercase, hapus angka & tanda baca, hapus URL, strip spasi |
| Feature Engineering | TF-IDF (max 10.000 fitur, bigram, min_df=5, max_df=0.9) |
| Class Imbalance | SMOTE (sampling_strategy=0.8) |
| Split | 80:20 dengan stratify |
| Model 1 | Multinomial Naive Bayes |
| Model 2 | Logistic Regression (C=1.5, class_weight='balanced') |

---

## Hasil Evaluasi

| Model | Accuracy | Recall (Spam) | F1 (Spam) | ROC AUC |
|---|---|---|---|---|
| Naive Bayes | 77.20% | 0.37 | 0.25 | — |
| Logistic Regression | 76.56% | 0.39 | 0.25 | 0.6717 |

> Accuracy yang lebih rendah dari versi sebelumnya (89%) adalah hal yang **disengaja** — model kini tidak lagi hanya memprediksi kelas mayoritas, melainkan benar-benar mendeteksi spam.

---

## Rencana Pengembangan

- [ ] Antarmuka web menggunakan **Flask**
- [ ] Input teks real-time dan tampilkan hasil prediksi + confidence score
- [ ] Visualisasi distribusi prediksi dengan Chart.js

---

## Referensi

Hussain, N., Mirza, H. T., Iqbal, F., Hussain, I., & Kaleem, M. (2020). *Detecting Spam Product Reviews in Roman Urdu Script.* The Computer Journal.
