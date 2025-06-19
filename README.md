 🩺 Diabetes Prediction with ML

Bu projede, bireylerin diyabet hastası olup olmadığını tahmin etmek amacıyla çeşitli makine öğrenmesi modelleri kullanılmıştır.

## 🔍 Özellikler

- Veri temizleme ve ön işleme
- Eksik değer doldurma (0 → NaN → median)
- Görselleştirme: dağılım, korelasyon, boxplot
- SMOTE ile dengesiz veriyi dengeleme
- Logistic Regression, Random Forest, XGBoost
- GridSearchCV ile hiperparametre optimizasyonu
- ROC AUC eğrisi ve feature importance
- Modelin `.pkl` dosyası olarak kaydedilmesi

## 📊 En İyi Model

- Model: XGBoost (GridSearch ile optimize edildi)
- Skor: F1 ≈ 0.85+
- ROC AUC ≈ 0.88+
