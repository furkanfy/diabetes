import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

df=pd.read_csv('C:/Users/ASUS/Documents/diabet/diabetes.csv')
print(df.head())
df.info()

df.describe()

print(df.isnull().sum())

df['Outcome'].value_counts()

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

df.fillna(df.median(), inplace=True)
df.head()

sns.countplot(x='Outcome', data=df)
plt.title('Diyabetli vs. Diyabetsiz Dağılımı')
plt.show()

df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.show()

sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('Outcome’a Göre Glucose Dağılımı')
plt.show()

from sklearn.model_selection import train_test_split

# X: bağımsız değişkenler, y: bağımlı değişken (Outcome)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Veriyi %80 eğitim - %20 test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Doğruluk oranı
print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))

# Detaylı performans raporu
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE

# SMOTE nesnesi oluştur
smote = SMOTE(random_state=42)

# Sadece eğitim verisi üzerinde SMOTE uygula
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model = LogisticRegression()
model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

# Performans ölç
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Modeli oluştur
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)  # SMOTE sonrası verilerle eğitiyoruz

# Test verisi ile tahmin
y_pred_rf = rf_model.predict(X_test)

# Performans metriklerini al
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Doğruluk Oranı:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Olasılıkları al
y_probs = rf_model.predict_proba(X_test)[:, 1]  # Sınıf 1 (diyabet) olasılıkları

# ROC değerleri
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# AUC skoru
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Skoru:", roc_auc)

# ROC grafiği
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc, color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi - Random Forest')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Değerlendirme
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Özellik isimlerini yeniden tanımla
feature_names = X.columns  # scaler'dan önce tanımlanmalı

# Özellik önemi grafiği
import matplotlib.pyplot as plt
import seaborn as sns
importance = xgb.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=feature_names)
plt.title('Feature Importance - XGBoost')
plt.tight_layout()
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='logloss'))
])

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=3)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
print("En iyi skor:", grid.best_score_)
import joblib

# Eğitilmiş GridSearchCV pipeline'ı kaydet
joblib.dump(grid.best_estimator_, 'xgb_pipeline.pkl')