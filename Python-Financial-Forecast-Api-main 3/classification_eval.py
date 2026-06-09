# classification_eval.py
# Amaç: logistic regression modelini sınıflandırma metrikleriyle değerlendirmek (UP vs DOWN)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from features_layer import build_feature_dataframe

# ==========================
# 1) AYARLAR
# ==========================

TICKER = "TSLA"   # İstersen "AAPL", "MSFT" vb. yapıp tekrar çalıştırabilirsin
PERIOD = "2y"

# ==========================
# 2) VERİYİ build_feature_dataframe İLE AL
# ==========================

df, X, y = build_feature_dataframe(TICKER, period=PERIOD)

print("=== Kolonlar ===")
print(list(df.columns))
print("================")

print(f"Toplam örnek sayısı: {len(df)}")
if "Date" in df.columns:
    print(f"Tarih aralığı: {df['Date'].iloc[0]}  -->  {df['Date'].iloc[-1]}")

print(f"Pozitif sınıf (UP=1) oranı: {y.mean():.2f}")

# X ve y zaten features_layer içinde seçildi
X = X.values
y = y.values

# ==========================
# 3) Train / Test ayrımı
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,   # zaman serisini bozma
)

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# ==========================
# 4) Ölçekleme + Model
# ==========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ==========================
# 5) Metrikler
# ==========================

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Baseline: "her zaman UP" dersek
baseline_pred = [1] * len(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("\n=== SINIFLANDIRMA SONUÇLARI ===")
print(f"TICKER : {TICKER}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Baseline (her zaman UP): {baseline_acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (rows = gerçek, cols = tahmin)")
print(cm)
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
