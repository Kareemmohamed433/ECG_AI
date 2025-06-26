import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# تحميل البيانات
normal_df = pd.read_csv("data/ptbdb_normal.csv", header=None)
abnormal_df = pd.read_csv("data/ptbdb_abnormal.csv", header=None)

# أخذ 5 عينات عشوائية من كل نوع
normal_samples = normal_df.sample(5, random_state=42)
abnormal_samples = abnormal_df.sample(5, random_state=42)

# دمج العينات
samples = pd.concat([normal_samples, abnormal_samples])
X = samples.iloc[:, :-1].values
y_true = samples.iloc[:, -1].values

# تحويل التسميات إلى كلمات
y_labels = np.where(y_true == 1, 'abnormal', 'normal')

# تجهيز البيانات للنموذج
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# تحميل النموذج
model = load_model("best_model.keras")

# التنبؤ
y_pred_prob = model.predict(X_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# فك ترميز التسميات
label_encoder = LabelEncoder()
label_encoder.fit(['normal', 'abnormal'])  # نفس ما استخدمته في التدريب
y_true_encoded = label_encoder.transform(y_labels)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# المقارنة و الطباعة
for i in range(len(X)):
    print(f"✅ إشارة رقم {i+1}")
    print(f"  ➤ التسمية الفعلية  : {y_labels[i]}")
    print(f"  ➤ التسمية المتوقعة: {y_pred_labels[i]}")
    print("-" * 40)

    # رسم الإشارة
    plt.figure(figsize=(10, 2))
    plt.plot(X[i], label='ECG Signal')
    plt.title(f"Actual: {y_labels[i]}  |  Predicted: {y_pred_labels[i]}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
