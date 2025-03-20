import pandas as pd
import os 

import warnings

# 모든 경고를 무시하도록 설정
warnings.filterwarnings('ignore')

forecast = pd.read_csv(os.path.join("Data", "forecast.csv"), index_col=0)
weather = pd.read_csv(os.path.join("Data", "weather.csv"), index_col=0)

pv    = pd.read_csv(os.path.join("Data", "pv_day_merged.csv"), index_col=0)
excol = [col for col in pv.columns if "시간당발전량" not in col]
pv    = pv.drop(columns=excol)

# time_steps = 14


# pv = pv.loc[[idx for idx in pv.index if (int(idx[11:13]) >=7 and int(idx[11:13]) <= 20)]]
# pv_e = pv[:-14]
# # pv_e  = pv[:-24]
# test    = pv.iloc[-14:]
pv.to_csv("pv.csv")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from sklearn.preprocessing import MinMaxScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

file_path = 'pv.csv'
data = pd.read_csv(file_path)
df = data.iloc[:-24]
test = data.iloc[-24:]
df['시간'] = pd.to_datetime(df['시간'])
# df.sort_values('시간', inplace=True)

# 시간 기반 피처 추가
df['hour'] = df['시간'].dt.hour / 24.0
features = df.columns.difference(['시간'])
data = df[features].values

# 데이터 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 24  # 하루(24시간) 단위 시퀀스
X, y = create_sequences(data_scaled, seq_length)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, shuffle=False)


# 11~13시 가중치 ++
def time_based_weight(data, hour_index, penalty=2.0):
    weights = np.ones(len(data))
    for i in range(len(data)):
        hour = int(data[i, hour_index] * 24)
        if 11 <= hour <= 13:
            weights[i] *= penalty
    return weights

hour_index = features.get_loc('hour')
weights = time_based_weight(X_train[:, -1, :], hour_index)


# LSTM 모델
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(64),
    layers.Dense(y_train.shape[1])
])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.compile(optimizer='adam', loss='mse')
model.summary()

# 모델 학습
history = model.fit(X_train, y_train, 
                    epochs=500, batch_size=32, validation_split=0.2,
                    sample_weight=weights, verbose=0, callbacks=[early_stopping])

# model.save("lstm.h5")

# 예측 함수
def predict_future(model, data, steps):
    predictions = []
    current_input = data[-1:].copy()
    for _ in range(steps):
        pred = model.predict(current_input)  # (1, 건물 수) 형태 반환
        predictions.append(pred[0])
        current_input = np.roll(current_input, -1, axis=1)  # 시퀀스 슬라이딩
        current_input[0, -1, :] = pred  # 새로운 예측 추가
    return np.array(predictions)  # (24, 건물 수) 형태 반환
        

# 미래 24시간 예측
future_steps = 24
future_predictions = predict_future(model, X_test, future_steps)
future_predictions_inv = scaler.inverse_transform(future_predictions)


# list(np.sum(future_predictions_inv, axis=1))

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

test_drop = test.drop('시간', axis=1)
y = test_drop.apply(sum, axis = 1).values.tolist()
result = list(np.sum(future_predictions_inv, axis=1))

t = [i for i in range(24)]

plt.xlabel('t')
plt.ylabel('pv')
plt.xticks(ticks=t, labels=t)

plt.bar(t,y, label="Actual PV")
plt.bar(t,result, label="Pred. PV")
plt.legend()
plt.show()

plt.savefig("lstm_seq,png")

mse = mean_squared_error(y, result)
print(mse)