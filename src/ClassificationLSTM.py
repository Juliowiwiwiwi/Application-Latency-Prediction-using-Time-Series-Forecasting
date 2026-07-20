import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


df = pd.read_csv('DataSets/latency_data_v2.csv')


df['users_change'] = df['active_users'].diff().fillna(0)
df['error_change'] = df['error_rate'].diff().fillna(0)
df['memory_change'] = df['memory_usage'].diff().fillna(0)
df['requests_change'] = df['request_count'].diff().fillna(0)


feature_cols = [
    'request_count', 'requests_change', 
    'active_users', 'users_change', 
    'cpu_usage', 'memory_usage', 'memory_change',
    'p99_latency_ms',
    'error_rate', 'error_change'
]

#(1 if > 275ms, 0 otherwise)
THRESHOLD = 275
df['target_spike'] = (df['p99_latency_ms'] > THRESHOLD).astype(int)


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

target_data = df['target_spike'].values

#Sliding Windows(30Min Forecast)
def create_sequences_clf(data, target, lookback_steps=12, delay_steps=6):
    X, y = [], []
    for i in range(len(data) - lookback_steps - delay_steps + 1):
        X.append(data[i : i + lookback_steps])
        y.append(target[i + lookback_steps + delay_steps - 1])
    return np.array(X), np.array(y)

X, y = create_sequences_clf(scaled_features, target_data, lookback_steps=12, delay_steps=6)

total_samples = len(X)
train_end = int(total_samples * 0.70)
cv_end = int(total_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_cv, y_cv = X[train_end:cv_end], y[train_end:cv_end]
X_test, y_test = X[cv_end:], y[cv_end:]

clf_model = Sequential()
clf_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
clf_model.add(LSTM(units=64, return_sequences=False))
clf_model.add(Dropout(rate=0.2))
clf_model.add(Dense(1, activation='sigmoid'))

clf_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])


print("Training the 30Min Classification Alert System")
history = clf_model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_cv, y_cv),
    callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
    batch_size=32,
    verbose=1
)

predictions_prob = clf_model.predict(X_test)
predicted_classes = (predictions_prob > 0.35).astype(int).flatten()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Spike'], 
            yticklabels=['Actual Normal', 'Actual Spike'])
plt.title('30-Minute Alert System Accuracy')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=['Normal', 'Spike (>275ms)']))