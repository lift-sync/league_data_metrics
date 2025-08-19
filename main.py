import hashlib
import pandas as pd
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

rows = []
with open('salida.txt', encoding='latin-1') as text:
    for line in text:
        password = line.strip()
        hash = hashlib.md5(password.encode()).hexdigest()
        dict = {
            'key': password,
            'hash': hash
        }
        rows.append(dict)


df = pd.DataFrame(rows)

df = df.sample(frac=0.1, random_state=42)

X = df["key"].astype(str).values
y = df["hash"].astype(str).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train)
print(y_train)

n=8
df2 = pd.DataFrame({
    "idx": np.arange(n),
    "ones": np.ones(n, dtype=int),
    "pwd": X_train[:n]
})
print(df2)               # tabla limpia
X_train2 = df2.to_numpy()

print(X_train2)

model = Sequential([
    LSTM(10),
    LSTM(20)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train2, y_train, batch_size=10, epochs=20)

model.evaluate(X_test, y_test, verbose=1)
