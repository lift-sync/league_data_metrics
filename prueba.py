import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras import Model

# --- Carga dataset ---
rows = []
with open('salida.txt', encoding='latin-1') as text:
    for line in text:
        pwd = line.strip()
        hash_hex = hashlib.md5(pwd.encode('utf-8', errors='ignore')).hexdigest()
        rows.append({'key': pwd, 'hash': hash_hex})

df = pd.DataFrame(rows).sample(frac=0.1, random_state=42).reset_index(drop=True)

X = df["key"].astype(str).values          # contraseñas en texto
y_str = df["hash"].astype(str).values     # hashes en texto

X_train, X_test, y_train_str, y_test_str = train_test_split(
    X, y_str, test_size=0.2, random_state=42
)

# --- y a enteros (requisito para sparse_categorical_crossentropy) ---
le = LabelEncoder()
y_train = le.fit_transform(y_train_str)
y_test  = le.transform(y_test_str)
num_classes = len(le.classes_)

# --- Vectorización char-level dentro del modelo ---
max_tokens = 128
seq_len = 32

vec = TextVectorization(
    standardize=None,
    split='character',
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=seq_len
)
vec.adapt(X_train)  # ¡solo con train!

# --- Modelo (LSTM) ---
inputs = Input(shape=(1,), dtype=tf.string)
x = vec(inputs)                              # tf.string -> ints
x = Embedding(input_dim=max_tokens, output_dim=32)(x)
x = LSTM(64)(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256, verbose=1)
model.evaluate(X_test, y_test, verbose=1)