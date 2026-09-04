import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import time

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================
# LOAD
# ============================

X = pd.read_csv(
    "fused_features2.csv"
).values.astype(
    np.float32
)

y = np.load(
    "labels2.npy"
)

print(
    "\nLoaded Features:",
    X.shape
)

# ============================
# SCALER
# ============================

scaler = joblib.load(
    "multimodal_scaler.pkl"
)

X = scaler.transform(
    X
)

# ============================
# LABEL ENCODER
# ============================

enc = joblib.load(
    "multimodal_label_encoder.pkl"
)

y = enc.transform(
    y
)

# ============================
# MODEL
# ============================

model = tf.keras.models.load_model(

    "multimodal_final_model_2.h5",

    compile=False

)

print(
    "Model Loaded"
)

# ============================
# BATCH INFERENCE
# ============================

start = time.time()

prob = model.predict(

    X,

    batch_size=128,

    verbose=1

)

end = time.time()

pred = np.argmax(
    prob,
    axis=1
)

# ============================
# METRICS
# ============================

acc = accuracy_score(
    y,
    pred
)

print(
    "\nAccuracy:",
    round(
        acc*100,
        2
    ),
    "%"
)

print(
    "\nClassification Report\n"
)

report = classification_report(
    y,
    pred
)

print(
    report
)

cm = confusion_matrix(
    y,
    pred
)

print(
    "\nConfusion Matrix\n"
)

print(
    cm
)

# ============================
# SAVE RESULTS
# ============================

pd.DataFrame({

    "Actual":
    enc.inverse_transform(
        y
    ),

    "Predicted":
    enc.inverse_transform(
        pred
    )

}).to_csv(

    "predictions.csv",

    index=False

)

pd.DataFrame(
    cm
).to_csv(

    "confusion_matrix.csv",

    index=False

)

with open(
    "classification_report.txt",
    "w"
) as f:

    f.write(
        report
    )

# ============================
# TIME
# ============================

total = end-start

avg = total/len(X)

print(
    "\nTotal Inference:",
    round(
        total,
        3
    ),
    "sec"
)

print(
    "Average Per Sample:",
    round(
        avg*1000,
        4
    ),
    "ms"
)

print(
    "\nTotal Samples:",
    len(X)
)

print(
    "\nGenerated:"
)

print(
    "predictions.csv"
)

print(
    "confusion_matrix.csv"
)

print(
    "classification_report.txt"
)
