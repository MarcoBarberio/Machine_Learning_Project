import pandas as pd
import numpy as np
from KNN import hold_out   # importa il tuo modulo
from loss import mee_np

# Carica CUP
df = pd.read_csv("ML-CUP25-TR.csv", header=None,comment="#")
X = df.iloc[:, 1:13].values.astype(np.float32)
Y = df.iloc[:, 13:17].values.astype(np.float32)
print(Y)
# Normalizzazione
mean = X.mean(axis=0, keepdims=True)
std = X.std(axis=0, keepdims=True) + 1e-8
Xn = (X - mean) / std

# Test hold-out
k_values = [1,3,5,7,9,11,15,30,100,500]

model, best_k = hold_out(Xn, Y, k_values, validation_split=0.2, is_regression=True)

print("\nMIGLIOR k:", best_k)

# valutazione su tutto il training
y_pred = model.predict_regression(Xn)
error = mee_np(Y, y_pred)
print("Errore MEE sul training:", error)

df_test = pd.read_csv("ML-CUP25-TS.csv", header=None, comment="#")
ids_test = df_test.iloc[:, 0].values.astype(int)
X_test = df_test.iloc[:, 1:13].values.astype(np.float32)

# Normalizza con mean/std del training
X_test_n = (X_test - mean) / std

# ================================
# 5. Predizione sul test set
# ================================
y_pred_test = model.predict_regression(X_test_n)

# Stampa alcune predizioni
print("\nPrime 5 predizioni sul test set:")
for i in range(5):
    print(f"ID={ids_test[i]} → {y_pred_test[i]}")

# ================================
# 6. (OPZIONALE) Salva le predizioni su file CSV
#    UTILE per la consegna CUP
# ================================
output_path = "KNN_ML-CUP25-TS_PRED.csv"

with open(output_path, "w") as f:
    f.write("# KNN baseline\n")
    f.write("# ML-CUP25 test predictions\n")
    f.write("#\n")
    f.write("# id, o1, o2, o3, o4\n")
    for idx, row in zip(ids_test, y_pred_test):
        o1, o2, o3, o4 = row
        f.write(f"{idx},{o1:.6f},{o2:.6f},{o3:.6f},{o4:.6f}\n")

print(f"\nFile salvato: {output_path}")