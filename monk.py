import pandas as pd
import numpy as np
from KNN import hold_out

def load_monk(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )
    y = df.iloc[:, 0].values.astype(np.float32)
    X = df.iloc[:, 1:7]
    return X, y

for monk_id in [1, 2, 3]:

    print(f"\n================ MONK-{monk_id} =================")

    # ================================
    # 1. Load TRAIN e TEST
    # ================================
    X_train, y_train = load_monk(f"monks-{monk_id}.train")
    X_test, y_test = load_monk(f"monks-{monk_id}.test")

    # ================================
    # 2. ONE-HOT encoding (fondamentale)
    # ================================
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all_oh = pd.get_dummies(X_all)

    X_train_oh = X_all_oh.iloc[:len(X_train)].values.astype(np.float32)
    X_test_oh = X_all_oh.iloc[len(X_train):].values.astype(np.float32)

    # ================================
    # 3. Hold-out per scegliere k
    # ================================
    k_values = [1,3,5,7,9,11,15,30,100,500]

    model, best_k = hold_out(
        X_train_oh,
        y_train,
        k_values,
        validation_split=0.2,
        is_regression=False
    )

    print("Best k:", best_k)

    # ================================
    # 4. Accuracy
    # ================================
    acc_train = np.mean(
        model.predict_classification(X_train_oh).flatten() == y_train
    )
    acc_test = np.mean(
        model.predict_classification(X_test_oh).flatten() == y_test
    )

    print("Accuracy TRAIN:", acc_train)
    print("Accuracy TEST:", acc_test)
