"""
ML Project - Type B (Compare models within TensorFlow)

Modelli:
- Neural Network (NN)
- Linear Regression
- SVM lineare (per MONK, classificazione)
- k-NN (implementato con TensorFlow)

Dataset:
- ML-CUP25-TR.csv (training + validation + internal test)
- ML-CUP25-TS.csv (blind test, senza target)
- MONK (file da specificare)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import itertools  # <-- AGGIUNTO PER GRID SEARCH

# ============================================================
# Utility comuni
# ============================================================

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def mee(y_true, y_pred):
    diff = y_true - y_pred
    eucl = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1))
    return tf.reduce_mean(eucl)


def mee_np(y_true, y_pred):
    diff = y_true - y_pred
    eucl = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(eucl)


# ============================================================
# 1) CUP: caricamento e split TR/VL/TS
# ============================================================

def load_cup_train(path="ML-CUP25-TR.csv"):
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].values
    X = df.iloc[:, 1:13].values.astype(np.float32)
    y = df.iloc[:, 13:17].values.astype(np.float32)
    return ids, X, y


def load_cup_test(path="ML-CUP25-TS.csv"):
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].values
    X = df.iloc[:, 1:13].values.astype(np.float32)
    return ids, X


def split_train_val_test(X, y, tr_ratio=0.6, val_ratio=0.2):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    n_tr = int(tr_ratio * n)
    n_val = int(val_ratio * n)
    n_ts = n - n_tr - n_val

    X_tr = X[:n_tr]
    y_tr = y[:n_tr]

    X_val = X[n_tr:n_tr + n_val]
    y_val = y[n_tr:n_tr + n_val]

    X_ts = X[n_tr + n_val:]
    y_ts = y[n_tr + n_val:]

    return (X_tr, y_tr), (X_val, y_val), (X_ts, y_ts)


# ============================================================
# 2) Modelli in TensorFlow
# ============================================================

def build_nn(input_dim, output_dim, hidden_units=(64, 32), l2=1e-4, lr=1e-3):
    model = tf.keras.Sequential()
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)
        ))
    model.add(tf.keras.layers.Dense(output_dim, activation='linear'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse'
    )
    return model


def build_linear_regression(input_dim, output_dim, lr=1e-2, optimizer_name="Adam"):
    if optimizer_name == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_name == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(output_dim, activation='linear')
    ])

    model.compile(
        optimizer=opt,
        loss='mse'
    )
    return model


def build_svm_classifier(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=(input_dim,))
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Hinge()
    )
    return model


# ============================================================
# 3) k-NN in TensorFlow (regressione)
# ============================================================

def knn_predict_regression_tf(X_train, y_train, X_test, k=5):
    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

    distances = tf.norm(
        tf.expand_dims(X_test_tf, 1) - tf.expand_dims(X_train_tf, 0),
        axis=2
    )
    knn_idx = tf.argsort(distances, axis=1)[:, :k]
    knn_y = tf.gather(y_train_tf, knn_idx)
    return tf.reduce_mean(knn_y, axis=1).numpy()


# ============================================================
# 4) Training helper (CUP)
# ============================================================

def train_model_cup(model, X_tr, y_tr, X_val, y_val, epochs=500, batch_size=32):
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[cb]
    )
    return model, history


def evaluate_cup_model(name, model, X_tr, y_tr, X_val, y_val, X_ts, y_ts):
    y_tr_pred = model.predict(X_tr, verbose=0)
    y_val_pred = model.predict(X_val, verbose=0)
    y_ts_pred = model.predict(X_ts, verbose=0)

    mee_tr = mee_np(y_tr, y_tr_pred)
    mee_val = mee_np(y_val, y_val_pred)
    mee_ts = mee_np(y_ts, y_ts_pred)

    print(f"\n=== {name} (CUP) ===")
    print(f"MEE TR: {mee_tr:.4f}")
    print(f"MEE VL: {mee_val:.4f}")
    print(f"MEE TS: {mee_ts:.4f}")

    return {
        "name": name,
        "mee_tr": mee_tr,
        "mee_val": mee_val,
        "mee_ts": mee_ts
    }


# ============================================================
# 5) CUP: pipeline completa (versione originale)
# ============================================================

def run_cup_experiments():
    print(">>> Caricamento CUP...")
    ids_tr, X, y = load_cup_train("ML-CUP25-TR.csv")
    (X_tr, y_tr), (X_val, y_val), (X_ts, y_ts) = split_train_val_test(X, y)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_val_n = (X_val - mean) / std
    X_ts_n = (X_ts - mean) / std

    results = []

    nn_model = build_nn(input_dim, output_dim, hidden_units=(64, 32), l2=1e-4)
    nn_model, _ = train_model_cup(nn_model, X_tr_n, y_tr, X_val_n, y_val)
    results.append(evaluate_cup_model("Neural Network", nn_model,
                                      X_tr_n, y_tr, X_val_n, y_val, X_ts_n, y_ts))

    lin_model = build_linear_regression(input_dim, output_dim)
    lin_model, _ = train_model_cup(lin_model, X_tr_n, y_tr, X_val_n, y_val)
    results.append(evaluate_cup_model("Linear Regression", lin_model,
                                      X_tr_n, y_tr, X_val_n, y_val, X_ts_n, y_ts))

    y_tr_k = knn_predict_regression_tf(X_tr_n, y_tr, X_tr_n, k=5)
    y_val_k = knn_predict_regression_tf(X_tr_n, y_tr, X_val_n, k=5)
    y_ts_k = knn_predict_regression_tf(X_tr_n, y_tr, X_ts_n, k=5)

    results.append({
        "name": "k-NN",
        "mee_tr": mee_np(y_tr, y_tr_k),
        "mee_val": mee_np(y_val, y_val_k),
        "mee_ts": mee_np(y_ts, y_ts_k)
    })

    print("\n>>> RISULTATI BASE (SENZA GRID SEARCH):")
    for r in results:
        print(r)

    return {
        "results": results,
        "mean": mean,
        "std": std
    }

# ============================================================
# 6) GRID SEARCH COMPLETA (154 configurazioni)
# ============================================================

def grid_search_cup(X_tr_n, y_tr, X_val_n, y_val, input_dim, output_dim):

    results = []

    # ----------------------------------------------
    # 1) GRID NEURAL NETWORK (135 configs)
    # ----------------------------------------------
    hidden_list = [(16,), (32,), (64,), (64, 32), (128, 64)]
    lr_list = [1e-2, 1e-3, 1e-4]
    l2_list = [0, 1e-4, 1e-3]
    batch_list = [16, 32, 64]

    nn_configs = list(itertools.product(hidden_list, lr_list, l2_list, batch_list))
    print(f">>> Grid Search NN configs: {len(nn_configs)}")

    for hidden, lr, l2, batch in nn_configs:

        model = build_nn(input_dim, output_dim, hidden_units=hidden, l2=l2, lr=lr)

        cb = tf.keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor="val_loss"
        )

        model.fit(
            X_tr_n, y_tr,
            validation_data=(X_val_n, y_val),
            epochs=300,
            batch_size=batch,
            verbose=0,
            callbacks=[cb]
        )

        y_tr_pred = model.predict(X_tr_n, verbose=0)
        y_val_pred = model.predict(X_val_n, verbose=0)

        results.append({
            "model": "NN",
            "hidden": hidden,
            "lr": lr,
            "l2": l2,
            "batch": batch,
            "mee_tr": mee_np(y_tr, y_tr_pred),
            "mee_val": mee_np(y_val, y_val_pred)
        })

    # ----------------------------------------------
    # 2) GRID LINEAR REGRESSION (12 configs)
    # ----------------------------------------------
    lr_list_lin = [1e-1, 1e-2, 1e-3, 1e-4]
    opt_list = ["SGD", "Adam", "RMSprop"]

    lin_configs = list(itertools.product(lr_list_lin, opt_list))
    print(f">>> Grid Search Linear configs: {len(lin_configs)}")

    for lr, opt in lin_configs:

        model = build_linear_regression(input_dim, output_dim, lr=lr, optimizer_name=opt)

        cb = tf.keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor="val_loss"
        )

        model.fit(
            X_tr_n, y_tr,
            validation_data=(X_val_n, y_val),
            epochs=300,
            batch_size=32,
            verbose=0,
            callbacks=[cb]
        )

        y_tr_pred = model.predict(X_tr_n, verbose=0)
        y_val_pred = model.predict(X_val_n, verbose=0)

        results.append({
            "model": "Linear",
            "lr": lr,
            "optimizer": opt,
            "mee_tr": mee_np(y_tr, y_tr_pred),
            "mee_val": mee_np(y_val, y_val_pred)
        })

    # ----------------------------------------------
    # 3) GRID k-NN (7 configs)
    # ----------------------------------------------
    k_list = [1, 3, 5, 7, 9, 11, 15]
    print(f">>> Grid Search k-NN configs: {len(k_list)}")

    for k in k_list:
        y_tr_pred = knn_predict_regression_tf(X_tr_n, y_tr, X_tr_n, k)
        y_val_pred = knn_predict_regression_tf(X_tr_n, y_tr, X_val_n, k)

        results.append({
            "model": "kNN",
            "k": k,
            "mee_tr": mee_np(y_tr, y_tr_pred),
            "mee_val": mee_np(y_val, y_val_pred)
        })

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("mee_val")

    print("\n>>> TOP 10 CONFIGURAZIONI:")
    print(df_sorted.head(10))

    return df_sorted


# ============================================================
# 7) CUP: ESECUZIONE GRID SEARCH + NORMALIZZAZIONE
# ============================================================

def run_cup_grid_search():
    print(">>> Avvio GRID SEARCH sulla CUP...")

    ids_tr, X, y = load_cup_train("ML-CUP25-TR.csv")
    (X_tr, y_tr), (X_val, y_val), (X_ts, y_ts) = split_train_val_test(X, y)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-8

    X_tr_n = (X_tr - mean) / std
    X_val_n = (X_val - mean) / std
    X_ts_n = (X_ts - mean) / std

    df_results = grid_search_cup(X_tr_n, y_tr, X_val_n, y_val, input_dim, output_dim)
    best = df_results.iloc[0]

    print("\n>>> MIGLIOR MODELLO (VALIDATION):")
    print(best)

    return df_results, best, mean, std, (X_tr_n, y_tr, X_val_n, y_val, X_ts_n, y_ts)


# ============================================================
# 8) MONK (rimasto invariato)
# ============================================================

def load_monk_dataset(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    y_tr = train_df.iloc[:, 0].values.astype(np.float32)
    X_tr = train_df.iloc[:, 1:].values.astype(np.float32)

    y_ts = test_df.iloc[:, 0].values.astype(np.float32)
    X_ts = test_df.iloc[:, 1:].values.astype(np.float32)

    y_tr_svm = 2 * y_tr - 1
    y_ts_svm = 2 * y_ts - 1

    return (X_tr, y_tr, y_tr_svm), (X_ts, y_ts, y_ts_svm)


def run_monk_experiment(train_path, test_path, monk_name="MONK-1"):

    print(f"\n>>> MONK experiment: {monk_name}")

    (X_tr, y_tr, y_tr_svm), (X_ts, y_ts, y_ts_svm) = load_monk_dataset(train_path, test_path)

    input_dim = X_tr.shape[1]

    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True) + 1e-8

    X_tr_n = (X_tr - mean) / std
    X_ts_n = (X_ts - mean) / std

    # NN
    nn = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_tr_n, y_tr, epochs=300, batch_size=16, verbose=0)
    _, acc_tr = nn.evaluate(X_tr_n, y_tr, verbose=0)
    _, acc_ts = nn.evaluate(X_ts_n, y_ts, verbose=0)
    print(f"NN - acc TR: {acc_tr:.4f}, acc TS: {acc_ts:.4f}")

    # Linear
    lin = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(input_dim,))
    ])
    lin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lin.fit(X_tr_n, y_tr, epochs=300, batch_size=16, verbose=0)
    _, acc_tr_lin = lin.evaluate(X_tr_n, y_tr, verbose=0)
    _, acc_ts_lin = lin.evaluate(X_ts_n, y_ts, verbose=0)
    print(f"Linear - acc TR: {acc_tr_lin:.4f}, acc TS: {acc_ts_lin:.4f}")

    # SVM
    svm = build_svm_classifier(input_dim)
    svm.fit(X_tr_n, y_tr_svm, epochs=300, batch_size=16, verbose=0)
    y_tr_svm_pred = np.where(svm.predict(X_tr_n) >= 0, 1, -1)
    y_ts_svm_pred = np.where(svm.predict(X_ts_n) >= 0, 1, -1)
    print(f"SVM - acc TR: {np.mean(y_tr_svm_pred == y_tr_svm):.4f}, acc TS: {np.mean(y_ts_svm_pred == y_ts_svm):.4f}")

    # kNN
    def knn_predict_class_tf(X_train, y_train, X_test, k=5):
        X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        distances = tf.norm(tf.expand_dims(X_test_tf, 1) - tf.expand_dims(X_train_tf, 0), axis=2)
        idx = tf.argsort(distances, axis=1)[:, :k]
        knn_y = tf.gather(y_train_tf, idx)
        return (tf.reduce_mean(knn_y, axis=1).numpy() >= 0.5).astype(int)

    y_tr_knn = knn_predict_class_tf(X_tr_n, y_tr, X_tr_n, 5)
    y_ts_knn = knn_predict_class_tf(X_tr_n, y_tr, X_ts_n, 5)
    print(f"k-NN - acc TR: {np.mean(y_tr_knn == y_tr):.4f}, acc TS: {np.mean(y_ts_knn == y_ts):.4f}")

# ============================================================
# 9) CUP: addestramento finale e file per il BLIND TEST
# ============================================================

def train_final_and_save_cup_predictions(model_type="nn",
                                         team_name="TEAMNAME",
                                         cup_tr_path="ML-CUP25-TR.csv",
                                         cup_ts_path="ML-CUP25-TS.csv"):
    """
    - Allena il modello finale su TUTTO ML-CUP25-TR (TR+VL+TS interni)
    - Genera il file team-name_ML-CUP25-TS.csv nel formato richiesto
      (4 righe di commento + 1000 righe: id,o1,o2,o3,o4)
    """
    print(">>> Addestramento FINALE CUP per blind test...")

    # Carico tutti i dati TR
    _, X_all, y_all = load_cup_train(cup_tr_path)
    ids_test, X_test = load_cup_test(cup_ts_path)

    # Normalizzazione su TUTTI i dati di training
    mean = X_all.mean(axis=0, keepdims=True)
    std = X_all.std(axis=0, keepdims=True) + 1e-8
    X_all_n = (X_all - mean) / std
    X_test_n = (X_test - mean) / std

    input_dim = X_all.shape[1]
    output_dim = y_all.shape[1]

    # Scelta modello in base a model_type
    if model_type == "nn":
        # Usa una configurazione "buona" (es. da grid search)
        model = build_nn(input_dim, output_dim,
                         hidden_units=(64, 32),
                         l2=1e-4,
                         lr=1e-3)
    elif model_type == "linear":
        model = build_linear_regression(input_dim, output_dim,
                                        lr=1e-2,
                                        optimizer_name="Adam")
    else:
        raise ValueError("model_type deve essere 'nn' o 'linear'.")

    cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=30,
        restore_best_weights=True
    )

    model.fit(
        X_all_n, y_all,
        epochs=1000,
        batch_size=32,
        verbose=0,
        callbacks=[cb]
    )

    # Predizioni sul blind test
    y_pred_test = model.predict(X_test_n, verbose=0)

    # Salvataggio file CSV nel formato richiesto
    out_path = f"{team_name}_ML-CUP25-TS.csv"
    with open(out_path, "w") as f:
        f.write(f"# {team_name} members\n")
        f.write(f"# {team_name}\n")
        f.write("# ML-CUP25 v1\n")
        f.write("# dd/mm/yyyy\n")  # TODO: metti la data vera (es. 01/12/2025)

        for idx, row in zip(ids_test, y_pred_test):
            o1, o2, o3, o4 = row
            f.write(f"{int(idx)},{o1:.6f},{o2:.6f},{o3:.6f},{o4:.6f}\n")

    print(f">>> File blind test scritto in: {out_path}")


# ============================================================
# 10) MAIN
# ============================================================

if __name__ == "__main__":
    # 1) Esperimenti base CUP (senza grid search, solo 3 modelli)
    print("=== STEP 1: Esperimenti base CUP (NN, Linear, k-NN) ===")
    base_info = run_cup_experiments()

    # 2) GRID SEARCH completa CUP (154 configurazioni)
    print("\n=== STEP 2: GRID SEARCH CUP ===")
    df_results, best, mean, std, data = run_cup_grid_search()

    # Salvo i risultati della grid search per il report/slide
    df_results.to_csv("grid_search_results.csv", index=False)
    print(">>> Salvato grid_search_results.csv (tutti i risultati della grid search).")

    # 3) (Opzionale) Esegui MONK (decommenta se hai i file pronti)
    # run_monk_experiment("monk1_train.csv", "monk1_test.csv", monk_name="MONK-1")
    # run_monk_experiment("monk2_train.csv", "monk2_test.csv", monk_name="MONK-2")
    # run_monk_experiment("monk3_train.csv", "monk3_test.csv", monk_name="MONK-3")

    # 4) (Opzionale) Addestramento finale e file per blind test CUP
    #    Prima di decommentare: sostituisci TEAMNAME con il nome del team
    # train_final_and_save_cup_predictions(
    #     model_type="nn",      # oppure "linear"
    #     team_name="TEAMNAME"
    # )

    print("\n>>> FINE ESECUZIONE SCRIPT.")
