import tensorflow as tf
import numpy as np
from loss import mee_np
class knn_model:
    def __init__(self, x_train, y_train, k):
        # Convert training data to TensorFlow tensors. Tensors must be stored in order to use them to calculate distances during prediction.
        self.x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        self.num_samples = x_train.shape[0]
        # k must be ensured that it does not exceed number of training samples. k must be less or equal to the number of training samples.
        self.k=min(k, self.num_samples)  
        # every input sample has an output with a four dimensions for the cup, but if it's classification it could be one dimension
        if(len(self.y_train.shape)==1):
            self.n_outputs=1
        else:
            self.n_outputs=self.y_train.shape[1]

    def predict_regression(self, x_new):
        x_new_tf = tf.convert_to_tensor(x_new, dtype=tf.float32)
        # Calculate L2 distances between X_test and X_train
        distances = tf.norm(tf.expand_dims(x_new_tf, axis=1) - tf.expand_dims(self.x_train, axis=0), axis=2)
        # Order the distances by descent order and get the indices of the k nearest neighbors
        knn_indices = tf.argsort(distances, axis=1)[:, :self.k]
        # Given the k nearest neighbors' indices, gather their outputs
        knn_outputs = tf.gather(self.y_train, knn_indices)
        # Calculate the mean of the k nearest neighbors' outputs
        y_pred = tf.reduce_mean(knn_outputs, axis=1)
        return y_pred.numpy()

    def predict_classification(self, x_new):
        x_new_tf = tf.convert_to_tensor(x_new, dtype=tf.float32)
        distances = tf.norm(tf.expand_dims(x_new_tf, 1) - tf.expand_dims(self.x_train, 0), axis=2)
        knn_idx = tf.argsort(distances, axis=1)[:, :self.k]
        knn_y = tf.gather(self.y_train, knn_idx)
        # voto di maggioranza
        y_pred = tf.reduce_mean(knn_y, axis=1)
        return tf.where(y_pred >= 0.5, 1.0, 0.0).numpy()
    
def grid_search(x_train,y_train,x_validation,y_validation,k_values,is_regression):
    min_error=float('inf')
    error=0
    best_k=k_values[0]
    for k in k_values:
        model=knn_model(x_train,y_train,k)
        if is_regression:
            y_pred=model.predict_regression(x_validation)
            error=mee_np(y_validation,y_pred)
        else:
            y_pred=model.predict_classification(x_validation)
            error = np.mean(y_pred.flatten() != y_validation.flatten())
        if error<min_error:
            min_error=error
            best_k=k
        print(f'k={k}, MEE={error}'if is_regression else f'k={k}, Error rate={error}')
    return best_k

def hold_out(x,y,k_values,validation_split,is_regression):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    split = int(x.shape[0] * (1 - validation_split))
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    best_k=grid_search(x_train,y_train,x_val,y_val,k_values,is_regression)
    return knn_model(x,y,best_k),best_k

