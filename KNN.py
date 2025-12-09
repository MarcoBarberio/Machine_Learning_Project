import tensorflow as tf
import numpy as np
from loss import mee
class knn_model:
    def __init__(self, x_train, y_train, k):
        # Convert training data to TensorFlow tensors. Tensors must be stored in order to use them to calculate distances during prediction.
        self.x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        self.num_samples = x_train.shape[0]
        # k must be ensured that it does not exceed number of training samples. k must be less or equal to the number of training samples.
        self.k=min(k, self.num_samples)  
        # every input sample has an output with a four dimensions
        self.n_outputs=self.y_train.shape[1]

    def predict_regression(self, x_test):
        x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
        # Calculate L2 distances between X_test and X_train
        distances = tf.norm(tf.expand_dims(x_test_tf, axis=1) - tf.expand_dims(self.x_train, axis=0), axis=2)
        # Order the distances by descent order and get the indices of the k nearest neighbors
        knn_indices = tf.argsort(distances, axis=1)[:, :self.k]
        # Given the k nearest neighbors' indices, gather their outputs
        knn_outputs = tf.gather(self.y_train, knn_indices)
        # Calculate the mean of the k nearest neighbors' outputs
        y_pred = tf.reduce_mean(knn_outputs, axis=1)
        return y_pred
    
def grid_search(x_train,y_train,x_validation,y_validation,k_values):
    min_error=float('inf')
    best_k=k_values[0]
    for k in k_values:
        model=knn_model(x_train,y_train,k)
        y_pred=model.predict_regression(x_validation)
        error=mee(y_validation,y_pred)
        if error<min_error:
            min_error=error
            best_k=k
        print(f'k={k}, MEE={error}')
    return best_k

