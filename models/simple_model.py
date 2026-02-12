import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


class SimpleModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a simple LSTM model.
        """
        model = Sequential([
            LSTM(units=63, input_shape=self.input_shape),
            Dropout(0.5),
            Dense(units=16, activation='gelu'),
            Dense(units=self.num_classes, activation='softmax')
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0) # clipnorm für

        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
        """
        Trains the model.
        """
        print("Starting training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model on the test set.
        """
        labels = ["Empty", "Lying", "Sitting", "Standing", "Walking"]
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        print("Confusion Matrix:")
        y_pred = self.model.predict(X_test)
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true_classes = tf.argmax(y_test, axis=1)
        confusion_matrix = tf.math.confusion_matrix(y_true_classes, y_pred_classes)
        #plot confusion matrix with its labels:
        sns.heatmap(confusion_matrix, annot=True, fmt='d',cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.show()

        target_names = ['Empty', 'Lying', 'Sitting', 'Standing', 'Walking']
        y_pred = self.model.predict(X_test)
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=target_names))

        self.model.summary()

        return loss, accuracy
