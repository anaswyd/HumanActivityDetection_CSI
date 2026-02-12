import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

class CNNModel:
    def __init__(self, input_shape, num_classes):
        """
        Initialisiert das CNN Modell für Zeitreihenklassifikation.

        Args:
            input_shape: Tupel (Zeitschritte, Features), z.B. (500, 256)
            num_classes: Anzahl der Aktivitäten (Labels)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        print(f"Erstelle 1D-CNN Modell mit Input Shape {self.input_shape}...")

        model = Sequential([
            Conv1D(filters=32, kernel_size=5, activation='gelu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            GlobalAveragePooling1D(),

            Dropout(0.5),
            Dense( 16, activation='gelu'),
            Dense(self.num_classes, activation='softmax')
        ])

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        # Callbacks für intelligentes Training
        callbacks = [
            # Stoppt, wenn das Modell nicht mehr besser wird (verhindert Overfitting)
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduziert Lernrate, wenn es stagniert
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]

        print("\nStarte CNN Training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
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