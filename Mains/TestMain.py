import os
import numpy as np
import tensorflow as tf
from Preprocessing.TestPreparation import Preprocessing
from models.cnn_model import CNNModel
from models.simple_model import SimpleModel
import matplotlib.pyplot as plt

def main():
    # Setup
    base_dir = os.path.dirname(__file__)
    data_folder = os.path.join(base_dir, 'Data')

    FIXED_LENGTH = 500

    # Training Parameter
    EPOCHS = 20
    BATCH_SIZE = 16  # kleinere size bei wenigen Daten empfohlen

    # Daten laden
    preprocessor = Preprocessing(data_folder)

    print("\nStarte Datenvorbereitung...")
    X_train, X_test, y_train, y_test, label_encoder = preprocessor.prepare_data(
        fixed_length=FIXED_LENGTH
    )

    if X_train is None:
        print("Keine Daten gefunden...")
        return

    print(f"\nTrainings-Samples: {len(X_train)}")
    print(f"Test-Samples:      {len(X_test)}")
    print(f"Input Shape:       {X_train.shape[1:]} (Zeitschritte, Subcarrier)")

    # Modell initialisieren
    input_shape = (FIXED_LENGTH, X_train.shape[2])  # (500, 256)
    num_classes = len(label_encoder.classes_)

    simple_model = SimpleModel(input_shape, num_classes)


    history = simple_model.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # plot training results:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # plot loss value
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    print("\nEvaluating Model...")
    simple_model.evaluate(X_test, y_test)

    cnn_model = CNNModel(input_shape, num_classes)
    history = cnn_model.train(X_train, y_train, X_test, y_test, epochs=EPOCHS)


    # plot training results:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # plot loss value
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    print("\nEvaluating Model...")
    cnn_model.evaluate(X_test, y_test)

    print("Pipeline fertig")


# Für einzelne auch acc berechnen & anzeigen lassen..

if __name__ == "__main__":
    main()