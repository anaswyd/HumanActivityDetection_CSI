import os
import numpy as np
import tensorflow as tf
from Preprocessing import CSI_Preprocessing
from models.cnn_model import CNNModel
from models.simple_model import SimpleModel
import matplotlib.pyplot as plt


def plot_training_history(history, model_name, augmentation_status):
    """
    Plottet Accuracy und Loss für ein trainiertes Modell.

    Args:
        history: Training History Objekt von Keras
        model_name: Name des Modells (z.B. "Simple Model", "CNN Model")
        augmentation_status: String wie "mit Augmentation" oder "ohne Augmentation"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy ({augmentation_status})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss Plot
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title(f'{model_name} - Loss ({augmentation_status})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                             model_name, augmentation_status, epochs=20, batch_size=16):
    """
    Trainiert und evaluiert ein Modell.

    Args:
        model: Das zu trainierende Modell (SimpleModel oder CNNModel)
        X_train, y_train: Training Daten
        X_test, y_test: Test Daten
        model_name: Name des Modells
        augmentation_status: "mit Augmentation" oder "ohne Augmentation"
        epochs: Anzahl der Trainingsepochen
        batch_size: Batch Size für Training

    Returns:
        history: Training History
        test_accuracy: Finale Test Accuracy
        test_loss: Finale Test Loss
    """
    print(f"\n{'=' * 80}")
    print(f"Training: {model_name} ({augmentation_status})")
    print(f"{'=' * 80}")

    # Training
    history = model.train(X_train, y_train, X_test, y_test,
                          epochs=epochs, batch_size=batch_size)

    # Plots
    plot_training_history(history, model_name, augmentation_status)

    # Evaluation
    print(f"\nEvaluiere {model_name} ({augmentation_status})...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f"\n{model_name} ({augmentation_status}) - Finale Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} ({augmentation_status}) - Finale Test Loss: {test_loss:.4f}")

    return history, test_accuracy, test_loss


def main():
    """
    Hauptfunktion: Trainiert beide Modelle (Simple & CNN) jeweils mit und ohne Augmentation.
    """
    # ========== Setup ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, '..', 'Data')

    # Parameter
    FIXED_LENGTH = 500
    EPOCHS = 20
    BATCH_SIZE = 16
    TEST_SIZE = 0.2
    AUGMENTATION_FACTOR = 2  # Jeder Sample wird 2x augmentiert (Original + 1x augmentiert)

    print("=" * 80)
    print("CSI Data Classification Pipeline - Vergleich mit/ohne Data Augmentation")
    print("=" * 80)

    # ========== Daten ohne Augmentation laden ==========
    print("\n[1/2] Lade Daten OHNE Augmentation...")
    preprocessor_no_aug = CSI_Preprocessing.CSI_Preprocessing(data_folder)
    X_train_no_aug, X_test_no_aug, y_train_no_aug, y_test_no_aug, label_encoder = \
        preprocessor_no_aug.prepare_data(
            fixed_length=FIXED_LENGTH,
            test_size=TEST_SIZE,
            use_augmentation=False
        )

    if X_train_no_aug is None:
        print("Fehler: Keine Daten gefunden!")
        return

    print(f"\nDaten OHNE Augmentation:")
    print(f"  Training Samples: {len(X_train_no_aug)}")
    print(f"  Test Samples:     {len(X_test_no_aug)}")
    print(f"  Input Shape:      {X_train_no_aug.shape[1:]} (Zeitschritte, Features)")
    print(f"  Anzahl Klassen:   {len(label_encoder.classes_)}")
    print(f"  Klassen:          {label_encoder.classes_}")

    # ========== Daten mit Augmentation laden ==========
    print(f"\n[2/2] Lade Daten MIT Augmentation (Faktor: {AUGMENTATION_FACTOR})...")
    preprocessor_aug = CSI_Preprocessing.CSI_Preprocessing(data_folder)
    X_train_aug, X_test_aug, y_train_aug, y_test_aug, _ = \
        preprocessor_aug.prepare_data(
            fixed_length=FIXED_LENGTH,
            test_size=TEST_SIZE,
            use_augmentation=True,
            augmentation_factor=AUGMENTATION_FACTOR
        )

    print(f"\nDaten MIT Augmentation:")
    print(f"  Training Samples: {len(X_train_aug)}")
    print(f"  Test Samples:     {len(X_test_aug)}")

    # ========== Model Setup ==========
    input_shape = (FIXED_LENGTH, X_train_no_aug.shape[2])
    num_classes = len(label_encoder.classes_)

    # Dictionary zum Speichern der Ergebnisse
    results = {}

    # ========== 1. Simple Model OHNE Augmentation ==========
    simple_model_no_aug = SimpleModel(input_shape, num_classes)
    history_simple_no_aug, acc_simple_no_aug, loss_simple_no_aug = train_and_evaluate_model(
        simple_model_no_aug, X_train_no_aug, y_train_no_aug, X_test_no_aug, y_test_no_aug,
        "Simple Model", "ohne Augmentation", EPOCHS, BATCH_SIZE
    )
    results['Simple (ohne Aug)'] = acc_simple_no_aug

    # ========== 2. Simple Model MIT Augmentation ==========
    simple_model_aug = SimpleModel(input_shape, num_classes)
    history_simple_aug, acc_simple_aug, loss_simple_aug = train_and_evaluate_model(
        simple_model_aug, X_train_aug, y_train_aug, X_test_aug, y_test_aug,
        "Simple Model", "mit Augmentation", EPOCHS, BATCH_SIZE
    )
    results['Simple (mit Aug)'] = acc_simple_aug

    # ========== 3. CNN Model OHNE Augmentation ==========
    cnn_model_no_aug = CNNModel(input_shape, num_classes)
    history_cnn_no_aug, acc_cnn_no_aug, loss_cnn_no_aug = train_and_evaluate_model(
        cnn_model_no_aug, X_train_no_aug, y_train_no_aug, X_test_no_aug, y_test_no_aug,
        "CNN Model", "ohne Augmentation", EPOCHS, BATCH_SIZE
    )
    results['CNN (ohne Aug)'] = acc_cnn_no_aug

    # ========== 4. CNN Model MIT Augmentation ==========
    cnn_model_aug = CNNModel(input_shape, num_classes)
    history_cnn_aug, acc_cnn_aug, loss_cnn_aug = train_and_evaluate_model(
        cnn_model_aug, X_train_aug, y_train_aug, X_test_aug, y_test_aug,
        "CNN Model", "mit Augmentation", EPOCHS, BATCH_SIZE
    )
    results['CNN (mit Aug)'] = acc_cnn_aug

    # ========== Zusammenfassung ==========
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG DER ERGEBNISSE")
    print("=" * 80)

    print(f"\nSimple Model:")
    print(f"  Ohne Augmentation: {acc_simple_no_aug:.4f} ({acc_simple_no_aug * 100:.2f}%)")
    print(f"  Mit Augmentation:  {acc_simple_aug:.4f} ({acc_simple_aug * 100:.2f}%)")
    improvement_simple = ((acc_simple_aug - acc_simple_no_aug) / acc_simple_no_aug) * 100
    print(f"  Verbesserung:      {improvement_simple:+.2f}%")

    print(f"\nCNN Model:")
    print(f"  Ohne Augmentation: {acc_cnn_no_aug:.4f} ({acc_cnn_no_aug * 100:.2f}%)")
    print(f"  Mit Augmentation:  {acc_cnn_aug:.4f} ({acc_cnn_aug * 100:.2f}%)")
    improvement_cnn = ((acc_cnn_aug - acc_cnn_no_aug) / acc_cnn_no_aug) * 100
    print(f"  Verbesserung:      {improvement_cnn:+.2f}%")

    # Vergleichsplot
    print("\nErstelle Vergleichsplot...")
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[m] * 100 for m in models]
    colors = ['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4']

    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Werte auf Balken schreiben
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Vergleich: Modell-Performance mit/ohne Data Augmentation',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("Pipeline abgeschlossen!")
    print("=" * 80)


if __name__ == "__main__":
    main()