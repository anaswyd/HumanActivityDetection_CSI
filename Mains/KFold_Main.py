import os
from Preprocessing import CSI_Preprocessing
from models.simple_model import SimpleModel
from models.cnn_model import CNNModel
from Validation.KFoldValidator import KFoldCrossValidator, compare_kfold_results


def main():
    """
    Hauptpipeline: K-Fold Cross-Validation für CSI-Klassifikation.

    Trainiert und evaluiert beide Modelle (Simple & CNN) jeweils
    mit und ohne Data Augmentation using K-Fold Validation.
    """

    # ========== KONFIGURATION ==========

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, '..', 'Data')

    # Daten-Parameter
    FIXED_LENGTH = 500
    AUGMENTATION_FACTOR = 2

    # K-Fold Parameter
    N_SPLITS = 5

    # Training Parameter
    EPOCHS = 20
    BATCH_SIZE = 16

    print("=" * 80)
    print("CSI DATA CLASSIFICATION - K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"\nKonfiguration:")
    print(f"  Sequence Length:    {FIXED_LENGTH}")
    print(f"  K-Folds:            {N_SPLITS}")
    print(f"  Epochs pro Fold:    {EPOCHS}")
    print(f"  Batch Size:         {BATCH_SIZE}")
    print(f"  Augmentation Factor: {AUGMENTATION_FACTOR}")

    # ========== DATEN LADEN ==========
    print(f"\n{'=' * 80}")
    print("DATEN LADEN")
    print(f"{'=' * 80}")

    preprocessor = CSI_Preprocessing.CSI_Preprocessing(data_folder)
    X_all, y_all, label_encoder = preprocessor.prepare_data_for_kfold(
        fixed_length=FIXED_LENGTH,
        use_augmentation=False  # Augmentation wird pro Fold gemacht
    )

    if X_all is None:
        print("Fehler: Keine Daten gefunden!")
        return

    print(f"\nGeladene Daten:")
    print(f"  Total Samples:   {len(X_all)}")
    print(f"  Input Shape:     {X_all.shape[1:]} (Zeitschritte, Features)")
    print(f"  Anzahl Klassen:  {len(label_encoder.classes_)}")
    print(f"  Klassen:         {label_encoder.classes_}")

    # ========== SETUP ==========
    input_shape = (FIXED_LENGTH, X_all.shape[2])
    num_classes = len(label_encoder.classes_)

    model_params = {
        'input_shape': input_shape,
        'num_classes': num_classes
    }

    training_params = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }

    # K-Fold Validator
    kfold_validator = KFoldCrossValidator(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    # Ergebnisse speichern
    all_results = []
    config_names = []

    # ========== 1. LSTM MODEL OHNE AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("1/4: LSTM MODEL - OHNE AUGMENTATION")
    print(f"{'=' * 80}")

    results_lstm_no_aug = kfold_validator.run(
        model_class=SimpleModel,
        X=X_all,
        y=y_all,
        model_params=model_params,
        training_params=training_params,
        use_augmentation=False
    )

    kfold_validator.print_summary("LSTM Model (ohne Aug)")
    kfold_validator.plot_results("LSTM Model (ohne Aug)")

    all_results.append(results_lstm_no_aug)
    config_names.append("LSTM\n(ohne Aug)")

    # ========== 2. LSTM MODEL MIT AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("2/4: LSTM MODEL - MIT AUGMENTATION")
    print(f"{'=' * 80}")

    kfold_validator = KFoldCrossValidator(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results_lstm_aug = kfold_validator.run(
        model_class=SimpleModel,
        X=X_all,
        y=y_all,
        model_params=model_params,
        training_params=training_params,
        use_augmentation=True,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    kfold_validator.print_summary("LSTM Model (mit Aug)")
    kfold_validator.plot_results("LSTM Model (mit Aug)")

    all_results.append(results_lstm_aug)
    config_names.append("LSTM\n(mit Aug)")

    # ========== 3. CNN MODEL OHNE AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("3/4: CNN MODEL - OHNE AUGMENTATION")
    print(f"{'=' * 80}")

    kfold_validator = KFoldCrossValidator(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results_cnn_no_aug = kfold_validator.run(
        model_class=CNNModel,
        X=X_all,
        y=y_all,
        model_params=model_params,
        training_params=training_params,
        use_augmentation=False
    )

    kfold_validator.print_summary("CNN Model (ohne Aug)")
    kfold_validator.plot_results("CNN Model (ohne Aug)")

    all_results.append(results_cnn_no_aug)
    config_names.append("CNN\n(ohne Aug)")

    # ========== 4. CNN MODEL MIT AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("4/4: CNN MODEL - MIT AUGMENTATION")
    print(f"{'=' * 80}")

    kfold_validator = KFoldCrossValidator(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results_cnn_aug = kfold_validator.run(
        model_class=CNNModel,
        X=X_all,
        y=y_all,
        model_params=model_params,
        training_params=training_params,
        use_augmentation=True,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    kfold_validator.print_summary("CNN Model (mit Aug)")
    kfold_validator.plot_results("CNN Model (mit Aug)")

    all_results.append(results_cnn_aug)
    config_names.append("CNN\n(mit Aug)")

    # ========== FINALER VERGLEICH ==========
    print(f"\n{'=' * 80}")
    print("FINALER VERGLEICH ALLER KONFIGURATIONEN")
    print(f"{'=' * 80}")

    # Detaillierte Zusammenfassung
    print("\nDetaillierte Ergebnisse:")
    print("-" * 80)

    config_names_clean = [
        "LSTM Model (ohne Aug)",
        "LSTM Model (mit Aug)",
        "CNN Model (ohne Aug)",
        "CNN Model (mit Aug)"
    ]

    for name, result in zip(config_names_clean, all_results):
        print(f"\n{name}:")
        print(f"  Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} "
              f"({result['mean_accuracy'] * 100:.2f}% ± {result['std_accuracy'] * 100:.2f}%)")
        print(f"  Mean Loss:     {result['mean_loss']:.4f} ± {result['std_loss']:.4f}")
        print(f"  Best Fold:     {result['max_accuracy']:.4f} ({result['max_accuracy'] * 100:.2f}%)")
        print(f"  Worst Fold:    {result['min_accuracy']:.4f} ({result['min_accuracy'] * 100:.2f}%)")
        print(f"  Range:         {result['max_accuracy'] - result['min_accuracy']:.4f}")

    # Vergleichsplot
    print("\nErstelle Vergleichsplot...")
    compare_kfold_results(all_results, config_names)

    # ========== BESTE KONFIGURATION ==========
    best_idx = max(range(len(all_results)),
                   key=lambda i: all_results[i]['mean_accuracy'])
    best_config = config_names_clean[best_idx]
    best_result = all_results[best_idx]

    print(f"\n{'=' * 80}")
    print("EMPFEHLUNG")
    print(f"{'=' * 80}")
    print(f"\nBeste Konfiguration: {best_config}")
    print(f"  Mean Accuracy: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
    print(f"                 ({best_result['mean_accuracy'] * 100:.2f}% ± {best_result['std_accuracy'] * 100:.2f}%)")

    # Verbesserung durch Augmentation analysieren
    lstm_improvement = ((results_lstm_aug['mean_accuracy'] - results_lstm_no_aug['mean_accuracy'])
                        / results_lstm_no_aug['mean_accuracy'] * 100)
    cnn_improvement = ((results_cnn_aug['mean_accuracy'] - results_cnn_no_aug['mean_accuracy'])
                       / results_cnn_no_aug['mean_accuracy'] * 100)

    print(f"\nEffekt von Data Augmentation:")
    print(f"  LSTM Model: {lstm_improvement:+.2f}%")
    print(f"  CNN Model:    {cnn_improvement:+.2f}%")

    if lstm_improvement > 0 or cnn_improvement > 0:
        print("\nAugmentation verbessert die Performance - empfohlen!")
    else:
        print("\nAugmentation hilft nicht - Dataset ausreichend groß.")

    # Modell-Vergleich
    lstm_best = max(results_lstm_no_aug['mean_accuracy'], results_lstm_aug['mean_accuracy'])
    cnn_best = max(results_cnn_no_aug['mean_accuracy'], results_cnn_aug['mean_accuracy'])

    print(f"\nModell-Vergleich:")
    print(f"  LSTM Model (best): {lstm_best:.4f} ({lstm_best * 100:.2f}%)")
    print(f"  CNN Model (best):    {cnn_best:.4f} ({cnn_best * 100:.2f}%)")

    if cnn_best > lstm_best:
        improvement = ((cnn_best - lstm_best) / lstm_best * 100)
        print(f"\nCNN Model ist {improvement:.2f}% besser")
    else:
        improvement = ((lstm_best - cnn_best) / cnn_best * 100)
        print(f"\nLSTM Model reicht aus ({improvement:.2f}% besser als CNN)")

    print(f"\n{'=' * 80}")
    print("K-FOLD VALIDATION ABGESCHLOSSEN")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()