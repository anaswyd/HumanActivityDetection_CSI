import os
from Validation import CSIGroupKFold
from models.simple_model import SimpleModel
from models.cnn_model import CNNModel
import matplotlib.pyplot as plt
import numpy as np


def compare_results(results_list, config_names):
    """Vergleicht mehrere Ergebnisse."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(config_names))
    means = [r['mean_accuracy'] * 100 for r in results_list]
    stds = [r['std_accuracy'] * 100 for r in results_list]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    width = 0.6

    bars1 = ax1.bar(x, means, width, yerr=stds, capsize=8,
                    color=colors[:len(config_names)],
                    edgecolor='black', linewidth=2, alpha=0.85,
                    error_kw={'linewidth': 2, 'ecolor': 'darkgray'})

    for bar, mean, std in zip(bars1, means, stds):
        height = bar.get_height()
        y_pos = height + std + 2
        ax1.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{mean:.2f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'±{std:.2f}%',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax1.set_ylabel('Mean Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('CSI Group K-Fold: Accuracy Vergleich',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 110])
    ax1.set_axisbelow(True)

    # Loss
    means_loss = [r['mean_loss'] for r in results_list]
    stds_loss = [r['std_loss'] for r in results_list]

    bars2 = ax2.bar(x, means_loss, width, yerr=stds_loss, capsize=8,
                    color=colors[:len(config_names)],
                    edgecolor='black', linewidth=2, alpha=0.85,
                    error_kw={'linewidth': 2, 'ecolor': 'darkgray'})

    for bar, mean, std in zip(bars2, means_loss, stds_loss):
        height = bar.get_height()
        y_pos = height + std + max(means_loss) * 0.05
        ax2.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{mean:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax2.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'±{std:.4f}',
                 ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax2.set_ylabel('Mean Loss', fontsize=14, fontweight='bold')
    ax2.set_title('CSI Group K-Fold: Loss Vergleich',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    max_loss = max(means_loss)
    max_std = max(stds_loss)
    ax2.set_ylim([0, max_loss + max_std + max_loss * 0.2])

    plt.tight_layout()
    plt.show()


def main():
    """
    CSI Group K-Fold Cross-Validation Pipeline.

    Optimiert für CSI-Daten:
    - Behandelt Sessions als Gruppen (kein Data Leakage)
    - Funktioniert auch mit wenigen Sessions pro Klasse
    - Besser als Standard K-Fold (gruppiert)
    - Besser als LOSO (wenn wenig Sessions)
    """

    # ========== KONFIGURATION ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, '..', 'Data')

    FIXED_LENGTH = 500
    N_SPLITS = 5
    AUGMENTATION_FACTOR = 2
    EPOCHS = 20
    BATCH_SIZE = 16

    print("=" * 80)
    print("CSI DATA CLASSIFICATION - GROUP K-FOLD CROSS-VALIDATION")
    print("Sessions gruppiert, kein Data Leakage")
    print("=" * 80)
    print(f"\nKonfiguration:")
    print(f"  Sequence Length:     {FIXED_LENGTH}")
    print(f"  K-Folds:             {N_SPLITS}")
    print(f"  Epochs pro Fold:     {EPOCHS}")
    print(f"  Batch Size:          {BATCH_SIZE}")
    print(f"  Augmentation Factor: {AUGMENTATION_FACTOR}")
   # print(f"\nVorteile von Group K-Fold:")
   # print(f"  ✓ Sessions werden als Gruppen behandelt (kein Data Leakage)")
   # print(f"  ✓ Funktioniert auch mit wenigen Sessions pro Klasse")
   # print(f"  ✓ Stellt sicher dass alle Klassen im Training sind")

    # Dummy load für Input Shape
    from Preprocessing import CSI_Preprocessing
    prep = CSI_Preprocessing.CSI_Preprocessing(data_folder)
    X_dummy, _, le = prep.prepare_data_for_kfold(fixed_length=FIXED_LENGTH)

    if X_dummy is None:
        print("\nFehler: Keine Daten gefunden!")
        return

    input_shape = (FIXED_LENGTH, X_dummy.shape[2])
    num_classes = len(le.classes_)

    model_params = {
        'input_shape': input_shape,
        'num_classes': num_classes
    }

    training_params = {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }

    all_results = []
    config_names = []

    # ========== 1. LSTM MODEL OHNE AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("1/4: LSTM MODEL - OHNE AUGMENTATION")
    print(f"{'=' * 80}")

    validator1 = CSIGroupKFold.CSIGroupKFold(
        data_folder=data_folder,
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    results_lstm_no_aug = validator1.run(
        model_class=SimpleModel,
        model_params=model_params,
        training_params=training_params,
        fixed_length=FIXED_LENGTH,
        use_augmentation=False
    )

    if results_lstm_no_aug:
        validator1.print_summary("LSTM Model (ohne Aug)")
        validator1.plot_results("LSTM Model (ohne Aug)")
        all_results.append(results_lstm_no_aug)
        config_names.append("LSTM\n(ohne Aug)")

    # ========== 2. LSTM MODEL MIT AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("2/4: LSTM MODEL - MIT AUGMENTATION")
    print(f"{'=' * 80}")

    validator2 = CSIGroupKFold.CSIGroupKFold(
        data_folder=data_folder,
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    results_lstm_aug = validator2.run(
        model_class=SimpleModel,
        model_params=model_params,
        training_params=training_params,
        fixed_length=FIXED_LENGTH,
        use_augmentation=True,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    if results_lstm_aug:
        validator2.print_summary("LSTM Model (mit Aug)")
        validator2.plot_results("LSTM Model (mit Aug)")
        all_results.append(results_lstm_aug)
        config_names.append("LSTM\n(mit Aug)")

    # ========== 3. CNN MODEL OHNE AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("3/4: CNN MODEL - OHNE AUGMENTATION")
    print(f"{'=' * 80}")

    validator3 = CSIGroupKFold.CSIGroupKFold(
        data_folder=data_folder,
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    results_cnn_no_aug = validator3.run(
        model_class=CNNModel,
        model_params=model_params,
        training_params=training_params,
        fixed_length=FIXED_LENGTH,
        use_augmentation=False
    )

    if results_cnn_no_aug:
        validator3.print_summary("CNN Model (ohne Aug)")
        validator3.plot_results("CNN Model (ohne Aug)")
        all_results.append(results_cnn_no_aug)
        config_names.append("CNN\n(ohne Aug)")

    # ========== 4. CNN MODEL MIT AUGMENTATION ==========
    print(f"\n{'=' * 80}")
    print("4/4: CNN MODEL - MIT AUGMENTATION")
    print(f"{'=' * 80}")

    validator4 = CSIGroupKFold.CSIGroupKFold(
        data_folder=data_folder,
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    results_cnn_aug = validator4.run(
        model_class=CNNModel,
        model_params=model_params,
        training_params=training_params,
        fixed_length=FIXED_LENGTH,
        use_augmentation=True,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    if results_cnn_aug:
        validator4.print_summary("CNN Model (mit Aug)")
        validator4.plot_results("CNN Model (mit Aug)")
        all_results.append(results_cnn_aug)
        config_names.append("CNN\n(mit Aug)")

    # ========== FINALER VERGLEICH ==========
    if len(all_results) >= 2:
        print(f"\n{'=' * 80}")
        print("FINALER VERGLEICH")
        print(f"{'=' * 80}")

        config_names_clean = [
            "LSTM Model (ohne Aug)",
            "LSTM Model (mit Aug)",
            "CNN Model (ohne Aug)",
            "CNN Model (mit Aug)"
        ][:len(all_results)]

        for name, result in zip(config_names_clean, all_results):
            print(f"\n{name}:")
            print(f"  Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
            print(f"  Mean Loss:     {result['mean_loss']:.4f} ± {result['std_loss']:.4f}")

        # Vergleichsplot
        compare_results(all_results, config_names)

        # Beste Konfiguration
        best_idx = max(range(len(all_results)),
                       key=lambda i: all_results[i]['mean_accuracy'])
        best_config = config_names_clean[best_idx]

        print(f"\n{'=' * 80}")
        print("EMPFEHLUNG")
        print(f"{'=' * 80}")
        print(f"\nBeste Konfiguration: {best_config}")
        print(f"  Mean Accuracy: {all_results[best_idx]['mean_accuracy']:.4f} ± "
              f"{all_results[best_idx]['std_accuracy']:.4f}")

        # Augmentation Effekt
        if len(all_results) >= 4:
            lstm_improvement = ((results_lstm_aug['mean_accuracy'] - results_lstm_no_aug['mean_accuracy'])
                                / results_lstm_no_aug['mean_accuracy'] * 100)
            cnn_improvement = ((results_cnn_aug['mean_accuracy'] - results_cnn_no_aug['mean_accuracy'])
                               / results_cnn_no_aug['mean_accuracy'] * 100)

            print(f"\nEffekt von Data Augmentation:")
            print(f"  LSTM Model: {lstm_improvement:+.2f}%")
            print(f"  CNN Model:  {cnn_improvement:+.2f}%")

    print(f"\n{'=' * 80}")
    print("CSI GROUP K-FOLD VALIDATION ABGESCHLOSSEN")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()