import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class KFoldCrossValidator:
    """
    K-Fold Cross-Validation für CSI-Klassifikationsmodelle.

    Führt stratifizierte K-Fold Validation durch und liefert robuste
    Metriken (Mean ± Std) für Modell-Evaluation.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Initialisiert den K-Fold Validator.

        Args:
            n_splits: Anzahl der Folds (Standard: 5)
            shuffle: Ob Daten vor Split gemischt werden sollen
            random_state: Seed für Reproduzierbarkeit
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        self.kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        self.results = None

    def _augment_data(self, X_train, y_train, augmentation_factor=2):
        """
        Wendet Data Augmentation auf Training Daten an.

        Args:
            X_train: Training Daten (N, timesteps, features)
            y_train: Training Labels (N, num_classes)
            augmentation_factor: Wie oft augmentiert werden soll

        Returns:
            X_augmented, y_augmented
        """
        augmented_X = []
        augmented_y = []

        for i in range(len(X_train)):
            # Original behalten
            augmented_X.append(X_train[i])
            augmented_y.append(y_train[i])

            # Augmentierte Versionen erstellen
            for _ in range(augmentation_factor - 1):
                aug_sample = self._augment_single_sample(X_train[i])
                augmented_X.append(aug_sample)
                augmented_y.append(y_train[i])

        return np.array(augmented_X, dtype=np.float32), np.array(augmented_y, dtype=np.float32)

    def _augment_single_sample(self, data):
        """
        Augmentiert einen einzelnen Sample.

        Techniken:
        - Magnitude Scaling (±10%)
        - Gaussian Noise (1%)
        - Time Shifting (±10 Schritte)
        """
        augmented = data.copy()

        # Magnitude Scaling
        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale_factor

        # Gaussian Noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = augmented + noise

        # Time Shifting
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            augmented = np.roll(augmented, shift, axis=0)

        return augmented.astype(np.float32)

    def run(self, model_class, X, y, model_params, training_params,
            use_augmentation=False, augmentation_factor=2):
        """
        Führt K-Fold Cross-Validation durch.

        Args:
            model_class: Modell-Klasse (z.B. SimpleModel, CNNModel)
            X: Alle Daten (N, timesteps, features)
            y: Alle Labels (N, num_classes) - one-hot encoded
            model_params: Dict mit Parametern für Modell-Initialisierung
            training_params: Dict mit Training-Parametern
            use_augmentation: Ob Augmentation verwendet werden soll
            augmentation_factor: Augmentations-Faktor

        Returns:
            Dict mit Ergebnissen
        """
        print(f"\n{'=' * 80}")
        print(f"K-Fold Cross-Validation: {self.n_splits} Folds")
        print(f"Augmentation: {'Ja' if use_augmentation else 'Nein'}")
        print(f"{'=' * 80}")

        # Ergebnisse speichern
        fold_accuracies = []
        fold_losses = []
        fold_histories = []
        trained_models = []

        # Integer Labels für Stratified Split
        y_int = np.argmax(y, axis=1)

        # K-Fold durchführen
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X, y_int), 1):
            print(f"\n--- Fold {fold_idx}/{self.n_splits} ---")

            # Split
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            print(f"Train: {len(X_train_fold)} samples, Val: {len(X_val_fold)} samples")

            # Optional: Augmentation
            if use_augmentation:
                print(f"Wende Augmentation an (Faktor: {augmentation_factor})...")
                X_train_fold, y_train_fold = self._augment_data(
                    X_train_fold, y_train_fold, augmentation_factor
                )
                print(f"Nach Augmentation: {len(X_train_fold)} training samples")

            # Scaling (nur auf Train fitten!)
            scaler = StandardScaler()
            N, T, F = X_train_fold.shape

            X_train_2d = X_train_fold.reshape(N * T, F)
            scaler.fit(X_train_2d)
            X_train_fold = scaler.transform(X_train_2d).reshape(N, T, F)

            # Validation: nur transformieren
            N_val, T_val, F_val = X_val_fold.shape
            X_val_fold = scaler.transform(X_val_fold.reshape(N_val * T_val, F_val)).reshape(N_val, T_val, F_val)

            # NaN-Safety
            X_train_fold = np.nan_to_num(X_train_fold, nan=0.0)
            X_val_fold = np.nan_to_num(X_val_fold, nan=0.0)

            # Modell erstellen und trainieren
            model = model_class(**model_params)

            history = model.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                **training_params
            )

            # Evaluation
            val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold)

            print(f"Fold {fold_idx} Ergebnis: Accuracy = {val_acc:.4f}, Loss = {val_loss:.4f}")

            # Speichern
            fold_accuracies.append(val_acc)
            fold_losses.append(val_loss)
            fold_histories.append(history)
            trained_models.append(model)

        # Ergebnisse zusammenfassen
        self.results = {
            'fold_accuracies': fold_accuracies,
            'fold_losses': fold_losses,
            'fold_histories': fold_histories,
            'models': trained_models,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_loss': np.mean(fold_losses),
            'std_loss': np.std(fold_losses),
            'min_accuracy': min(fold_accuracies),
            'max_accuracy': max(fold_accuracies)
        }

        return self.results

    def print_summary(self, model_name="Model"):
        """
        Gibt eine Zusammenfassung der K-Fold Ergebnisse aus.
        """
        if self.results is None:
            print("Keine Ergebnisse vorhanden! Führen Sie zuerst run() aus.")
            return

        print(f"\n{'=' * 80}")
        print(f"K-FOLD VALIDATION ZUSAMMENFASSUNG - {model_name}")
        print(f"{'=' * 80}")

        print(f"\nEinzelne Folds:")
        for i, (acc, loss) in enumerate(zip(self.results['fold_accuracies'],
                                            self.results['fold_losses']), 1):
            print(f"  Fold {i}: Accuracy = {acc:.4f} ({acc * 100:.2f}%), Loss = {loss:.4f}")

        print(f"\nGesamtstatistik:")
        print(f"  Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"  Mean Loss:     {self.results['mean_loss']:.4f} ± {self.results['std_loss']:.4f}")
        print(f"  Min Accuracy:  {self.results['min_accuracy']:.4f}")
        print(f"  Max Accuracy:  {self.results['max_accuracy']:.4f}")
        print(f"  Range:         {self.results['max_accuracy'] - self.results['min_accuracy']:.4f}")

    def plot_results(self, model_name="Model"):
        """
        Erstellt Visualisierungen der K-Fold Ergebnisse.
        """
        if self.results is None:
            print("Keine Ergebnisse vorhanden!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Accuracy pro Fold (Bar Chart)
        ax1 = axes[0, 0]
        folds = [f"Fold {i + 1}" for i in range(self.n_splits)]
        accuracies = [acc * 100 for acc in self.results['fold_accuracies']]

        bars = ax1.bar(folds, accuracies, color='skyblue', edgecolor='black', linewidth=1.5)
        ax1.axhline(y=self.results['mean_accuracy'] * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_accuracy"] * 100:.2f}%')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Accuracy pro Fold', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Accuracy Verteilung (Box Plot)
        '''
        ax2 = axes[0, 1]
        ax2.boxplot([acc * 100 for acc in self.results['fold_accuracies']],
                    labels=['Accuracy'], widths=0.5)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model_name} - Accuracy Verteilung', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        stats_text = f"Mean: {self.results['mean_accuracy'] * 100:.2f}%\n"
        stats_text += f"Std: {self.results['std_accuracy'] * 100:.2f}%"
        ax2.text(1.15, self.results['mean_accuracy'] * 100, stats_text, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        '''
        axes[0, 1].axis('off')

        # 3. Training Curves (alle Folds)
        ax3 = axes[1, 0]
        for i, history in enumerate(self.results['fold_histories'], 1):
            if hasattr(history, 'history'):
                ax3.plot(history.history['val_accuracy'], label=f'Fold {i}', alpha=0.7)

        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title(f'{model_name} - Val Accuracy über Epochs', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(alpha=0.3)

        # 4. Loss pro Fold (Bar Chart)
        ax4 = axes[1, 1]
        losses = self.results['fold_losses']
        bars = ax4.bar(folds, losses, color='salmon', edgecolor='black', linewidth=1.5)
        ax4.axhline(y=self.results['mean_loss'], color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_loss"]:.4f}')

        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{loss:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax4.set_title(f'{model_name} - Loss pro Fold', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_best_model(self):
        """
        Gibt das Modell mit der besten Validation Accuracy zurück.

        Returns:
            best_model, best_accuracy, best_fold_idx
        """
        if self.results is None:
            print("Keine Ergebnisse vorhanden!")
            return None, None, None

        best_fold_idx = np.argmax(self.results['fold_accuracies'])
        best_model = self.results['models'][best_fold_idx]
        best_accuracy = self.results['fold_accuracies'][best_fold_idx]

        return best_model, best_accuracy, best_fold_idx + 1


def compare_kfold_results(results_list, config_names):
    """
    Vergleicht mehrere K-Fold Ergebnisse (z.B. verschiedene Modelle/Konfigurationen).

    Args:
        results_list: Liste von result Dicts aus KFoldCrossValidator.run()
        config_names: Liste von Namen für die Konfigurationen
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy Vergleich
    x = np.arange(len(config_names))
    means = [r['mean_accuracy'] * 100 for r in results_list]
    stds = [r['std_accuracy'] * 100 for r in results_list]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#FFEAA7']
    bars1 = ax1.bar(x, means, yerr=stds, capsize=10, color=colors[:len(config_names)],
                    edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, mean, std in zip(bars1, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std,
                 f'{mean:.2f}%\n±{std:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Mean Accuracy (%)', fontsize=13, fontweight='bold')
    #ax1.set_title('K-Fold Validation: Accuracy Vergleich', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # Loss Vergleich
    means_loss = [r['mean_loss'] for r in results_list]
    stds_loss = [r['std_loss'] for r in results_list]

    bars2 = ax2.bar(x, means_loss, yerr=stds_loss, capsize=10,
                    color=colors[:len(config_names)],
                    edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, mean, std in zip(bars2, means_loss, stds_loss):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + std,
                 f'{mean:.4f}\n±{std:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Mean Loss', fontsize=13, fontweight='bold')
    #ax2.set_title('K-Fold Validation: Loss Vergleich', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()