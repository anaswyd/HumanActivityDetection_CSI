import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import os
import pandas as pd


class CSIGroupKFold:
    """
    Group K-Fold Cross-Validation für CSI-Daten.

    Löst das Problem von LOSO wenn nur wenige Sessions pro Klasse vorhanden sind.

    Funktionsweise:
    - Gruppiert Sessions (verhindert Data Leakage)
    - Stellt sicher dass alle Klassen in jedem Fold sind
    - Besser als LOSO für kleine Datasets
    - Besser als Standard K-Fold (respektiert Sessions)
    """

    def __init__(self, data_folder, n_splits=5, shuffle=True, random_state=42):
        """
        Args:
            data_folder: Ordner mit CSV-Dateien
            n_splits: Anzahl Folds
            shuffle: Ob Sessions gemischt werden
            random_state: Seed für Reproduzierbarkeit
        """
        self.data_folder = data_folder
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.results = None

        self.label_map = {
            "Empty": "Empty",
            "Lying": "Lying",
            "Sitting": "Sitting",
            "Standing": "Standing",
            "Walking": "Walking"
        }

        np.random.seed(random_state)

    def _load_sessions_grouped(self, csv_files, fixed_length=500):
        """
        Lädt alle Sessions und erstellt Sample-zu-Session Mapping.

        Returns:
            X_all: Alle Samples
            y_all: Alle Labels
            groups: Session-Zuordnung für jeden Sample
            session_info: Metadaten
        """
        label_encoder = LabelEncoder()
        known_labels = sorted(list(self.label_map.values()))
        label_encoder.fit(known_labels)

        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(label_encoder.transform(known_labels).reshape(-1, 1))

        all_X = []
        all_y = []
        all_groups = []
        session_info = []

        session_id = 0

        for csv_file in csv_files:
            # Label aus Dateiname
            label_str = "Unknown"
            for key, val in self.label_map.items():
                if key in csv_file:
                    label_str = val
                    break

            if label_str == "Unknown":
                print(f"Warnung: Unbekanntes Label in {csv_file}")
                continue

            try:
                file_path = os.path.join(self.data_folder, csv_file)
                df = pd.read_csv(file_path, dtype=np.float32)
                df.fillna(0, inplace=True)

                data = df.select_dtypes(include=[np.number]).values

                # Padding/Schneiden
                if len(data) >= fixed_length:
                    data = data[:fixed_length]
                else:
                    padding = np.zeros((fixed_length - len(data), data.shape[1]), dtype=np.float32)
                    data = np.vstack([data, padding])

                # Label enkodieren
                y_int = label_encoder.transform([label_str])[0]
                y_onehot = one_hot_encoder.transform([[y_int]])[0]

                # Speichern
                all_X.append(data)
                all_y.append(y_onehot)
                all_groups.append(session_id)

                session_info.append({
                    'session_id': session_id,
                    'filename': csv_file,
                    'label': label_str
                })

                session_id += 1

            except Exception as e:
                print(f"Fehler bei {csv_file}: {e}")
                continue

        X_all = np.array(all_X, dtype=np.float32)
        y_all = np.array(all_y, dtype=np.float32)
        groups = np.array(all_groups)

        return X_all, y_all, groups, session_info, label_encoder

    def _augment_single_sample(self, data):
        """Augmentiert einen Sample."""
        augmented = data.copy()

        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale_factor

        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, augmented.shape)
            augmented = augmented + noise

        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            augmented = np.roll(augmented, shift, axis=0)

        return augmented.astype(np.float32)

    def run(self, model_class, model_params, training_params,
            fixed_length=500, use_augmentation=False, augmentation_factor=2):
        """
        Führt Group K-Fold Cross-Validation durch.

        Args:
            model_class: Modell-Klasse
            model_params: Dict für Modell-Init
            training_params: Dict für Training
            fixed_length: Sequenzlänge
            use_augmentation: Augmentation aktivieren
            augmentation_factor: Augmentations-Faktor

        Returns:
            Dict mit Ergebnissen
        """
        print(f"\n{'=' * 80}")
        print(f"CSI GROUP K-FOLD CROSS-VALIDATION")
        print(f"Sessions gruppiert, kein Data Leakage")
        print(f"{'=' * 80}")

        # Sessions laden
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]

        if len(csv_files) == 0:
            print("Keine CSV-Dateien gefunden!")
            return None

        print(f"\nGefundene Sessions: {len(csv_files)}")

        # Optional: Shuffle
        if self.shuffle:
            np.random.shuffle(csv_files)

        # Daten laden (gruppiert)
        X_all, y_all, groups, session_info, label_encoder = self._load_sessions_grouped(
            csv_files, fixed_length
        )

        if len(X_all) == 0:
            print("Keine gültigen Daten geladen!")
            return None

        print(f"Geladene Sessions: {len(X_all)}")
        print(f"Folds: {self.n_splits}")
        print(f"Augmentation: {'Ja' if use_augmentation else 'Nein'}")

        # Prüfe Sessions pro Label
        print(f"\nSessions pro Label:")
        label_counts = {}
        for info in session_info:
            label = info['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} sessions")

        # GroupKFold
        gkf = GroupKFold(n_splits=self.n_splits)

        fold_accuracies = []
        fold_losses = []
        fold_histories = []
        trained_models = []
        fold_details = []

        # K-Fold durchführen
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups), 1):
            print(f"\n--- Fold {fold_idx}/{self.n_splits} ---")

            X_train_fold = X_all[train_idx]
            X_val_fold = X_all[val_idx]
            y_train_fold = y_all[train_idx]
            y_val_fold = y_all[val_idx]

            # Zeige welche Sessions in diesem Fold sind
            train_sessions = [session_info[i]['filename'] for i in train_idx]
            val_sessions = [session_info[i]['filename'] for i in val_idx]

            print(f"Train Sessions ({len(train_sessions)}): {', '.join(train_sessions[:3])}...")
            print(f"Val Sessions ({len(val_sessions)}): {', '.join(val_sessions)}")

            # Prüfe ob alle Labels im Training sind
            train_labels = set([session_info[i]['label'] for i in train_idx])
            val_labels = set([session_info[i]['label'] for i in val_idx])
            print(f"Train Labels: {train_labels}")
            print(f"Val Labels: {val_labels}")

            if len(train_labels) != len(label_encoder.classes_):
                print(f"⚠️  Warnung: Nicht alle Klassen im Training! "
                      f"({len(train_labels)}/{len(label_encoder.classes_)})")

            # Optional: Augmentation
            if use_augmentation:
                print(f"Wende Augmentation an (Faktor: {augmentation_factor})...")
                augmented_X = []
                augmented_y = []

                for i in range(len(X_train_fold)):
                    augmented_X.append(X_train_fold[i])
                    augmented_y.append(y_train_fold[i])

                    for _ in range(augmentation_factor - 1):
                        aug_sample = self._augment_single_sample(X_train_fold[i])
                        augmented_X.append(aug_sample)
                        augmented_y.append(y_train_fold[i])

                X_train_fold = np.array(augmented_X, dtype=np.float32)
                y_train_fold = np.array(augmented_y, dtype=np.float32)
                print(f"Nach Augmentation: {len(X_train_fold)} training samples")

            # Scaling
            scaler = StandardScaler()
            N, T, F = X_train_fold.shape

            X_train_2d = X_train_fold.reshape(N * T, F)
            scaler.fit(X_train_2d)
            X_train_fold = scaler.transform(X_train_2d).reshape(N, T, F)

            N_val, T_val, F_val = X_val_fold.shape
            X_val_fold = scaler.transform(X_val_fold.reshape(N_val * T_val, F_val)).reshape(N_val, T_val, F_val)

            # NaN-Safety
            X_train_fold = np.nan_to_num(X_train_fold, nan=0.0)
            X_val_fold = np.nan_to_num(X_val_fold, nan=0.0)

            # Modell trainieren
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
            fold_details.append({
                'fold': fold_idx,
                'train_sessions': train_sessions,
                'val_sessions': val_sessions,
                'accuracy': val_acc,
                'loss': val_loss
            })

        # Ergebnisse zusammenfassen
        self.results = {
            'fold_accuracies': fold_accuracies,
            'fold_losses': fold_losses,
            'fold_histories': fold_histories,
            'models': trained_models,
            'fold_details': fold_details,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_loss': np.mean(fold_losses),
            'std_loss': np.std(fold_losses),
            'min_accuracy': min(fold_accuracies),
            'max_accuracy': max(fold_accuracies),
            'n_folds': self.n_splits
        }

        return self.results

    def print_summary(self, model_name="Model"):
        """Gibt Zusammenfassung aus."""
        if self.results is None:
            print("Keine Ergebnisse!")
            return

        print(f"\n{'=' * 80}")
        print(f"CSI GROUP K-FOLD ZUSAMMENFASSUNG - {model_name}")
        print(f"{'=' * 80}")

        print(f"\nPro Fold:")
        for detail in self.results['fold_details']:
            print(f"  Fold {detail['fold']}: Acc = {detail['accuracy']:.4f}, "
                  f"Loss = {detail['loss']:.4f}")
            print(f"    Val Sessions: {', '.join(detail['val_sessions'])}")

        print(f"\nGesamtstatistik:")
        print(f"  Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"  Mean Loss:     {self.results['mean_loss']:.4f} ± {self.results['std_loss']:.4f}")
        print(f"  Min Accuracy:  {self.results['min_accuracy']:.4f}")
        print(f"  Max Accuracy:  {self.results['max_accuracy']:.4f}")
        print(f"  Range:         {self.results['max_accuracy'] - self.results['min_accuracy']:.4f}")

    def plot_results(self, model_name="Model"):
        """Erstellt Visualisierungen."""
        if self.results is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Accuracy pro Fold
        ax1 = axes[0, 0]
        folds = [f"Fold {i + 1}" for i in range(self.n_splits)]
        accuracies = [acc * 100 for acc in self.results['fold_accuracies']]

        bars = ax1.bar(folds, accuracies, color='skyblue', edgecolor='black', linewidth=1.5)
        ax1.axhline(y=self.results['mean_accuracy'] * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_accuracy"] * 100:.2f}%')

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Accuracy pro Fold (Group K-Fold)',
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 110])

        # 2. Box Plot
        ax2 = axes[0, 1]
        ax2.boxplot([acc * 100 for acc in self.results['fold_accuracies']],
                    labels=['Accuracy'], widths=0.5)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy Verteilung', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Training Curves
        ax3 = axes[1, 0]
        for i, history in enumerate(self.results['fold_histories'], 1):
            if hasattr(history, 'history'):
                ax3.plot(history.history['val_accuracy'], label=f'Fold {i}', alpha=0.7)

        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Val Accuracy über Epochs', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(alpha=0.3)

        # 4. Loss pro Fold
        ax4 = axes[1, 1]
        losses = self.results['fold_losses']
        bars = ax4.bar(folds, losses, color='salmon', edgecolor='black', linewidth=1.5)
        ax4.axhline(y=self.results['mean_loss'], color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_loss"]:.4f}')

        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{loss:.4f}', ha='center', va='bottom', fontsize=9)

        ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax4.set_title(f'{model_name} - Loss pro Fold', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()