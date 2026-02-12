import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


class LeaveOneSessionOut:
    """
    Leave-One-Session-Out Cross-Validation für CSI-Daten.

    Speziell entwickelt für Channel State Information Daten, wo:
    - Jede CSV-Datei = eine Session/Aufnahme
    - Samples innerhalb einer Session zeitlich korreliert sind
    - Realistische Evaluation bedeutet: Teste auf komplett neuen Sessions

    Vorteile gegenüber Standard K-Fold:
    - Kein Data Leakage zwischen zeitlich korrelierten Samples
    - Testet Generalisierung auf neue Aufnahme-Sessions
    - Realistisch für CSI-Deployment (neue Person, neuer Tag, neue Umgebung)
    """

    def __init__(self, data_folder, shuffle_sessions=True, random_state=42):
        """
        Initialisiert Leave-One-Session-Out Validator.

        Args:
            data_folder: Ordner mit CSV-Dateien (jede = eine Session)
            shuffle_sessions: Ob Sessions vor Split gemischt werden
            random_state: Seed für Reproduzierbarkeit
        """
        self.data_folder = data_folder
        self.shuffle_sessions = shuffle_sessions
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

    def _load_session_data(self, csv_files, fixed_length=500):
        """
        Lädt alle Sessions und hält sie getrennt.

        Returns:
            sessions_data: Liste von (X, y, filename) für jede Session
            label_encoder: Fitted Label Encoder
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        # Label Encoder vorbereiten
        known_labels = sorted(list(self.label_map.values()))
        label_encoder = LabelEncoder()
        label_encoder.fit(known_labels)

        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(label_encoder.transform(known_labels).reshape(-1, 1))

        sessions_data = []

        for csv_file in csv_files:
            # Label aus Dateiname
            label_str = "Unknown"
            for key, val in self.label_map.items():
                if key in csv_file:
                    label_str = val
                    break

            if label_str == "Unknown":
                print(f"Warnung: Unbekanntes Label in {csv_file}, überspringe...")
                continue

            try:
                import pandas as pd
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
                y_int = label_encoder.transform([label_str])
                y_onehot = one_hot_encoder.transform(y_int.reshape(-1, 1))[0]

                sessions_data.append({
                    'X': data,
                    'y': y_onehot,
                    'filename': csv_file,
                    'label': label_str
                })

            except Exception as e:
                print(f"Fehler beim Laden von {csv_file}: {e}")
                continue

        return sessions_data, label_encoder, one_hot_encoder

    def _augment_single_sample(self, data):
        """Augmentiert einen einzelnen Sample."""
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
        Führt Leave-One-Session-Out Cross-Validation durch.

        Args:
            model_class: Modell-Klasse
            model_params: Dict für Modell-Initialisierung
            training_params: Dict für Training
            fixed_length: Sequenzlänge
            use_augmentation: Ob Augmentation verwendet werden soll
            augmentation_factor: Augmentations-Faktor

        Returns:
            Dict mit Ergebnissen
        """
        print(f"\n{'=' * 80}")
        print(f"LEAVE-ONE-SESSION-OUT CROSS-VALIDATION")
        print(f"Für CSI-Daten optimiert")
        print(f"{'=' * 80}")

        # Sessions laden
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]

        if len(csv_files) == 0:
            print(f"Keine CSV-Dateien gefunden in {self.data_folder}")
            return None

        print(f"\nGefundene Sessions: {len(csv_files)}")

        # Optional: Shuffle sessions
        if self.shuffle_sessions:
            np.random.shuffle(csv_files)

        # Daten laden (Sessions getrennt halten!)
        sessions_data, label_encoder, one_hot_encoder = self._load_session_data(
            csv_files, fixed_length
        )

        if len(sessions_data) == 0:
            print("Keine gültigen Sessions geladen!")
            return None

        n_sessions = len(sessions_data)
        print(f"Erfolgreich geladene Sessions: {n_sessions}")
        print(f"Augmentation: {'Ja' if use_augmentation else 'Nein'}")

        # Ergebnisse speichern
        fold_accuracies = []
        fold_losses = []
        fold_histories = []
        trained_models = []
        session_results = []

        # Leave-One-Session-Out
        for test_idx in range(n_sessions):
            print(f"\n--- Session {test_idx + 1}/{n_sessions} als Test ---")

            # Test Session
            test_session = sessions_data[test_idx]
            X_test = test_session['X'].reshape(1, *test_session['X'].shape)
            y_test = test_session['y'].reshape(1, -1)

            # Training Sessions (alle außer test_idx)
            train_sessions = [sessions_data[i] for i in range(n_sessions) if i != test_idx]

            # Kombiniere Training Sessions
            X_train_list = [s['X'] for s in train_sessions]
            y_train_list = [s['y'] for s in train_sessions]

            X_train = np.array(X_train_list, dtype=np.float32)
            y_train = np.array(y_train_list, dtype=np.float32)

            print(f"Test Session: {test_session['filename']} (Label: {test_session['label']})")
            print(f"Train: {len(X_train)} sessions, Test: 1 session")

            # Optional: Augmentation nur auf Training
            if use_augmentation:
                print(f"Wende Augmentation an (Faktor: {augmentation_factor})...")
                augmented_X = []
                augmented_y = []

                for i in range(len(X_train)):
                    # Original
                    augmented_X.append(X_train[i])
                    augmented_y.append(y_train[i])

                    # Augmentierte Versionen
                    for _ in range(augmentation_factor - 1):
                        aug_sample = self._augment_single_sample(X_train[i])
                        augmented_X.append(aug_sample)
                        augmented_y.append(y_train[i])

                X_train = np.array(augmented_X, dtype=np.float32)
                y_train = np.array(augmented_y, dtype=np.float32)
                print(f"Nach Augmentation: {len(X_train)} training samples")

            # Scaling (nur auf Train fitten!)
            scaler = StandardScaler()
            N, T, F = X_train.shape

            X_train_2d = X_train.reshape(N * T, F)
            scaler.fit(X_train_2d)
            X_train = scaler.transform(X_train_2d).reshape(N, T, F)

            # Test: nur transformieren
            N_test, T_test, F_test = X_test.shape
            X_test = scaler.transform(X_test.reshape(N_test * T_test, F_test)).reshape(N_test, T_test, F_test)

            # NaN-Safety
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            # Modell erstellen und trainieren
            model = model_class(**model_params)

            history = model.train(
                X_train, y_train,
                X_test, y_test,  # Validation auf Test Session
                **training_params
            )

            # Evaluation
            test_loss, test_acc = model.evaluate(X_test, y_test)

            print(f"Session {test_idx + 1} Ergebnis: Accuracy = {test_acc:.4f}, Loss = {test_loss:.4f}")

            # Speichern
            fold_accuracies.append(test_acc)
            fold_losses.append(test_loss)
            fold_histories.append(history)
            trained_models.append(model)
            session_results.append({
                'session_idx': test_idx,
                'filename': test_session['filename'],
                'label': test_session['label'],
                'accuracy': test_acc,
                'loss': test_loss
            })

        # Ergebnisse zusammenfassen
        self.results = {
            'fold_accuracies': fold_accuracies,
            'fold_losses': fold_losses,
            'fold_histories': fold_histories,
            'models': trained_models,
            'session_results': session_results,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_loss': np.mean(fold_losses),
            'std_loss': np.std(fold_losses),
            'min_accuracy': min(fold_accuracies),
            'max_accuracy': max(fold_accuracies),
            'n_sessions': n_sessions
        }

        return self.results

    def print_summary(self, model_name="Model"):
        """Gibt eine Zusammenfassung der LOSO-Ergebnisse aus."""
        if self.results is None:
            print("Keine Ergebnisse vorhanden!")
            return

        print(f"\n{'=' * 80}")
        print(f"LEAVE-ONE-SESSION-OUT ZUSAMMENFASSUNG - {model_name}")
        print(f"{'=' * 80}")

        print(f"\nPro Session:")
        for sr in self.results['session_results']:
            print(f"  Session {sr['session_idx'] + 1} ({sr['filename']}): "
                  f"Acc = {sr['accuracy']:.4f}, Loss = {sr['loss']:.4f}, Label = {sr['label']}")

        print(f"\nGesamtstatistik:")
        print(f"  Mean Accuracy: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"  Mean Loss:     {self.results['mean_loss']:.4f} ± {self.results['std_loss']:.4f}")
        print(f"  Min Accuracy:  {self.results['min_accuracy']:.4f}")
        print(f"  Max Accuracy:  {self.results['max_accuracy']:.4f}")
        print(f"  Range:         {self.results['max_accuracy'] - self.results['min_accuracy']:.4f}")

        # Analyse nach Labels
        print(f"\nPerformance nach Label:")
        label_accs = {}
        for sr in self.results['session_results']:
            label = sr['label']
            if label not in label_accs:
                label_accs[label] = []
            label_accs[label].append(sr['accuracy'])

        for label, accs in sorted(label_accs.items()):
            mean_acc = np.mean(accs)
            print(f"  {label}: {mean_acc:.4f} ({len(accs)} sessions)")

    def plot_results(self, model_name="Model"):
        """Erstellt Visualisierungen der LOSO-Ergebnisse."""
        if self.results is None:
            print("Keine Ergebnisse vorhanden!")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Accuracy pro Session (Bar Chart)
        ax1 = fig.add_subplot(gs[0, :])
        sessions = [f"S{sr['session_idx'] + 1}\n{sr['label'][:3]}"
                    for sr in self.results['session_results']]
        accuracies = [sr['accuracy'] * 100 for sr in self.results['session_results']]

        # Farben nach Label
        colors_map = {'Empty': '#95a5a6', 'Lying': '#3498db', 'Sitting': '#e74c3c',
                      'Standing': '#2ecc71', 'Walking': '#f39c12'}
        colors = [colors_map.get(sr['label'], '#34495e')
                  for sr in self.results['session_results']]

        bars = ax1.bar(range(len(sessions)), accuracies, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=self.results['mean_accuracy'] * 100, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_accuracy"] * 100:.2f}%')

        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Accuracy pro Session (LOSO)',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(sessions)))
        ax1.set_xticklabels(sessions, fontsize=8, rotation=0)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 110])

        # 2. Box Plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.boxplot([acc * 100 for acc in self.results['fold_accuracies']],
                    labels=['Accuracy'], widths=0.5)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy Verteilung', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Performance nach Label
        ax3 = fig.add_subplot(gs[1, 1])
        label_accs = {}
        for sr in self.results['session_results']:
            label = sr['label']
            if label not in label_accs:
                label_accs[label] = []
            label_accs[label].append(sr['accuracy'] * 100)

        labels = sorted(label_accs.keys())
        means = [np.mean(label_accs[l]) for l in labels]
        colors_bar = [colors_map.get(l, '#34495e') for l in labels]

        bars = ax3.bar(labels, means, color=colors_bar,
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax3.set_ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Performance nach Aktivität', fontsize=13, fontweight='bold')
        ax3.set_xticklabels(labels, rotation=15, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 110])

        # 4. Loss pro Session
        ax4 = fig.add_subplot(gs[2, :])
        losses = [sr['loss'] for sr in self.results['session_results']]
        bars = ax4.bar(range(len(sessions)), losses, color=colors,
                       edgecolor='black', linewidth=1.5, alpha=0.8)
        ax4.axhline(y=self.results['mean_loss'], color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {self.results["mean_loss"]:.4f}')

        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{loss:.3f}', ha='center', va='bottom', fontsize=8)

        ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax4.set_title(f'{model_name} - Loss pro Session (LOSO)', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(sessions)))
        ax4.set_xticklabels(sessions, fontsize=8, rotation=0)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.show()

    def get_worst_sessions(self, n=3):
        """
        Gibt die N schlechtesten Sessions zurück (für Analyse).

        Returns:
            Liste von Dicts mit Session-Info
        """
        if self.results is None:
            return None

        sorted_sessions = sorted(self.results['session_results'],
                                 key=lambda x: x['accuracy'])
        return sorted_sessions[:n]