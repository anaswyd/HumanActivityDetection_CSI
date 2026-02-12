import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class CSI_Preprocessing:
    def __init__(self, data_folder=None):
        self.data_folder = data_folder
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

        self.label_map = {
            "Empty": "Empty",
            "Lying": "Lying",
            "Sitting": "Sitting",
            "Standing": "Standing",
            "Walking": "Walking"
        }

    def _augment_data(self, data):
        """
        Wendet Data Augmentation auf einen einzelnen Sample an.

        Args:
            data: Array der Form (timesteps, features)

        Returns:
            Augmentierter Array der gleichen Form
        """
        augmented = data.copy()

        # 1. Magnitude Scaling (0.9 - 1.1)
        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented = augmented * scale_factor

        # 2. Gaussian Noise
        if np.random.random() < 0.5:
            noise_factor = 0.01
            noise = np.random.normal(0, noise_factor, augmented.shape)
            augmented = augmented + noise

        # 3. Time Shifting (zirkulär)
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            augmented = np.roll(augmented, shift, axis=0)

        return augmented.astype(np.float32)

    def prepare_data(self, fixed_length=500, test_size=0.2, use_augmentation=False,
                     augmentation_factor=2):
        """
        Bereitet die Daten für das Training vor.

        Args:
            fixed_length: Länge auf die alle Samples gepaddet/geschnitten werden
            test_size: Anteil der Test-Daten (0.0 - 1.0)
            use_augmentation: Ob Data Augmentation verwendet werden soll
            augmentation_factor: Wie oft jeder Trainings-Sample augmentiert wird
                                (nur wenn use_augmentation=True)

        Returns:
            X_train, X_test, y_train, y_test, label_encoder
        """
        if not self.data_folder:
            print("Kein data_folder gesetzt!")
            return None, None, None, None, None

        if not os.path.exists(self.data_folder):
            print(f"Data folder existiert nicht: {self.data_folder}")
            return None, None, None, None, None

        files = [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]

        if not files:
            print(f"Keine CSV-Dateien gefunden in {self.data_folder}")
            return None, None, None, None, None

        print(f"Lade {len(files)} CSV-Dateien...")

        # Labels vorbereiten
        known_labels = sorted(list(self.label_map.values()))
        self.label_encoder.fit(known_labels)
        self.one_hot_encoder.fit(self.label_encoder.transform(known_labels).reshape(-1, 1))

        all_Xs = []
        all_ys = []

        # Alle Dateien laden und vorbereiten
        for filename in files:
            # Label aus Dateinamen extrahieren
            label_str = "Unknown"
            for key, val in self.label_map.items():
                if key in filename:
                    label_str = val
                    break

            if label_str == "Unknown":
                print(f"Warnung: Unbekanntes Label in Datei {filename}, überspringe...")
                continue

            try:
                file_path = os.path.join(self.data_folder, filename)
                df = pd.read_csv(file_path, dtype=np.float32)
                df.fillna(0, inplace=True)

                # Nur numerische Spalten verwenden
                data = df.select_dtypes(include=[np.number]).values

                # Auf fixed_length bringen (Padding oder Schneiden)
                if len(data) >= fixed_length:
                    data = data[:fixed_length]
                else:
                    padding = np.zeros((fixed_length - len(data), data.shape[1]), dtype=np.float32)
                    data = np.vstack([data, padding])

                all_Xs.append(data)
                all_ys.append(label_str)

            except Exception as e:
                print(f"Fehler beim Laden von {filename}: {e}")
                continue

        if not all_Xs:
            print("Keine Daten erfolgreich geladen!")
            return None, None, None, None, None

        X_all = np.array(all_Xs, dtype=np.float32)  # Shape: (N, fixed_length, features)
        print(f"Geladene Daten: {X_all.shape}")

        # Labels enkodieren
        y_int = self.label_encoder.transform(all_ys)
        y_final = self.one_hot_encoder.transform(y_int.reshape(-1, 1))

        # Train-Test Split (WICHTIG: vor Scaling!)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_final, test_size=test_size, random_state=42,
            shuffle=True, stratify=y_int
        )

        print(f"Split: {len(X_train)} Training, {len(X_test)} Test Samples")

        # Data Augmentation (nur auf Training Daten!)
        if use_augmentation and augmentation_factor > 0:
            print(f"Wende Data Augmentation an (Faktor: {augmentation_factor})...")
            augmented_X = []
            augmented_y = []

            for i in range(len(X_train)):
                # Original behalten
                augmented_X.append(X_train[i])
                augmented_y.append(y_train[i])

                # Augmentierte Versionen hinzufügen
                for _ in range(augmentation_factor - 1):
                    aug_sample = self._augment_data(X_train[i])
                    augmented_X.append(aug_sample)
                    augmented_y.append(y_train[i])

            X_train = np.array(augmented_X, dtype=np.float32)
            y_train = np.array(augmented_y, dtype=np.float32)
            print(f"Nach Augmentation: {len(X_train)} Training Samples")

        # Globales Scaling (WICHTIG: nur auf Train fitten, dann Test transformieren)
        print("Führe Globales Scaling durch...")
        N, T, F = X_train.shape

        # Train: Fit und Transform
        X_train_2d = X_train.reshape(N * T, F)
        self.scaler.fit(X_train_2d)
        X_train = self.scaler.transform(X_train_2d).reshape(N, T, F)

        # Test: Nur Transform
        N_test, T_test, F_test = X_test.shape
        X_test_2d = X_test.reshape(N_test * T_test, F_test)
        X_test = self.scaler.transform(X_test_2d).reshape(N_test, T_test, F_test)

        # NaN-Safety (falls konstante Spalten existieren)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        print(f"Finale Shapes: X_train={X_train.shape}, X_test={X_test.shape}")

        return X_train, X_test, y_train, y_test, self.label_encoder

    def save_processed_data(self, X_train, X_test, y_train, y_test, output_folder):
        """
        Speichert die vorbereiteten Daten als .npy Dateien.

        Args:
            X_train, X_test, y_train, y_test: Die zu speichernden Arrays
            output_folder: Ordner zum Speichern
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
        np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
        np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(output_folder, 'y_test.npy'), y_test)

        print(f"Daten gespeichert in {output_folder}")

    def load_processed_data(self, input_folder):
        """
        Lädt vorher gespeicherte vorbereitete Daten.

        Args:
            input_folder: Ordner mit den .npy Dateien

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train = np.load(os.path.join(input_folder, 'X_train.npy'))
        X_test = np.load(os.path.join(input_folder, 'X_test.npy'))
        y_train = np.load(os.path.join(input_folder, 'y_train.npy'))
        y_test = np.load(os.path.join(input_folder, 'y_test.npy'))

        print(f"Daten geladen aus {input_folder}")
        return X_train, X_test, y_train, y_test

    def prepare_data_for_kfold(self, fixed_length=500, use_augmentation=False,
                               augmentation_factor=2):
        """
        Bereitet Daten für K-Fold Cross-Validation vor (ohne Train-Test Split).

        Args:
            fixed_length: Länge auf die alle Samples normalisiert werden
            use_augmentation: Ob Data Augmentation verwendet werden soll
            augmentation_factor: Wie oft jeder Sample augmentiert wird

        Returns:
            X_all: Alle Samples (N, timesteps, features)
            y_all: Alle Labels (N, num_classes)
            label_encoder: Fitted Label Encoder
        """
        if not self.data_folder or not os.path.exists(self.data_folder):
            print(f"Data folder existiert nicht: {self.data_folder}")
            return None, None, None

        files = [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]
        if not files:
            print(f"Keine CSV-Dateien gefunden in {self.data_folder}")
            return None, None, None

        print(f"Lade {len(files)} CSV-Dateien für K-Fold Validation...")

        # Labels vorbereiten
        known_labels = sorted(list(self.label_map.values()))
        self.label_encoder.fit(known_labels)
        self.one_hot_encoder.fit(self.label_encoder.transform(known_labels).reshape(-1, 1))

        all_Xs = []
        all_ys = []

        # Alle Dateien laden
        for filename in files:
            label_str = "Unknown"
            for key, val in self.label_map.items():
                if key in filename:
                    label_str = val
                    break

            if label_str == "Unknown":
                continue

            try:
                file_path = os.path.join(self.data_folder, filename)
                df = pd.read_csv(file_path, dtype=np.float32)
                df.fillna(0, inplace=True)
                data = df.select_dtypes(include=[np.number]).values

                # Padding/Schneiden
                if len(data) >= fixed_length:
                    data = data[:fixed_length]
                else:
                    padding = np.zeros((fixed_length - len(data), data.shape[1]), dtype=np.float32)
                    data = np.vstack([data, padding])

                all_Xs.append(data)
                all_ys.append(label_str)
            except Exception as e:
                print(f"Fehler beim Laden von {filename}: {e}")
                continue

        X_all = np.array(all_Xs, dtype=np.float32)
        print(f"Geladene Daten: {X_all.shape}")

        # Labels enkodieren
        y_int = self.label_encoder.transform(all_ys)
        y_all = self.one_hot_encoder.transform(y_int.reshape(-1, 1))

        # Optional: Augmentation auf ALLE Daten
        if use_augmentation and augmentation_factor > 0:
            print(f"Wende Data Augmentation an (Faktor: {augmentation_factor})...")
            augmented_X = []
            augmented_y = []

            for i in range(len(X_all)):
                augmented_X.append(X_all[i])
                augmented_y.append(y_all[i])

                for _ in range(augmentation_factor - 1):
                    aug_sample = self._augment_data(X_all[i])
                    augmented_X.append(aug_sample)
                    augmented_y.append(y_all[i])

            X_all = np.array(augmented_X, dtype=np.float32)
            y_all = np.array(augmented_y, dtype=np.float32)
            print(f"Nach Augmentation: {len(X_all)} Samples")

        return X_all, y_all, self.label_encoder