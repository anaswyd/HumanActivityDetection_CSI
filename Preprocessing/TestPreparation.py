import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import gc


class Preprocessing:
    def __init__(self, data_folder=None):
        self.data_folder = data_folder
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

        self.label_map = {
            "Empty": "Empty", "Lying": "Lying", "Sitting": "Sitting",
            "Standing": "Standing", "Walking": "Walking"
        }

    def prepare_data(self, fixed_length=500, test_size=0.2):
        if not self.data_folder: return None, None, None, None, None

        files = [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]
        print(f"Lade {len(files)} Dateien...")

        # Labels vorbereiten
        known_labels = sorted(list(self.label_map.values()))
        self.label_encoder.fit(known_labels)
        self.one_hot_encoder.fit(self.label_encoder.transform(known_labels).reshape(-1, 1))

        all_Xs = []
        all_ys = []

        # 1. Alles Laden (Stapeln)
        for i, filename in enumerate(files):
            label_str = "Unknown"
            for key, val in self.label_map.items():
                if key in filename:
                    label_str = val
                    break
            if label_str == "Unknown": continue

            try:
                df = pd.read_csv(os.path.join(self.data_folder, filename), dtype=np.float32)
                df.fillna(0, inplace=True)
                data = df.select_dtypes(include=[np.number]).values

                # Padding auf 500
                if len(data) >= fixed_length:
                    data = data[:fixed_length]
                else:
                    padding = np.zeros((fixed_length - len(data), data.shape[1]), dtype=np.float32)
                    data = np.vstack([data, padding])

                all_Xs.append(data)
                all_ys.append(label_str)
            except:
                pass

        X_all = np.array(all_Xs, dtype=np.float32)  # (N, 500, 256)

        # Labels
        y_int = self.label_encoder.transform(all_ys)
        y_final = self.one_hot_encoder.transform(y_int.reshape(-1, 1))

        # Split ZUERST (Wichtig für sauberes Scaling)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_final, test_size=test_size, random_state=42, shuffle=True, stratify=y_int
        )

        print("Führe Globales Scaling durch ...")
        # TRICK DES PROFS: 3D -> 2D -> Scaling -> 3D
        # Wir fitten NUR auf Train-Daten (Datenleck vermeiden!)
        N, T, F = X_train.shape
        X_train_2d = X_train.reshape(N * T, F)
        self.scaler.fit(X_train_2d)

        # Transformieren (Train)
        X_train = self.scaler.transform(X_train_2d).reshape(N, T, F)

        # Transformieren (Test)
        N_test, T_test, F_test = X_test.shape
        X_test = self.scaler.transform(X_test.reshape(N_test * T_test, F_test)).reshape(N_test, T_test, F_test)

        # NaN-Safety (StandardScaler mag konstante Spalten nicht)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        return X_train, X_test, y_train, y_test, self.label_encoder