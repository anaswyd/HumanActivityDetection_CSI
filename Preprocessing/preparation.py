import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#ANzahl Dateien != anzahl im plot

class Preprocessing:
    def __init__(self, data_folder=None):
        self.data_folder = data_folder
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def load_csv_data(self):
        """
        Loads all CSV files from the specified data folder into a single pandas DataFrame.
        Adds a 'Label' column based on the filename.
        """
        if not self.data_folder or not os.path.exists(self.data_folder):
            print(f"Data folder not found or not set: {self.data_folder}")
            return pd.DataFrame()

        dataframes = []
        print(f"Scanning {self.data_folder} for CSV files...")
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.data_folder, filename)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Determine label
                    label = "Unknown"
                    if "Empty" in filename: label = "Empty"
                    elif "Lying" in filename: label = "Lying"
                    elif "Sitting" in filename: label = "Sitting"
                    elif "Standing" in filename: label = "Standing"
                    elif "Walking" in filename: label = "Walking"
                    
                    df['Label'] = label
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    def load_processed_data(self, filepath):
        """
        Loads a preprocessed CSV file into a pandas DataFrame.
        
        Args:
            filepath: Path to the processed CSV file.
            
        Returns:
            DataFrame containing the processed data.
        """
        if not os.path.exists(filepath):
            print(f"Processed file not found: {filepath}")
            return pd.DataFrame()
            
        try:
            print(f"Loading processed data from {filepath}...")
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return pd.DataFrame()

    def normalize_data(self, df):
        """
        Normalizes numeric columns using Sklearn MinMaxScaler to handle zero variance
        """
        if df.empty:
            return df

        # Identify numeric columns (excluding label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'Label' in numeric_cols:
            numeric_cols = numeric_cols.drop('Label')

        df_norm = df.copy()

        # Verwenden des Scalers statt manueller Rechnung
        scaler = MinMaxScaler()
        df_norm[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df_norm

    def save_data(self, df, output_folder, filename):
        """
        Saves the DataFrame to a CSV file.
        """
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                print(f"Error creating directory {output_folder}: {e}")
                return

        output_path = os.path.join(output_folder, filename)
        try:
            df.to_csv(output_path, index=False)
            print(f"Data successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving data to {output_path}: {e}")

    def prepare_data_for_lstm(self, df, time_steps=20, step=5, test_size=0.2):
        """
        Prepares the data for LSTM training.
        
        Args:
            df: The normalized DataFrame with a 'Label' column.
            time_steps: Number of time steps for the LSTM.
            step: Stride for the sliding window.
            test_size: Fraction of data to use for testing.
            
        Returns:
            X_train, X_test, y_train, y_test, label_encoder
        """
        if df.empty:
            return None, None, None, None, None

        print("Encoding labels...")
        # Encode labels to integers
        y_int = self.label_encoder.fit_transform(df['Label'])
        # One-hot encode the integers
        y_onehot = self.one_hot_encoder.fit_transform(y_int.reshape(-1, 1))
        
        print("Converting features to float32...")
        # Use float32 to save memory
        X = df.drop(columns=['Label']).values.astype(np.float32)
        
        print(f"Creating sequences with time_steps={time_steps} and step={step}...")
        Xs, ys = [], []
        # Use a step (stride) to reduce the number of samples and avoid memory errors
        for i in range(0, len(X) - time_steps, step):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y_onehot[i + time_steps - 1])
            
        Xs = np.array(Xs)
        ys = np.array(ys)
        
        print(f"Prepared data shape: X={Xs.shape}, y={ys.shape}")
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=42, shuffle=True)
        
        return X_train, X_test, y_train, y_test, self.label_encoder
