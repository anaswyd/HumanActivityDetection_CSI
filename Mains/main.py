import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing.preparation import Preprocessing
from models.simple_model import SimpleModel

#32298963

def main():
    # Define paths
    base_dir = os.path.dirname(__file__)
    data_folder = os.path.join(base_dir, 'Data')
    output_folder = os.path.join(base_dir, 'ProcessedData')
    processed_file = os.path.join(output_folder, 'normalized_data.csv')
    
    # Parameters
    TIME_STEPS = 30
    STEP = 20  # Stride for sliding window
    EPOCHS = 20
    BATCH_SIZE = 64
    
    # Initialize Preprocessing
    preprocessor = Preprocessing(data_folder)

    # --- Option 1: Load Raw Data and Process ---
    print("--- Loading Raw Data ---")
    raw_df = preprocessor.load_csv_data()
    
    if raw_df.empty:
        print("No raw data found. Checking for processed data...")
        df_to_process = pd.DataFrame()
    else:
        print(f"Raw data shape: {raw_df.shape}")
        print("Normalizing raw data...")
        normalized_df = preprocessor.normalize_data(raw_df)
        
        # We can use this directly
        df_to_process = normalized_df
        
        # Optionally save for future use/comparison
        # print("Saving processed data for comparison...")
        # preprocessor.save_data(normalized_df, output_folder, 'normalized_data.csv')

    # --- Option 2: Load Processed Data (For Comparison) ---
    # This block demonstrates loading the saved data if it exists
    if os.path.exists(processed_file):
        print("\n--- Loading Processed Data (Comparison) ---")
        loaded_processed_df = preprocessor.load_processed_data(processed_file)
        print(f"Loaded processed data shape: {loaded_processed_df.shape}")
        
        # If we didn't load raw data, use this
        if df_to_process.empty:
            df_to_process = loaded_processed_df
    
    # --- Model Training ---
    if df_to_process.empty:
        print("No data available for training. Exiting.")
        return

    print("\n--- Preparing Data for LSTM ---")
    X_train, X_test, y_train, y_test, label_encoder = preprocessor.prepare_data_for_lstm(
        df_to_process, time_steps=TIME_STEPS, step=STEP
    )
    
    if X_train is None:
        print("Failed to prepare data.")
        return

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Classes: {label_encoder.classes_}")
    
    print("\n--- Initializing and Training Model ---")
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_encoder.classes_)

    print("Checking for NaNs in X_train:", np.isnan(X_train).sum())
    print("Checking for Infs in X_train:", np.isinf(X_train).sum())

    #Notfall-Bereinigung falls immer noch NaNs da sind
    if np.isnan(X_train).sum() > 0:
        print("Warnung: NaNs gefunden! Ersetze durch 0...")
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
    
    simple_model = SimpleModel(input_shape, num_classes)
    
    history = simple_model.train(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #plot training results:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #plot loss value
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    print("\n--- Evaluating Model ---")
    simple_model.evaluate(X_test, y_test)
    
    print("Pipeline finished.")


#Für einzelne auch acc berechnen & anzeigen lassen..

if __name__ == "__main__":
    main()
