import joblib
import numpy as np
import os

# --- KONFIGURATION ---
# Hier den Namen deiner Datei eintragen!
# War es "csi_data_filtered.pkl" oder "csi_data_walking.pkl"?
INPUT_FILE = "csi_data_filtered.pkl" 
OUTPUT_CSV = "mein_experiment_export.csv"

def convert_pkl_to_csv():
    print(f"Suche nach {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"FEHLER: Datei '{INPUT_FILE}' nicht gefunden!")
        return

    # 1. Alte Daten laden
    data = joblib.load(INPUT_FILE)
    
    # Prüfen, ob das Format stimmt (Dictionary mit X und y)
    if isinstance(data, dict) and 'X' in data and 'y' in data:
        X = np.array(data['X'])
        y = np.array(data['y']).reshape(-1, 1) # Label als Spalte formen
        
        print(f"Daten geladen: {len(X)} Zeilen gefunden.")
        print(f"Anzahl Features pro Zeile: {X.shape[1]}")
        
        # 2. Automatisch Header generieren (egal ob 128 oder 192 Features)
        num_features = X.shape[1]
        header_parts = []
        
        # Logik: Wir versuchen zu erraten, was was ist, basierend auf der Anzahl
        if num_features == 192:
            # Das ist das NEUE Format (Mean, Std, Diff)
            print("Format erkannt: Mean + Std + Diff (192 Spalten)")
            for i in range(64): header_parts.append(f"Subcarrier_{i}_Mean")
            for i in range(64): header_parts.append(f"Subcarrier_{i}_Std")
            for i in range(64): header_parts.append(f"Subcarrier_{i}_Diff")
        elif num_features == 128:
            # Das ist das ALTE Format (Mean + Std)
            print("Format erkannt: Mean + Std (128 Spalten)")
            for i in range(64): header_parts.append(f"Subcarrier_{i}_Mean")
            for i in range(64): header_parts.append(f"Subcarrier_{i}_Std")
        else:
            # Fallback
            print(f"Unbekanntes Format ({num_features} Spalten). Nummeriere einfach durch.")
            for i in range(num_features): header_parts.append(f"Feature_{i}")
            
        header_parts.append("Label_Activity")
        header_str = ",".join(header_parts)
        
        # 3. Daten zusammenfügen
        full_data = np.hstack([X, y])
        
        # 4. Speichern
        print(f"Speichere als {OUTPUT_CSV}...")
        np.savetxt(OUTPUT_CSV, full_data, delimiter=",", header=header_str, comments='', fmt='%.6f')
        print("FERTIG! Jetzt kannst du die CSV in Excel öffnen.")
        
    else:
        print("Fehler: Das Format der PKL-Datei ist unerwartet.")

if __name__ == "__main__":
    convert_pkl_to_csv()
