import numpy as np
import matplotlib.pyplot as plt
import serial
import threading
import time
import os
import joblib
import queue
from collections import deque, Counter
from scipy.signal import medfilt, butter, lfilter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# --- KONFIGURATION ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200 
COLLECTION_TIME_SEC = 60 * 10  # 4 Minuten
MODEL_FILE = "csi_model_filtered.pkl"
DATA_FILE = "csi_data_filtered.pkl"   

# Filter-Einstellungen
WINDOW_SIZE = 50        # Sliding Window Größe
BUTTER_ORDER = 4        
BUTTER_CUTOFF = 0.1     

# Live-Performance Settings
PLOT_EVERY_N = 3        # Nur jedes 3. Frame plotten (GPU-Schonend)
PREDICTION_SMOOTH = 5   # Letzte N Predictions für Glättung
CONFIDENCE_THRESHOLD = 0.6  # Minimum Konfidenz für Anzeige

CLASS_NAMES = {
    0: "Raum Leer",
    1: "Sitzen",
    2: "Gehen"
}

# --- FILTER FUNKTIONEN ---
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

class CSIFingerprintAI:
    def __init__(self):
        self.scaler = StandardScaler()
        # Größeres Netzwerk mit Regularisierung
        self.model = MLPClassifier(
            hidden_layer_sizes=(150, 100, 50), 
            max_iter=2000,
            alpha=0.001,  # Regularisierung gegen Overfitting
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        self.is_trained = False
        
        # Prediction Smoothing Buffer
        self.prediction_buffer = deque(maxlen=PREDICTION_SMOOTH)

    def save_model(self):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, MODEL_FILE)
        print(f"✓ KI-Modell gespeichert.")

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                data = joblib.load(MODEL_FILE)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
                print(f"✓ Modell geladen.")
                return True
            except Exception as e:
                print(f"✗ Fehler beim Laden: {e}")
        return False

    def train(self, X_features, y):
        print(f"\n--- Training mit {len(X_features)} Feature-Paketen ---")
        
        # Train-Test Split für realistische Genauigkeit
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, 
                test_size=0.2, 
                stratify=y, 
                random_state=42
            )
        except ValueError:
            # Zu wenig Daten für Split
            print("⚠ Zu wenig Daten für Validation-Split, nutze alle Daten.")
            X_train, y_train = X_features, y
            X_test, y_test = X_features, y
        
        # Skalierung
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Training
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy:    {train_score*100:.1f}%")
        print(f"Test Accuracy:        {test_score*100:.1f}%")
        
        # Cross-Validation (wenn genug Daten)
        if len(X_train) >= 30:
            try:
                cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=min(5, len(X_train)//6))
                print(f"Cross-Validation:     {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
            except:
                pass
        
        self.is_trained = True
        
        # Warnung bei Overfitting
        if train_score - test_score > 0.15:
            print("⚠ Warnung: Mögliches Overfitting! Sammle mehr Daten.")
        
        self.save_model()

    def predict_smooth(self, feature_vector):
        """Prediction mit temporaler Glättung"""
        if not self.is_trained: 
            return None, 0.0
        
        scaled_vec = self.scaler.transform(feature_vector.reshape(1, -1))
        pred = self.model.predict(scaled_vec)[0]
        probs = self.model.predict_proba(scaled_vec)[0]
        prob = np.max(probs)
        
        # Buffer füllen
        self.prediction_buffer.append(pred)
        
        # Mehrheitsentscheidung (verhindert Flackern)
        if len(self.prediction_buffer) >= 3:
            smooth_pred = Counter(self.prediction_buffer).most_common(1)[0][0]
            return smooth_pred, prob, probs
        
        return pred, prob, probs

class CSIReader:
    def __init__(self):
        self.latest_complex = None 
        self.running = True
        
        # Puffer für das Sliding Window
        self.window_buffer = deque(maxlen=WINDOW_SIZE)
        
        # Thread-safe Queue für Features
        self.feature_queue = queue.Queue(maxsize=5)
        
        self.training_features = []
        self.training_labels = []
        self.collection_target = None
        self.collection_start_time = 0

    def get_features(self):
        """
        Verbesserte Feature-Extraktion:
        1. Mittelwert (Raumprofil)
        2. Standardabweichung (Bewegung/Atmung)
        3. Zeitliche Differenz (Bewegungsrate) - NEU!
        """
        if len(self.window_buffer) < WINDOW_SIZE:
            return None
        
        data_chunk = np.array(self.window_buffer)
        
        # Feature 1: Mittelwert (statisches Profil)
        feat_mean = np.mean(data_chunk, axis=0)
        
        # Feature 2: Standardabweichung (Variabilität)
        feat_std = np.std(data_chunk, axis=0)
        
        # Feature 3: Zeitliche Änderungsrate (Bewegungsdetektion)
        feat_diff = np.mean(np.abs(np.diff(data_chunk, axis=0)), axis=0)
        
        # Kombinieren: 64 + 64 + 64 = 192 Features
        combined = np.hstack([
            feat_mean,
            feat_std * 5,    # Gewichtet
            feat_diff * 10   # Bewegung ist wichtig
        ])
        return combined

    def save_data(self):
        if len(self.training_features) > 0:
            joblib.dump({'X': self.training_features, 'y': self.training_labels}, DATA_FILE)
            print(f"✓ {len(self.training_features)} Samples gespeichert!")

    def load_data(self):
        if os.path.exists(DATA_FILE):
            try:
                raw = joblib.load(DATA_FILE)
                self.training_features = raw['X']
                self.training_labels = raw['y']
                print(f"✓ {len(self.training_features)} Samples geladen.")
                return True
            except Exception as e:
                print(f"✗ Fehler beim Laden: {e}")
        return False

    def start(self): 
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        """Serial Thread - läuft parallel zur GUI"""
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            ser.reset_input_buffer()
            print(f"✓ Serial Port {SERIAL_PORT} geöffnet.")
        except Exception as e:
            print(f"✗ Serial Fehler: {e}")
            return

        while self.running:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "CSI_DATA" in line and "[" in line:
                    content = line.split("[")[1].split("]")[0]
                    raw = np.fromstring(content, dtype=int, sep=' ')
                    
                    if len(raw) >= 128:
                        data = raw[:128]
                        c_frame = data[0::2] + 1j * data[1::2]
                        self.latest_complex = c_frame
                        
                        # Amplitude + Median Filter
                        amp = np.abs(c_frame)
                        amp_clean = medfilt(amp, 5)  # Größerer Kernel
                        
                        # Amplitude + Median Filter
                        amp = np.abs(c_frame)
                        amp_clean = medfilt(amp, 5)  # Größerer Kernel für Glättung
                        
                        # --- ÄNDERUNG: KEINE PRO-FRAME NORMALISIERUNG ---
                        # Wir wollen die absolute Signalstärke behalten!
                        # Denn: Sitzen = Signal wird schwächer. Das ist wichtig!
                        
                        # amp_mean = np.mean(amp_clean)       <-- WEG
                        # amp_std = np.std(amp_clean)         <-- WEG
                        # if amp_std > 0:                     <-- WEG
                        #     amp_norm = (amp_clean - amp_mean) / amp_std
                        # else:
                        #     amp_norm = amp_clean
                        
                        # HIER direkt die geglätteten Rohdaten nehmen:
                        self.window_buffer.append(amp_clean)

                        # Features berechnen und in Queue schieben
                        if len(self.window_buffer) == WINDOW_SIZE:
                            features = self.get_features()
                            
                            # Training-Daten sammeln
                            if features is not None and self.collection_target is not None:
                                self.training_features.append(features)
                                self.training_labels.append(self.collection_target)
                            
                            # Features für Live-Prediction in Queue
                            if features is not None:
                                try:
                                    self.feature_queue.put_nowait(features)
                                except queue.Full:
                                    pass  # Alte verwerfen
                                
            except Exception as e:
                continue

def main():
    reader = CSIReader()
    reader.start()
    brain = CSIFingerprintAI()
    
    reader.load_data()
    brain.load_model()

    plt.ion()
    fig, (ax_amp, ax_var, ax_prob) = plt.subplots(3, 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.5)

    # Plot 1: Gemittelte Amplitude
    line_amp, = ax_amp.plot(np.zeros(64), 'b-', linewidth=2)
    ax_amp.set_title("Gemittelte Amplitude (Raumprofil)", fontsize=12, weight='bold')
    ax_amp.set_ylim(-2, 2)
    ax_amp.grid(True, alpha=0.3)

    # Plot 2: Varianz (Bewegungsenergie)
    line_var, = ax_var.plot(np.zeros(64), 'r-', linewidth=2)
    ax_var.set_title("Signal-Varianz (Bewegungsenergie)", fontsize=12, weight='bold')
    ax_var.set_ylim(0, 2)
    ax_var.grid(True, alpha=0.3)

    # Plot 3: KI Predictions
    bars = ax_prob.bar(CLASS_NAMES.values(), [0]*len(CLASS_NAMES), color='gray')
    ax_prob.set_ylim(0, 1.0)
    ax_prob.set_title("Erkannte Aktivität", fontsize=12, weight='bold')
    ax_prob.axhline(y=CONFIDENCE_THRESHOLD, color='orange', linestyle='--', label=f'Schwellwert ({CONFIDENCE_THRESHOLD})')
    ax_prob.legend()
    ax_prob.grid(True, alpha=0.3, axis='y')

    state_text = plt.figtext(0.5, 0.96, "BEREIT", ha="center", fontsize=14, weight="bold")

    if brain.is_trained:
        state_text.set_text("✓ Modell geladen und bereit")
        state_text.set_color("green")

    print("\n" + "="*50)
    print("       CSI BEWEGUNGSERKENNUNG - STEUERUNG")
    print("="*50)
    print(f"[0] → Raum LEER sammeln   ({COLLECTION_TIME_SEC}s)")
    print(f"[1] → SITZEN sammeln      ({COLLECTION_TIME_SEC}s)")
    print(f"[2] → GEHEN sammeln       ({COLLECTION_TIME_SEC}s)")
    print(f"[T] → KI Trainieren")
    print(f"[D] → Alle Daten löschen (Reset)")
    print("="*50 + "\n")

    def on_key(event):
        if event.key in ['0', '1', '2'] and reader.collection_target is None:
            label = int(event.key)
            if label in CLASS_NAMES:
                print(f"\n▶ Aufnahme gestartet: {CLASS_NAMES[label]}")
                reader.collection_start_time = time.time()
                reader.collection_target = label
                reader.window_buffer.clear()
                
        elif event.key == 't':
            if len(reader.training_features) > 0:
                state_text.set_text("⏳ TRAINING LÄUFT...")
                state_text.set_color("orange")
                plt.pause(0.1)
                reader.save_data()
                brain.train(np.array(reader.training_features), np.array(reader.training_labels))
                state_text.set_text("✓ Training abgeschlossen")
                state_text.set_color("green")
            else:
                print("✗ Keine Daten zum Trainieren!")
                
        elif event.key == 'delete' or event.key == 'd':
            print("\n⚠ Lösche alle Daten und Modelle...")
            reader.training_features = []
            reader.training_labels = []
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
            if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
            brain.is_trained = False
            brain.prediction_buffer.clear()
            state_text.set_text("✓ RESET - Bereit für neue Aufnahmen")
            state_text.set_color("blue")
            print("✓ Reset abgeschlossen.\n")

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Live-Loop mit Performance-Optimierung
    plot_counter = 0
    update_interval = 0.05  # Dynamisch anpassbar
    
    while True:
        # Check 1: Sammlung beendet?
        if reader.collection_target is not None:
            elapsed = time.time() - reader.collection_start_time
            if elapsed >= COLLECTION_TIME_SEC:
                label_name = CLASS_NAMES[reader.collection_target]
                count = sum(1 for l in reader.training_labels if l == reader.collection_target)
                reader.collection_target = None
                reader.window_buffer.clear()
                print(f"✓ Aufnahme beendet: {label_name} ({count} Samples gesammelt)\n")

        # Check 2: Neue Features aus Queue holen
        try:
            feats = reader.feature_queue.get_nowait()
            
            # Features splitten für Visualisierung
            mean_data = feats[:64]
            std_data = feats[64:128] / 5
            diff_data = feats[128:] / 10
            
            # Plot-Dezimierung (nur jedes N-te Frame zeichnen)
            plot_counter += 1
            if plot_counter >= PLOT_EVERY_N:
                plot_counter = 0
                
                # Amplitude Plot
                line_amp.set_ydata(mean_data)
                current_ylim = ax_amp.get_ylim()
                data_range = np.max(np.abs(mean_data))
                if data_range > current_ylim[1] * 0.8:
                    ax_amp.set_ylim(-data_range*1.2, data_range*1.2)
                
                # Varianz Plot
                line_var.set_ydata(std_data)
                if np.max(std_data) > ax_var.get_ylim()[1] * 0.8:
                    ax_var.set_ylim(0, np.max(std_data)*1.3)

            # KI Vorhersage (immer, auch ohne Plot-Update)
            if brain.is_trained and reader.collection_target is None:
                pred, conf, probs = brain.predict_smooth(feats)
                
                if pred is not None:
                    # Bars nur bei Plot-Update zeichnen
                    if plot_counter == 0:
                        for bar, p in zip(bars, probs):
                            bar.set_height(p)
                            if p == np.max(probs) and p > CONFIDENCE_THRESHOLD:
                                bar.set_color('green')
                            elif p > CONFIDENCE_THRESHOLD:
                                bar.set_color('orange')
                            else:
                                bar.set_color('gray')
                    
                    # Status Update
                    if conf > CONFIDENCE_THRESHOLD:
                        state_text.set_text(f"✓ {CLASS_NAMES[pred]} ({conf*100:.0f}%)")
                        state_text.set_color("green" if conf > 0.8 else "orange")
                        update_interval = 0.03  # Schnell bei hoher Konfidenz
                    else:
                        state_text.set_text(f"? Unsicher ({conf*100:.0f}%)")
                        state_text.set_color("gray")
                        update_interval = 0.08  # Langsamer bei Unsicherheit
            
            # Sammel-Status
            if reader.collection_target is not None:
                lbl = CLASS_NAMES[reader.collection_target]
                elapsed = time.time() - reader.collection_start_time
                remaining = max(0, int(COLLECTION_TIME_SEC - elapsed))
                samples = sum(1 for l in reader.training_labels if l == reader.collection_target)
                state_text.set_text(f"⏺ Sammle '{lbl}': {remaining}s | {samples} Samples")
                state_text.set_color("orange")
                
        except queue.Empty:
            pass  # Keine neuen Daten
        
        # GUI Update
        if plot_counter == 0:
            plt.pause(update_interval)
        else:
            time.sleep(0.01)  # Weniger Last ohne Plot-Update

if __name__ == "__main__": 
    main()