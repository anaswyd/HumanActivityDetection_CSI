# WiFi-Sensing: Personen- und Aktivitätserkennung mittels WLAN-Signalen

[cite_start]Dieses Projekt demonstriert die Nutzung von **Channel State Information (CSI)** gewöhnlicher WLAN-Signale zur kontaktlosen Erkennung menschlicher Aktivitäten[cite: 42, 1042]. [cite_start]Es verbindet modernste Signalverarbeitung mit Deep Learning und zeigt den Transfer von hochpräzisen Labordaten auf kostengünstige **IoT-Hardware (ESP32)** in realen, unkontrollierten Wohnumgebungen[cite: 51, 604].

## 📑 Projektübersicht
[cite_start]Im Gegensatz zu kamerabasierten Systemen bietet WiFi-Sensing eine privatsphäre-wahrende Alternative zur Überwachung im Sicherheits- und Gesundheitswesen[cite: 40, 43, 1043]. [cite_start]Durch die Analyse physikalischer Kanalveränderungen, die durch menschliche Bewegungen verursacht werden, können Aktivitäten wie Gehen, Sitzen oder Stehen algorithmisch klassifiziert werden[cite: 86, 629].

### Technische Kernaspekte:
* [cite_start]**Hybride Modellarchitektur**: Entwicklung eines ressourceneffizienten **CNN-LSTM-Netzwerks** (33.044 Parameter), das räumliche Subcarrier-Topologien für statische Posen mit zeitlicher Bewegungsdynamik für dynamische Aktivitäten kombiniert[cite: 757, 758, 773].
* [cite_start]**Robustes Signal-Processing**: Implementierung einer zweistufigen Filterkaskade aus **Hampel-Filtern** zur Ausreißereliminierung und **Savitzky-Golay-Filtern** zur Signalglättung und Erhaltung relevanter Wellenflanken[cite: 738, 747, 1399].
* [cite_start]**Wissenschaftliche Validierung**: Fokus auf die Vermeidung von *Temporal Data Leakage* durch den Einsatz von **Group K-Fold Cross-Validation** anstelle einfacher Train/Test-Splits[cite: 166, 175, 1165].
* [cite_start]**Echtzeit-Inferenz**: Entwicklung einer Live-Pipeline für ESP32-Mikrocontroller inklusive Sliding-Window-Puffer und Majority-Vote-Verfahren zur Vorhersagestabilisierung[cite: 715, 720, 1402].

## 🛠 Technologien & Methoden
* [cite_start]**Deep Learning**: TensorFlow / Keras (1D-CNN, LSTM, hybride Modelle), GELU-Aktivierung, Callbacks (Early Stopping, ReduceLROnPlateau)[cite: 185, 227, 232, 1444].
* [cite_start]**Data Science & Signalverarbeitung**: SMOTE zur Klassenbalancierung, Z-Transformation, Data Augmentation (Magnitude Scaling, Time Shifting, Gaußsches Rauschen)[cite: 128, 154, 156, 728].
* [cite_start]**Hardware**: ESP32 Mikrocontroller im 20-MHz-Band, serielle Echtzeit-Datenextraktion aus 64 Subcarriern[cite: 608, 611, 1308].

## 📊 Ergebnisse & Key Insights
[cite_start]Das Projekt belegt die signifikante Differenz zwischen idealisierten Laborbedingungen und realen IoT-Deployments (geprägt durch Multipath Propagation und Hardware-Jitter)[cite: 648, 656]:

| Methode | Datensatz / Umgebung | Split-Verfahren | Accuracy |
| :--- | :--- | :--- | :--- |
| **1D-CNN (mit Augmentation)** | Labor-Baseline | Group K-Fold | [cite_start]**99,33 %** [cite: 495, 554] |
| **Hybrid CNN-LSTM** | ESP32 (Wohnumgebung) | Standard-Split | [cite_start]**96,18 %** [cite: 806, 946] |
| **Hybrid CNN-LSTM** | ESP32 (Wohnumgebung) | Group K-Fold | [cite_start]**69,63 %** [cite: 814, 951] |

[cite_start]**Fazit**: Die Arbeit zeigt kritisch auf, dass hohe Einzelmetriken (wie die 96,18 % im Standard-Split) aus kontrollierten Datensätzen oft durch Data Leakage entstehen[cite: 905, 948]. [cite_start]Eine belastbare Bewertung für IoT-Systeme erfordert zwingend die Prüfung der Generalisierungsfähigkeit unter unbekannten Bedingungen (hier abgebildet durch die Group K-Fold-Validierung mit 69,63 %)[cite: 966, 1566].

## 📁 Dokumentation
Für tiefergehende Informationen zu den mathematischen Grundlagen, den Filterkaskaden und den vollständigen Evaluationsmetriken stehen unsere Ausarbeitungen zur Verfügung:

* [**Schriftliche Ausarbeitung (PDF)**](Paper/Personen__und_Aktivitätserkennung_mittels_WLAN_Signalen.pdf)
* [**Präsentationsfolien (PDF)**](Paper/Personen__und_Aktivitätserkennung_mittels_WLAN_Signalen_(2).pdf)

---
[cite_start]*Dieses Projekt wurde im Rahmen des Info-Projekts (WS 25/26) am Fachbereich Informatik und Ingenieurwissenschaften der Frankfurt University of Applied Sciences erstellt[cite: 4, 15, 998].*

[cite_start]**Autoren**: Anas Loukili & Yassin Ouchen [cite: 6, 8, 999]
