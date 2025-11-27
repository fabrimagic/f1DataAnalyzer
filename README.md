# F1 Data Visualizer

Un'applicazione desktop Tkinter per esplorare i dati delle sessioni di Formula 1 attraverso l'API pubblica di [openf1](https://openf1.org/). L'app scarica i dati live/archiviati, li organizza in tabelle e li visualizza con grafici interattivi tramite Matplotlib.

## Caratteristiche principali
- Ricerca delle sessioni per anno e caricamento dei risultati ufficiali.
- Profili pilota con cache in memoria per ridurre le richieste ripetute.
- Grafici dei distacchi e pressione in battaglia con selezione dei punti sul grafico.
- Analisi stint, gomme e tempi sul giro con grafici dedicati.
- Statistiche pit stop e finestre di strategia con visualizzazioni dedicate.
- Meteo, race control, timeline gara e radio di squadra (con riproduzione tramite lettore esterno se disponibile).

## Requisiti
- Python 3.10 o superiore.
- Tkinter (generalmente incluso con le installazioni standard di Python).
- Dipendenze Python installabili con `pip install -r requirements.txt`.
- Un lettore audio/video CLI opzionale come `mpv` o `ffplay` se si desidera riprodurre i messaggi radio.

## Esecuzione
1. Clona il repository e installa le dipendenze richieste:
   ```bash
   pip install -r requirements.txt
   ```
2. Avvia l'applicazione con:
   ```bash
   python intervals.py
   ```
3. Seleziona un anno, scegli una sessione e naviga tra le tab per analizzare distacchi, stint, pit stop, meteo e timeline della gara.

## Note
- Le chiamate all'API sono semplici `urllib` con gestione degli errori: la connettività di rete è necessaria per scaricare i dati.
- Alcune funzionalità (es. riproduzione radio) dipendono dalla presenza di strumenti esterni rilevati in PATH.
