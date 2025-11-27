# f1DataAnalyzer

## Introduzione
f1DataAnalyzer è un'applicazione desktop basata su Tkinter che consente di esplorare e confrontare i dati delle sessioni di Formula 1 tramite l'API pubblica di [openf1.org](https://openf1.org). L'interfaccia unisce tabelle interattive e grafici Matplotlib per offrire una vista completa su distacchi, stint gomme, pit stop, meteo, messaggi Race Control, team radio e una timeline unificata degli eventi di gara.

### Obiettivi principali
- Fornire un cruscotto unico per analizzare le sessioni (Race o Sprint) e commentare l'andamento della gara.
- Evidenziare distacchi, fasi di pressione/battaglia e momenti chiave (pit stop, sorpassi, segnalazioni Race Control).
- Collegare prestazioni e strategie a variabili esterne come meteo e scelta dei compound.

## Indice dei contenuti
- [Requisiti e installazione](#requisiti-e-installazione)
  - [Prerequisiti](#prerequisiti)
  - [Installazione e avvio](#installazione-e-avvio)
- [Panoramica dell'interfaccia utente](#panoramica-dellinterfaccia-utente)
  - [Flusso tipico d'uso](#flusso-tipico-duso)
- [Funzionalità dettagliate](#funzionalità-dettagliate)
  - [1. Selezione sessione e risultati piloti](#1-selezione-sessione-e-risultati-piloti)
  - [2. Analisi Gap / Distacchi](#2-analisi-gap--distacchi)
  - [3. Slipstream / Aria pulita](#3-slipstream--aria-pulita)
  - [4. Battle / Pressure Index](#4-battle--pressure-index)
  - [5. Gomme e Stint](#5-gomme-e-stint)
  - [6. Pit stop & Strategia](#6-pit-stop--strategia)
  - [7. Statistiche lap time](#7-statistiche-lap-time)
  - [8. Meteo e correlazione meteo-prestazioni](#8-meteo-e-correlazione-meteo-prestazioni)
  - [9. Race Control](#9-race-control)
  - [10. Team Radio](#10-team-radio)
  - [11. Race Timeline](#11-race-timeline)
  - [12. Altre funzionalità utili](#12-altre-funzionalità-utili)
- [Metodi di calcolo e logiche di analisi](#metodi-di-calcolo-e-logiche-di-analisi)
- [Come interpretare i risultati](#come-interpretare-i-risultati)
- [Limitazioni e note](#limitazioni-e-note)
- [Licenza e crediti](#licenza-e-crediti)

## Requisiti e installazione

### Prerequisiti
- **Python**: versione 3.10 o superiore.
- **Librerie Python**: installabili con `pip install -r requirements.txt` (Tkinter è generalmente incluso con Python). Matplotlib è necessario per i grafici; urllib, json e tkinter/ttk sono usati per la GUI e le API.
- **Player audio opzionale** per i team radio: l'app cerca automaticamente `ffplay`, `mpv`, `mpg123`, `cvlc` o `afplay` nel PATH. Senza un player disponibile i file audio vengono scaricati ma non riprodotti.

### Installazione e avvio
1. Clona il repository e installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
2. Avvia l'applicazione desktop:
   ```bash
   python intervals.py
   ```
3. Inserisci l'anno di interesse, premi **Recupera calendario**, quindi seleziona una sessione per iniziare l'analisi.

## Panoramica dell'interfaccia utente
L'applicazione presenta una finestra principale suddivisa in pannelli e tab:
- **Barra comandi**: inserimento anno, pulsante per scaricare il calendario e azioni rapide (risultati, grafici pilota, statistiche, pit strategy, undercut/overcut, pit window, Race Control, Team Radio).
- **Pannello calendario sessioni**: tabella con le sessioni dell'anno (doppio click per caricare i risultati).
- **Risultati piloti**: tabella con posizione, numero, nome, team, giri, punti, stato, gap e durata. La selezione di un pilota abilita le analisi successive.
- **Notebook di analisi** con tab dedicate: Distacchi & Gap, Gomme/Stint, Statistiche, Meteo (base e avanzata), Pit stop & strategia, Race Control, Team Radio, Race Timeline, Battle/Pressure Index.
- **Barra di stato**: messaggi guida su operazioni in corso o risultati caricati.

### Flusso tipico d'uso
1. Inserisci l'anno e scarica le sessioni.
2. Seleziona una sessione (Race o Sprint per le analisi complete) e visualizza i risultati.
3. Seleziona un pilota nella tabella risultati.
4. Apri la tab **Grafici & giri pilota** per caricare distacchi, stint, giri e tabelle correlate.
5. Esplora le altre tab (pit, meteo, timeline, team radio, battle/pressure) utilizzando i pulsanti dedicati quando richiesto.

## Funzionalità dettagliate

### 1. Selezione sessione e risultati piloti
- **Calendario sessioni**: elenco filtrato per anno, con colonne anno, data/ora UTC, nome sessione, tipo (Race, Sprint, Practice, Qualifying), circuito, località, paese, session_key e meeting_key.
- **Risultati sessione**: tabella con posizione finale, numero pilota, nome completo, team, giri completati, punti, stato (Finished, DNF, DNS, DSQ), gap e durata gara. Nei contesti Race/Sprint vengono caricati anche i pit stop e resi disponibili i grafici pilota.
- **Selezione pilota**: le analisi successive (distacchi, stint, team radio, meteo per pilota) usano il pilota evidenziato nella tabella.

### 2. Analisi Gap / Distacchi
- **Dati utilizzati**: endpoint `intervals` per gap_to_leader (distacco dal leader) e interval (distacco dal pilota immediatamente davanti).
- **Grafico distacchi**: linee sovrapposte di gap_to_leader e interval per giro. Cliccando un punto viene mostrato il valore esatto di entrambi gli indicatori.
- **Interpretazione**:
  - *gap_to_leader*: tempo rispetto al primo.
  - *interval*: differenza dal pilota precedente; serve per capire scia/battaglia.
- **Scia e aria pulita**: percentuali calcolate con `compute_slipstream_stats`, usando le soglie `interval < 1.0s` (in scia) e `interval > 2.5s` (aria pulita).
- **Pressure stints**: `find_pressure_stints` individua segmenti di avvicinamento/allontanamento rapidi (finestra di 3 giri, variazione >= 1.5s). I segmenti sono elencati in tabella e marcati sul grafico.

### 3. Slipstream / Aria pulita
- **Definizione operativa**: scia quando `interval < 1.0s`; aria pulita quando `interval > 2.5s`.
- **Statistiche**: percentuali di giri validi in scia e in aria pulita mostrate sotto il grafico distacchi.
- **Lettura**: valori alti di scia indicano pressione/attacco, aria pulita indica pista libera e gestione gara.

### 4. Battle / Pressure Index
- **Significato**: misura quanto un pilota ha attaccato o difeso, combinando distacchi, segmenti di pressione e sorpassi.
- **Dati**: intervals, sorpassi (`overtakes`), soglie di scia/aria pulita.
- **Calcolo** (`compute_battle_pressure_for_driver`):
  - *attack_laps*: giri con interval < 1.0s.
  - *clean_laps*: giri con interval > 2.5s.
  - *pressure_stints*: segmenti di avvicinamento (trend "Avvicinamento" su interval).
  - *defense_segments*: segmenti di allontanamento (trend "Allontanamento" su interval).
  - *sorpassi fatti/subiti*: conteggio da `build_overtake_index` (overtaking_driver_number vs overtaken_driver_number).
  - L'indice mostra numero di segmenti offensivi/difensivi, delta di tempo guadagnato o perso e sorpassi.
- **Visualizzazione**: tabella riassuntiva per pilota selezionato e mini-grafico dei distacchi con marker sui segmenti di pressione.

### 5. Gomme e Stint
- **Dati**: endpoint `stints` e `laps` per il pilota selezionato.
- **Tabella stint**: elenco di stint con compound, giri di inizio/fine, lunghezza e tempo medio (giri validi, esclude out-lap).
- **Mappa stint**: grafico a barre orizzontali per visualizzare i compound nel tempo gara.
- **Degrado stint**: selezionando uno stint è possibile tracciare lap_time vs giro per misurare l'evoluzione del passo.
- **Medie per compound**: bar chart con tempo medio per ciascun compound usato.

### 6. Pit stop & Strategia
- **Tabella pit**: per sessioni Race/Sprint, elenco di tutti i pit stop con giro e durata.
- **Statistiche pilota**: numero pit, durata media/migliore/peggiore e giri dei pit per ogni pilota.
- **Distribuzione pit per giro**: grafico che mostra quanti pit sono avvenuti su ogni giro.
- **Undercut/Overcut**: analisi automatica di coppie di piloti vicini in classifica. Confronta giri attorno ai pit per stimare il delta guadagnato/perduto.
- **Pit window & posizione virtuale**: stima della posizione virtuale dopo il pit usando gap/interval dei giri circostanti e una pit loss di riferimento (15s). Evidenzia giri sotto SC/VSC e segnala finestre “safe” o rischiose.

### 7. Statistiche lap time
- **Dati**: endpoint `laps` per tutti i piloti della sessione.
- **Calcolo**: per ogni pilota vengono usati solo i giri con `lap_duration` numerico e `is_pit_out_lap=False`. Si calcolano numero di giri utili, media, deviazione standard, miglior giro e gap dal miglior giro assoluto della sessione.
- **Tabella**: mostra posizione finale, nome, team, giri usati, lap medio, deviazione standard, best lap e gap dal best di sessione.

### 8. Meteo e correlazione meteo-prestazioni
- **Dati**: endpoint `weather` per track_temperature, air_temperature, humidity, wind_speed, rainfall.
- **Grafico meteo**: doppio pannello con andamento di temperature pista/aria e umidità/vento; indicazioni se è stata rilevata pioggia.
- **Statistiche sintetiche**: media/min/max per temperature, media per umidità e vento.
- **Correlazione track temp vs lap time**: abbina ogni giro valido al campione meteo più vicino nel tempo, esclude out-lap; calcola correlazione di Pearson e retta di regressione (trend line) e mostra i punti in scatter plot.
- **Analisi pioggia (impatto sul passo)**: classifica i campioni in fasi DRY/TRANSITION/WET, associa lap time più vicini e confronta velocità media tra le fasi.
- **Compound vs track temp**: confronta tempi medi per compound rispetto alla temperatura pista tramite scatter colorati e tabella riassuntiva.

### 9. Race Control
- **Dati**: endpoint `race_control` per l'intera sessione.
- **Tabella**: data/ora, giro, categoria, flag, scope e messaggio. Messaggi SC/VSC/Yellow/Red sono evidenziati e inclusi anche nella Race Timeline.

### 10. Team Radio
- **Dati**: endpoint `team_radio` per il pilota selezionato.
- **Tabella**: timestamp UTC, giro e URL di registrazione (colonna nascosta).
- **Riproduzione**: doppio click o pulsante **Riproduci selezione** scarica l'audio in un file temporaneo e lo avvia con il primo player disponibile (ffplay/mpv/mpg123/cvlc/afplay). In assenza di player viene mostrato un errore informativo.

### 11. Race Timeline
- **Definizione**: sequenza cronologica unificata degli eventi di gara.
- **Sorgenti dati**: sorpassi (overtakes), pit stop, Race Control (flag/SC/VSC), meteo (pioggia o variazioni track temp >=5°C), analisi gap (segmenti di pressione) e team radio del pilota selezionato.
- **Costruzione** (`compute_race_timeline`):
  - Converte timestamp ISO e, se assente, usa l'ora d'inizio giro come riferimento.
  - Crea eventi con tipo (Sorpasso, Pit stop, Race Control, Meteo, Gap/Pressure, Team Radio) e li ordina temporalmente.
  - Ogni evento include giro (se disponibile), descrizione e pilota/i coinvolti.
- **Uso**: pulsante **Genera Timeline Gara** popola la tabella; selezionando una riga si vede il dettaglio; **Esporta Timeline** salva un file di testo con tutti gli eventi ordinati.

### 12. Altre funzionalità utili
- **Reset automatici**: cambiando pilota o sessione vengono ripuliti grafici e tabelle pertinenti (distacchi, stint, meteo, Race Control, Team Radio, timeline) per evitare dati incoerenti.
- **Messaggi guida**: ogni sezione ha label descrittivi che indicano cosa fare (es. caricare meteo, selezionare pilota, ecc.).

## Metodi di calcolo e logiche di analisi
- **Scia / aria pulita**: interval < 1.0s = scia; interval > 2.5s = aria pulita. Percentuali basate sui giri con interval numerico.
- **Pressure stints**: finestre di 3 giri; variazioni di gap_to_leader o interval >= 1.5s marcano segmenti di avvicinamento (attacco) o allontanamento (difesa).
- **Battle/Pressure Index**: combina attack_laps, clean_laps, numero e delta dei pressure_stints (metric=interval), defense_segments, sorpassi fatti/subiti.
- **Associazione giri–meteo**: per ogni giro valido si trova il campione meteo con timestamp più vicino; si calcola correlazione Pearson tra track temp e lap time e una retta di regressione lineare.
- **Timeline di gara**: unione di sorpassi, pit stop, messaggi Race Control, eventi meteo (pioggia/variazioni), segmenti di pressione dai gap e team radio; ordinamento per timestamp (o giro se manca l'orario).
- **Statistiche lap time**: per pilota si considerano solo lap_duration numerici e out-lap esclusi; calcolo di media, deviazione standard, best lap e gap dal best assoluto.

## Come interpretare i risultati
- **Costanza e ritmo**: media e deviazione standard dei lap time indicano regolarità; best lap e gap dal best di sessione mostrano il potenziale sul giro singolo.
- **Degrado gomme e stint**: grafico di degrado evidenzia l'aumento dei tempi all'interno di uno stint; la mappa stint mostra la scelta dei compound e la loro lunghezza.
- **Scia vs aria pulita**: percentuali elevate di scia con molti pressure_stints indicano attacco prolungato; aria pulita alta suggerisce gestione gomme o pista libera.
- **Pit strategy**: confronta numero e durata dei pit, distribuzione per giro e analisi undercut/overcut per capire chi ha guadagnato nel cambio gomma; la pit window aiuta a valutare giri sicuri per fermarsi rispetto ai rivali.
- **Race Timeline**: consente di ricostruire la narrazione della gara combinando sorpassi, pit, segnalazioni e condizioni meteo; utile per preparare resoconti o commenti tecnici.
- **Meteo**: la correlazione track temp vs lap time e le fasi di pioggia aiutano a capire quanto le condizioni abbiano influenzato il passo o la scelta dei compound.

## Limitazioni e note
- L'app dipende dai dati di **openf1.org**: alcune sessioni possono mancare di campioni meteo, intervals, team radio o sorpassi rilevati automaticamente.
- Se il dataset è incompleto, alcune tabelle/grafici resteranno vuoti o mostreranno messaggi informativi.
- La riproduzione dei team radio richiede un player audio CLI installato e raggiungibile nel PATH.

## Licenza e crediti
- Licenza: da definire dal mantenitore del repository.
- Dati forniti da [openf1.org](https://openf1.org); verificare termini e condizioni del servizio.

