import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request
import urllib.error
import json
import math
from datetime import datetime

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Endpoint API --- #
API_SESSIONS_URL = "https://api.openf1.org/v1/sessions?year="
API_RESULTS_URL = "https://api.openf1.org/v1/session_result?session_key="
API_DRIVER_INFO_URL = "https://api.openf1.org/v1/drivers?session_key={session_key}&driver_number={driver_number}"
API_INTERVALS_URL = "https://api.openf1.org/v1/intervals?session_key={session_key}&driver_number={driver_number}"
API_STINTS_URL = "https://api.openf1.org/v1/stints?session_key={session_key}&driver_number={driver_number}"
API_LAPS_URL = "https://api.openf1.org/v1/laps?session_key={session_key}&driver_number={driver_number}"
API_PIT_URL = "https://api.openf1.org/v1/pit?session_key={session_key}"
API_WEATHER_URL = "https://api.openf1.org/v1/weather?session_key={session_key}"
API_RACE_CONTROL_URL = "https://api.openf1.org/v1/race_control?session_key={session_key}"

# Cache per le informazioni pilota
DRIVER_CACHE = {}
DRIVER_PROFILE_CACHE = {}

# Canvas matplotlib per i grafici (uno per tab)
gap_plot_canvas = None
stints_plot_canvas = None
gap_plot_frame = None

# Dati e handler per il click sul grafico distacchi
gap_fig = None
gap_ax = None
gap_click_cid = None
gap_click_data = {"laps": [], "gap_leader": [], "gap_prev": []}
gap_point_info_var = None  # inizializzato dopo la creazione della GUI
gap_slipstream_var = None
gap_pressure_tree = None
race_control_tree = None
race_control_info_var = None

# Impostazioni per identificare scia/aria pulita e momenti chiave
SLIPSTREAM_THRESHOLD = 1.0
CLEAN_AIR_THRESHOLD = 2.5
PRESSURE_DELTA_THRESHOLD = 1.5
PRESSURE_WINDOW = 3

# Meteo
weather_canvas = None
weather_fig = None
weather_plot_frame = None
weather_info_var = None
weather_stats_vars = {}
weather_perf_canvas = None
weather_perf_fig = None
weather_perf_info_var = None
weather_perf_plot_frame = None
weather_last_data = []
weather_last_session_key = None

# Statistiche piloti (lap time)
stats_tree = None

# Dati correnti per stint/laps del pilota selezionato
current_stints_data = []
current_laps_data = []
current_laps_session_key = None

# Riferimenti GUI per sezione gomme
stints_mode_var = None
stints_summary_tree = None
stints_combo = None
stints_plot_canvas_frame = None
current_stints_for_combo = []

# Dati correnti per pit & risultati (per analisi strategia)
current_pit_data = []
current_results_data = []

# Pit stop & strategia
pit_stats_tree = None
pit_strategy_canvas = None
pit_strategy_fig = None
pit_strategy_plot_frame = None


# --------------------- Funzioni per le API --------------------- #

def http_get_json(url: str):
    """GET semplice che ritorna JSON decodificato o solleva RuntimeError."""
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} per URL: {url}")
            data = response.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Errore di rete: {e}")
    except json.JSONDecodeError:
        raise RuntimeError("Risposta JSON non valida dall'API.")


def fetch_sessions(year: int):
    url = f"{API_SESSIONS_URL}{year}"
    sessions = http_get_json(url)
    if not isinstance(sessions, list):
        raise RuntimeError("Formato JSON inatteso per le sessioni.")
    return sessions


def fetch_session_results(session_key: int):
    url = f"{API_RESULTS_URL}{session_key}"
    results = http_get_json(url)
    if not isinstance(results, list):
        raise RuntimeError("Formato JSON inatteso per i risultati di sessione.")
    return results


def fetch_driver_full_name(driver_number, session_key: int):
    """Ritorna il full_name del pilota per (driver_number, session_key)."""
    try:
        dn_key = int(driver_number)
        skey = int(session_key)
    except (ValueError, TypeError):
        return ""

    cache_key = (skey, dn_key)
    if cache_key in DRIVER_CACHE:
        return DRIVER_CACHE[cache_key]

    profile = fetch_driver_profile(dn_key, skey)
    DRIVER_CACHE[cache_key] = profile.get("full_name", "")
    return DRIVER_CACHE[cache_key]


def fetch_driver_profile(driver_number, session_key):
    """Ritorna un dizionario con full_name e team_name per driver_number."""
    try:
        dn_key = int(driver_number)
        skey = int(session_key)
    except (ValueError, TypeError):
        return {"full_name": "", "team_name": ""}

    cache_key = (skey, dn_key)
    if cache_key in DRIVER_PROFILE_CACHE:
        return DRIVER_PROFILE_CACHE[cache_key]

    url = API_DRIVER_INFO_URL.format(session_key=skey, driver_number=dn_key)
    try:
        data = http_get_json(url)
    except RuntimeError:
        DRIVER_PROFILE_CACHE[cache_key] = {"full_name": "", "team_name": ""}
        return DRIVER_PROFILE_CACHE[cache_key]

    if not isinstance(data, list) or not data:
        DRIVER_PROFILE_CACHE[cache_key] = {"full_name": "", "team_name": ""}
        return DRIVER_PROFILE_CACHE[cache_key]

    first = data[0]
    full_name = first.get("full_name", "")
    if not isinstance(full_name, str):
        full_name = ""

    team_name = first.get("team_name", "")
    if not isinstance(team_name, str):
        team_name = ""

    DRIVER_PROFILE_CACHE[cache_key] = {"full_name": full_name, "team_name": team_name}
    return DRIVER_PROFILE_CACHE[cache_key]


def fetch_driver_team_name(driver_number, session_key):
    profile = fetch_driver_profile(driver_number, session_key)
    return profile.get("team_name", "")


def fetch_intervals(session_key: int, driver_number):
    """Dati di distacco (gap_to_leader, interval) per pilota."""
    try:
        dn_key = int(driver_number)
    except (ValueError, TypeError):
        raise RuntimeError("driver_number non valido per la chiamata intervals.")

    url = API_INTERVALS_URL.format(session_key=session_key, driver_number=dn_key)
    intervals = http_get_json(url)
    if not isinstance(intervals, list):
        raise RuntimeError("Formato JSON inatteso per i dati intervals.")
    return intervals


def fetch_stints(session_key: int, driver_number):
    """Stint gomme per pilota nella sessione."""
    try:
        dn_key = int(driver_number)
    except (ValueError, TypeError):
        raise RuntimeError("driver_number non valido per la chiamata stints.")

    url = API_STINTS_URL.format(session_key=session_key, driver_number=dn_key)
    stints = http_get_json(url)
    if not isinstance(stints, list):
        raise RuntimeError("Formato JSON inatteso per i dati stints.")
    return stints


def fetch_laps(session_key: int, driver_number):
    """Dati dei giri per pilota nella sessione."""
    try:
        dn_key = int(driver_number)
    except (ValueError, TypeError):
        raise RuntimeError("driver_number non valido per la chiamata laps.")

    url = API_LAPS_URL.format(session_key=session_key, driver_number=dn_key)
    laps = http_get_json(url)
    if not isinstance(laps, list):
        raise RuntimeError("Formato JSON inatteso per i dati laps.")
    return laps


def fetch_pit_stops(session_key: int):
    """Elenco pit stop per la sessione (tutti i piloti)."""
    url = API_PIT_URL.format(session_key=session_key)
    pits = http_get_json(url)
    if not isinstance(pits, list):
        raise RuntimeError("Formato JSON inatteso per i dati pit.")
    return pits


def fetch_weather(session_key: int):
    """Dati meteo per la singola sessione (session_key)."""
    url = API_WEATHER_URL.format(session_key=session_key)
    weather = http_get_json(url)
    if not isinstance(weather, list):
        raise RuntimeError("Formato JSON inatteso per i dati weather.")
    return weather


def fetch_race_control_messages(session_key: int):
    """Messaggi Race Control per l'intera sessione."""
    url = API_RACE_CONTROL_URL.format(session_key=session_key)
    messages = http_get_json(url)
    if not isinstance(messages, list):
        raise RuntimeError("Formato JSON inatteso per i dati Race Control.")
    return messages


# --------------------- Utility per la GUI --------------------- #

def format_time_from_seconds(seconds_value):
    """Converte un valore in secondi in stringa MM:SS.mmm. Se non valido, ritorna stringa vuota."""
    if not isinstance(seconds_value, (int, float)):
        return ""
    total_ms = int(round(seconds_value * 1000))
    minutes = total_ms // 60000
    rest_ms = total_ms % 60000
    seconds = rest_ms // 1000
    millis = rest_ms % 1000
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_iso_datetime(value: str):
    """Parsing sicuro per stringhe ISO8601 restituite dall'API (gestisce suffisso Z)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def compute_numeric_stats(values):
    """Ritorna (media, min, max) ignorando valori non numerici."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None, None, None
    mean_val = sum(nums) / len(nums)
    return mean_val, min(nums), max(nums)


def compute_mean(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def is_race_like(session_type: str) -> bool:
    """Considera 'gara' sia Race che Sprint."""
    st = (session_type or "").strip().lower()
    return st in ("race", "sprint")


def get_selected_session_info():
    """
    Ritorna (session_key, session_type, meeting_key) della sessione selezionata.
    Per il meteo usiamo solo session_key.
    """
    selection = sessions_tree.selection()
    if not selection:
        return None, None, None

    item_id = selection[0]
    values = sessions_tree.item(item_id, "values")
    if not values or len(values) < 9:
        return None, None, None

    session_type = values[3]
    session_key_str = values[7]
    meeting_key_str = values[8]

    session_key = int(session_key_str) if str(session_key_str).isdigit() else None
    meeting_key = int(meeting_key_str) if str(meeting_key_str).isdigit() else None

    return session_key, session_type, meeting_key


def get_selected_driver_info():
    """Ritorna (driver_number, driver_name) del pilota selezionato nei risultati."""
    selection = results_tree.selection()
    if not selection:
        return None, None

    item_id = selection[0]
    values = results_tree.item(item_id, "values")
    if not values or len(values) < 3:
        return None, None

    driver_number = values[1]
    driver_name = values[2]
    return driver_number, driver_name


def clear_driver_plots():
    """Rimuove i grafici distacchi e stint gomme e resetta handler click."""
    global gap_plot_canvas, stints_plot_canvas
    global gap_fig, gap_ax, gap_click_cid, gap_click_data, gap_point_info_var
    global current_stints_data, current_laps_data, current_laps_session_key, current_stints_for_combo

    # Disconnette eventuale handler click dal grafico distacchi
    if gap_fig is not None and gap_click_cid is not None:
        try:
            gap_fig.canvas.mpl_disconnect(gap_click_cid)
        except Exception:
            pass

    gap_fig = None
    gap_ax = None
    gap_click_cid = None
    gap_click_data = {"laps": [], "gap_leader": [], "gap_prev": []}

    if gap_plot_canvas is not None:
        gap_plot_canvas.get_tk_widget().destroy()
        gap_plot_canvas = None

    if stints_plot_canvas is not None:
        stints_plot_canvas.get_tk_widget().destroy()
        stints_plot_canvas = None

    if gap_point_info_var is not None:
        gap_point_info_var.set(
            "Clicca un punto sul grafico distacchi per vedere gap_to_leader e interval."
        )

    reset_gap_insights()

    # Reset dati stint/laps del pilota
    current_stints_data = []
    current_laps_data = []
    current_laps_session_key = None
    current_stints_for_combo = []
    if stints_combo is not None:
        stints_combo["values"] = ()
        stints_combo.set("")

    clear_stints_summary_table()
    clear_race_control_table()


def clear_weather_plot():
    """Rimuove il grafico meteo, se presente."""
    global weather_canvas, weather_fig, weather_info_var, weather_last_data, weather_last_session_key
    if weather_canvas is not None:
        weather_canvas.get_tk_widget().destroy()
        weather_canvas = None
    weather_fig = None
    if weather_info_var is not None:
        weather_info_var.set(
            "Meteo sessione: in attesa di una sessione selezionata."
        )

    weather_last_data = []
    weather_last_session_key = None

    update_weather_stats_box()
    clear_weather_performance_plot()


def clear_race_control_table():
    """Svuota la tabella Race Control e resetta il messaggio informativo."""
    global race_control_tree, race_control_info_var
    if race_control_tree is not None:
        for item in race_control_tree.get_children():
            race_control_tree.delete(item)

    if race_control_info_var is not None:
        race_control_info_var.set(
            "Messaggi Race Control: seleziona una sessione Race/Sprint, poi premi "
            "'Recupera messaggi Race Control' per visualizzare i dettagli."
        )


def clear_stats_table():
    """Svuota la tabella delle statistiche piloti, se esiste."""
    global stats_tree
    if stats_tree is not None:
        for item in stats_tree.get_children():
            stats_tree.delete(item)


def reset_gap_insights():
    """Ripristina le info scia/aria pulita e la tabella momenti chiave."""
    global gap_slipstream_var, gap_pressure_tree
    if gap_slipstream_var is not None:
        gap_slipstream_var.set("In scia: -- | Aria pulita: --")

    if gap_pressure_tree is not None:
        for item in gap_pressure_tree.get_children():
            gap_pressure_tree.delete(item)


def update_pressure_table(segments):
    """Popola la tabella con i tratti di avvicinamento/allontanamento."""
    global gap_pressure_tree
    if gap_pressure_tree is None:
        return

    for item in gap_pressure_tree.get_children():
        gap_pressure_tree.delete(item)

    for seg in segments:
        delta_str = f"{seg['delta']:+.2f}"
        gap_pressure_tree.insert(
            "",
            "end",
            values=(
                seg["metric"],
                seg["trend"],
                seg["start_lap"],
                seg["end_lap"],
                delta_str,
            ),
        )


def update_race_control_messages(session_key: int):
    """Scarica e popola i messaggi Race Control per l'intera sessione."""
    global race_control_tree, race_control_info_var

    clear_race_control_table()

    if race_control_tree is None:
        return

    if race_control_info_var is not None:
        race_control_info_var.set("Scarico messaggi Race Control per la sessione...")

    try:
        messages = fetch_race_control_messages(session_key)
    except RuntimeError as e:
        if race_control_info_var is not None:
            race_control_info_var.set("Errore nel recupero dei messaggi Race Control.")
        messagebox.showerror("Errore Race Control", str(e))
        return

    if not messages:
        if race_control_info_var is not None:
            race_control_info_var.set(
                "Nessun messaggio Race Control in questa sessione."
            )
        return

    try:
        sorted_msgs = sorted(
            messages,
            key=lambda m: (
                m.get("lap_number") if isinstance(m.get("lap_number"), int) else -1,
                m.get("date", ""),
            ),
        )
    except Exception:
        sorted_msgs = messages

    for msg in sorted_msgs:
        raw_date = msg.get("date", "")
        parsed = parse_iso_datetime(raw_date)
        date_str = parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else str(raw_date)

        lap_number = msg.get("lap_number", "")
        category = msg.get("category", "")
        flag = msg.get("flag", "")
        scope = msg.get("scope", "")
        message_text = msg.get("message", "")

        race_control_tree.insert(
            "",
            "end",
            values=(date_str, lap_number, category, flag, scope, message_text),
        )

    if race_control_info_var is not None:
        race_control_info_var.set(
            f"{len(sorted_msgs)} messaggi Race Control per la sessione (ordinati per giro e data)."
        )


def clear_stints_summary_table():
    """Svuota la tabella riassuntiva degli stint."""
    global stints_summary_tree
    if stints_summary_tree is not None:
        for item in stints_summary_tree.get_children():
            stints_summary_tree.delete(item)


def clear_pit_strategy():
    """Svuota tabella e grafico di 'Pit stop & strategia'."""
    global pit_stats_tree, pit_strategy_canvas, pit_strategy_fig

    if pit_stats_tree is not None:
        for item in pit_stats_tree.get_children():
            pit_stats_tree.delete(item)

    if pit_strategy_canvas is not None:
        pit_strategy_canvas.get_tk_widget().destroy()
        pit_strategy_canvas = None

    pit_strategy_fig = None


# --------------------- Gestione click sul grafico distacchi --------------------- #

def on_gap_plot_click(event):
    """
    Handler per il click sul grafico dei distacchi.
    Quando si clicca vicino a un giro, mostra i valori di gap_to_leader e interval
    relativi a quel giro nella label dedicata.
    """
    global gap_ax, gap_click_data, gap_point_info_var

    if gap_ax is None:
        return
    if event.inaxes is not gap_ax:
        return
    if event.xdata is None or not gap_click_data["laps"]:
        return

    x = event.xdata
    laps = gap_click_data["laps"]
    gaps_leader = gap_click_data["gap_leader"]
    gaps_prev = gap_click_data["gap_prev"]

    # Trova il giro più vicino sull'asse X
    nearest_idx = min(range(len(laps)), key=lambda i: abs(laps[i] - x))
    if abs(laps[nearest_idx] - x) > 0.5:
        # Click troppo lontano da un giro intero -> ignora
        return

    lap_num = laps[nearest_idx]
    gl = gaps_leader[nearest_idx]
    gp = gaps_prev[nearest_idx]

    gl_str = f"{gl:.3f}" if isinstance(gl, (int, float)) else "N/A"
    gp_str = f"{gp:.3f}" if isinstance(gp, (int, float)) else "N/A"

    if gap_point_info_var is not None:
        gap_point_info_var.set(
            f"Giro {lap_num}: gap_to_leader = {gl_str} s, interval = {gp_str} s"
        )


def compute_slipstream_stats(interval_values):
    """Calcola percentuali di giri in scia e in aria pulita con soglie fisse."""
    valid = [v for v in interval_values if isinstance(v, (int, float))]
    if not valid:
        return 0.0, 0.0

    in_scia = sum(1 for v in valid if v < SLIPSTREAM_THRESHOLD)
    aria_pulita = sum(1 for v in valid if v > CLEAN_AIR_THRESHOLD)
    total = len(valid)

    return (in_scia / total) * 100.0, (aria_pulita / total) * 100.0


def find_pressure_stints(laps, gap_leader_series, interval_series):
    """
    Identifica segmenti in cui il gap cala o aumenta rapidamente.

    Restituisce una lista di dict con: metric (leader/interval), trend, start_lap,
    end_lap, delta e indice centrale per il marker grafico.
    """

    def analyze_series(series, metric_name):
        segments = []
        for i in range(len(series) - PRESSURE_WINDOW + 1):
            start_val = series[i]
            end_val = series[i + PRESSURE_WINDOW - 1]
            if start_val is None or end_val is None:
                continue

            delta = end_val - start_val
            if delta <= -PRESSURE_DELTA_THRESHOLD:
                trend = "Avvicinamento"
            elif delta >= PRESSURE_DELTA_THRESHOLD:
                trend = "Allontanamento"
            else:
                continue

            center_idx = i + (PRESSURE_WINDOW // 2)
            segments.append(
                {
                    "metric": metric_name,
                    "trend": trend,
                    "start_lap": laps[i],
                    "end_lap": laps[i + PRESSURE_WINDOW - 1],
                    "delta": delta,
                    "marker_x": laps[center_idx],
                    "marker_y": series[center_idx],
                }
            )
        return segments

    segments = []
    segments.extend(analyze_series(gap_leader_series, "gap_to_leader"))
    segments.extend(analyze_series(interval_series, "interval"))
    return segments


# --------------------- Meteo sessione --------------------- #

def update_weather_stats_box(track_stats=None, air_stats=None, humidity_mean=None, wind_mean=None):
    """Aggiorna il box riepilogo meteo con i valori calcolati."""
    global weather_stats_vars
    for key in ("track", "air", "humidity", "wind"):
        if key not in weather_stats_vars:
            weather_stats_vars[key] = tk.StringVar()
    default_line = "--"
    track_mean, track_min, track_max = track_stats if track_stats else (None, None, None)
    air_mean, air_min, air_max = air_stats if air_stats else (None, None, None)

    def fmt_triplet(mean_v, min_v, max_v, unit=""):
        if not isinstance(mean_v, (int, float)):
            return default_line
        min_str = f"{min_v:.1f}" if isinstance(min_v, (int, float)) else "--"
        max_str = f"{max_v:.1f}" if isinstance(max_v, (int, float)) else "--"
        unit_suffix = f" {unit}" if unit else ""
        return f"Media: {mean_v:.1f}{unit_suffix} | Min: {min_str}{unit_suffix} | Max: {max_str}{unit_suffix}"

    weather_stats_vars.get("track", tk.StringVar()).set(
        f"Track temp {fmt_triplet(track_mean, track_min, track_max, '°C')}"
    )
    weather_stats_vars.get("air", tk.StringVar()).set(
        f"Air temp {fmt_triplet(air_mean, air_min, air_max, '°C')}"
    )

    if isinstance(humidity_mean, (int, float)):
        hum_line = f"Umidità media: {humidity_mean:.1f}%"
    else:
        hum_line = "Umidità media: --"
    weather_stats_vars.get("humidity", tk.StringVar()).set(hum_line)

    if isinstance(wind_mean, (int, float)):
        wind_line = f"Wind speed media: {wind_mean:.1f} m/s"
    else:
        wind_line = "Wind speed media: --"
    weather_stats_vars.get("wind", tk.StringVar()).set(wind_line)


def clear_weather_performance_plot():
    """Ripulisce il grafico di correlazione meteo-prestazioni."""
    global weather_perf_canvas, weather_perf_fig, weather_perf_info_var
    if weather_perf_canvas is not None:
        weather_perf_canvas.get_tk_widget().destroy()
        weather_perf_canvas = None
    weather_perf_fig = None
    if weather_perf_info_var is not None:
        weather_perf_info_var.set(
            "Carica una sessione e seleziona un pilota per vedere la correlazione track temp vs lap time."
        )

def update_weather_for_session(session_key: int):
    """
    Scarica e visualizza il meteo per la sessione selezionata,
    usando esclusivamente la session_key.
    """
    global weather_canvas, weather_fig, weather_plot_frame, weather_info_var
    global weather_last_data, weather_last_session_key

    clear_weather_plot()

    if session_key is None:
        status_var.set("Impossibile recuperare il meteo: session_key mancante.")
        return

    status_var.set(
        f"Scarico dati meteo per la sessione {session_key}..."
    )
    root.update_idletasks()

    try:
        weather_data = fetch_weather(session_key)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero dei dati meteo.")
        messagebox.showerror("Errore meteo", str(e))
        return

    if not weather_data:
        status_var.set("Nessun dato meteo disponibile per questa sessione.")
        return

    # Ordina per data (stringa ISO)
    try:
        weather_sorted = sorted(
            weather_data,
            key=lambda w: w.get("date", "") if isinstance(w.get("date", ""), str) else ""
        )
    except Exception:
        weather_sorted = weather_data

    weather_last_data = weather_sorted
    weather_last_session_key = session_key

    x_idx = []
    track_temps = []
    air_temps = []
    hums = []
    wind_speeds = []
    rainfall_vals = []

    for idx, w in enumerate(weather_sorted, start=1):
        x_idx.append(idx)

        tt = w.get("track_temperature", None)
        track_temps.append(tt if isinstance(tt, (int, float)) else None)

        at = w.get("air_temperature", None)
        air_temps.append(at if isinstance(at, (int, float)) else None)

        hum = w.get("humidity", None)
        hums.append(hum if isinstance(hum, (int, float)) else None)

        ws = w.get("wind_speed", None)
        wind_speeds.append(ws if isinstance(ws, (int, float)) else None)

        rf = w.get("rainfall", None)
        rainfall_vals.append(rf if isinstance(rf, (int, float)) else None)

    if not x_idx:
        status_var.set("Nessun dato meteo valido da visualizzare per questa sessione.")
        return

    # Calcola presenza pioggia e primo campione
    rain_present = any((isinstance(r, (int, float)) and r > 0) for r in rainfall_vals)
    first_rain_info = None
    for idx, w in enumerate(weather_sorted, start=1):
        rf_val = w.get("rainfall", None)
        if isinstance(rf_val, (int, float)) and rf_val > 0:
            first_rain_info = (idx, w.get("date", ""))
            break

    lines_info = [
        f"Pioggia rilevata: {'Sì' if rain_present else 'No'} (in base ai campioni meteo della sessione)."
    ]
    if first_rain_info is not None:
        lines_info.append(
            f"Primi campioni con pioggia: indice {first_rain_info[0]} / data {first_rain_info[1]}"
        )
    if weather_info_var is not None:
        weather_info_var.set("\n".join(lines_info))

    # Aggiorna box statistiche sintetiche
    track_stats = compute_numeric_stats(track_temps)
    air_stats = compute_numeric_stats(air_temps)
    humidity_mean = compute_mean(hums)
    wind_mean = compute_mean(wind_speeds)
    update_weather_stats_box(track_stats, air_stats, humidity_mean, wind_mean)

    weather_fig_local = Figure(figsize=(6, 3.2))
    ax1 = weather_fig_local.add_subplot(211)
    ax2 = weather_fig_local.add_subplot(212)

    ax1.plot(x_idx, track_temps, marker="o", label="Track temp (°C)")
    ax1.plot(x_idx, air_temps, marker="o", linestyle="--", label="Air temp (°C)")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title("Evoluzione temperatura asfalto / aria")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(x_idx, hums, marker="o", label="Umidità (%)")
    ax2.plot(x_idx, wind_speeds, marker="o", linestyle="--", label="Wind speed (m/s)")
    ax2.set_xlabel("Indice campione meteo")
    ax2.set_ylabel("Umidità / Vento")
    ax2.set_title("Evoluzione umidità e velocità del vento")
    ax2.grid(True)
    ax2.legend()

    global weather_canvas, weather_fig
    weather_fig = weather_fig_local
    weather_canvas = FigureCanvasTkAgg(weather_fig_local, master=weather_plot_frame)
    weather_canvas.get_tk_widget().pack(fill="both", expand=True)
    weather_canvas.draw()

    status_var.set(
        f"Dati meteo caricati per la sessione {session_key}."
    )

    update_weather_performance_plot(session_key)


def update_weather_performance_plot(session_key: int, driver_label: str = None):
    """Mostra la correlazione track temp vs lap time per il pilota attivo."""
    global weather_perf_canvas, weather_perf_fig

    clear_weather_performance_plot()

    if session_key is None:
        return

    if not weather_last_data or weather_last_session_key != session_key:
        if weather_perf_info_var is not None:
            weather_perf_info_var.set(
                "Seleziona una sessione e scarica il meteo per abilitare la correlazione."
            )
        return

    if not current_laps_data or current_laps_session_key != session_key:
        if weather_perf_info_var is not None:
            weather_perf_info_var.set(
                "Seleziona un pilota e carica i suoi giri per vedere la correlazione con la track temp."
            )
        return

    weather_points = []
    for w in weather_last_data:
        dt = parse_iso_datetime(w.get("date", ""))
        track_temp = w.get("track_temperature", None)
        if dt is None or not isinstance(track_temp, (int, float)):
            continue
        weather_points.append((dt, float(track_temp)))

    if not weather_points:
        if weather_perf_info_var is not None:
            weather_perf_info_var.set(
                "Impossibile calcolare la correlazione: nessun campione meteo con data/track temp valido."
            )
        return

    lap_points = []
    for lap in current_laps_data:
        if lap.get("is_pit_out_lap", False):
            continue
        lap_time = lap.get("lap_duration", None)
        lap_dt = parse_iso_datetime(lap.get("date_start", ""))
        if not isinstance(lap_time, (int, float)) or lap_dt is None:
            continue
        lap_points.append((float(lap_time), lap_dt, lap.get("lap_number", "")))

    if not lap_points:
        if weather_perf_info_var is not None:
            weather_perf_info_var.set(
                "Nessun giro valido (no out-lap, con lap_duration e timestamp) per la correlazione."
            )
        return

    matched = []
    for lap_time, lap_dt, lap_number in lap_points:
        nearest = min(
            weather_points,
            key=lambda wp: abs((lap_dt - wp[0]).total_seconds()),
        )
        matched.append(
            {
                "track_temp": nearest[1],
                "lap_time": lap_time,
                "lap_number": lap_number,
            }
        )

    if not matched:
        if weather_perf_info_var is not None:
            weather_perf_info_var.set(
                "Impossibile calcolare la correlazione: accoppiamento lap/meteo non riuscito."
            )
        return

    x_vals = [m["track_temp"] for m in matched]
    y_vals = [m["lap_time"] for m in matched]

    # Calcolo correlazione Pearson e retta di regressione semplice
    corr = None
    slope = None
    intercept = None
    if len(x_vals) >= 2:
        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(y_vals) / len(y_vals)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        den_x = sum((x - mean_x) ** 2 for x in x_vals)
        den_y = sum((y - mean_y) ** 2 for y in y_vals)
        if den_x > 0 and den_y > 0:
            corr = num / math.sqrt(den_x * den_y)
        if den_x > 0:
            slope = num / den_x
            intercept = mean_y - slope * mean_x

    fig_local = Figure(figsize=(6, 3.2))
    ax = fig_local.add_subplot(111)
    scatter = ax.scatter(x_vals, y_vals, c="tab:blue", alpha=0.8)

    if slope is not None and intercept is not None:
        xs_line = [min(x_vals), max(x_vals)]
        ys_line = [slope * x + intercept for x in xs_line]
        ax.plot(xs_line, ys_line, color="tab:orange", linestyle="--", label="Trend line")

    driver_title = f" - {driver_label}" if driver_label else ""
    ax.set_title(f"Track temp vs lap time{driver_title}")
    ax.set_xlabel("Track temperature (°C)")
    ax.set_ylabel("Lap time (s)")
    ax.grid(True)
    if slope is not None:
        ax.legend()

    weather_perf_fig = fig_local
    weather_perf_canvas = FigureCanvasTkAgg(fig_local, master=weather_perf_plot_frame)
    weather_perf_canvas.get_tk_widget().pack(fill="both", expand=True)
    weather_perf_canvas.draw()

    info_lines = [
        f"Giri abbinati: {len(matched)} (out-lap escluse, accoppiamento per timestamp più vicino)."
    ]
    if corr is not None:
        info_lines.append(f"Correlazione (Pearson): {corr:.2f}")
    if weather_perf_info_var is not None:
        weather_perf_info_var.set(" ".join(info_lines))


# --------------------- Statistiche piloti (lap time & consistenza) --------------------- #

def on_compute_session_stats_click():
    """
    Calcola statistiche di performance giro per ogni pilota della sessione:

      - Lap utili (lap_duration valido, is_pit_out_lap=False)
      - Tempo medio sul giro
      - Deviazione standard dei tempi sul giro
      - Miglior giro
      - Gap dal miglior giro assoluto della sessione
    """
    global stats_tree

    session_key, session_type, meeting_key = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    result_items = results_tree.get_children()
    if not result_items:
        messagebox.showinfo(
            "Info",
            "Carica prima i risultati della sessione, poi calcola le statistiche."
        )
        return

    clear_stats_table()

    drivers_info = []
    for item in result_items:
        vals = results_tree.item(item, "values")
        if not vals or len(vals) < 4:
            continue
        position = vals[0]
        driver_number = vals[1]
        driver_name = vals[2]
        team_name = vals[3]
        drivers_info.append(
            {
                "position": position,
                "driver_number": driver_number,
                "driver_name": driver_name,
                "team_name": team_name,
            }
        )

    if not drivers_info:
        messagebox.showinfo(
            "Info",
            "Nessun pilota valido trovato nei risultati per calcolare le statistiche."
        )
        return

    status_var.set(
        f"Calcolo statistiche lap time per {len(drivers_info)} piloti (sessione {session_key})..."
    )
    root.update_idletasks()

    stats_per_driver = []
    best_lap_session = None

    for d in drivers_info:
        dn = d["driver_number"]
        try:
            int_dn = int(dn)
        except (ValueError, TypeError):
            stats_per_driver.append(
                {
                    **d,
                    "laps_used": 0,
                    "avg": None,
                    "std": None,
                    "best": None,
                }
            )
            continue

        try:
            laps_data = fetch_laps(session_key, int_dn)
        except RuntimeError:
            stats_per_driver.append(
                {
                    **d,
                    "laps_used": 0,
                    "avg": None,
                    "std": None,
                    "best": None,
                }
            )
            continue

        lap_times = []
        for lap in laps_data:
            is_out = bool(lap.get("is_pit_out_lap", False))
            if is_out:
                continue
            ld = lap.get("lap_duration", None)
            if isinstance(ld, (int, float)):
                lap_times.append(float(ld))

        if not lap_times:
            stats_per_driver.append(
                {
                    **d,
                    "laps_used": 0,
                    "avg": None,
                    "std": None,
                    "best": None,
                }
            )
            continue

        n = len(lap_times)
        mean_val = sum(lap_times) / n
        var = sum((t - mean_val) ** 2 for t in lap_times) / n
        std_val = math.sqrt(var)
        best_val = min(lap_times)

        stats_per_driver.append(
            {
                **d,
                "laps_used": n,
                "avg": mean_val,
                "std": std_val,
                "best": best_val,
            }
        )

        if best_lap_session is None or best_val < best_lap_session:
            best_lap_session = best_val

    for s in stats_per_driver:
        laps_used = s["laps_used"]
        avg_val = s["avg"]
        std_val = s["std"]
        best_val = s["best"]

        avg_str = format_time_from_seconds(avg_val) if isinstance(avg_val, (int, float)) else ""
        std_str = format_time_from_seconds(std_val) if isinstance(std_val, (int, float)) else ""
        best_str = format_time_from_seconds(best_val) if isinstance(best_val, (int, float)) else ""

        if isinstance(best_val, (int, float)) and isinstance(best_lap_session, (int, float)):
            gap_best = best_val - best_lap_session
            gap_best_str = f"{gap_best:.3f}"
        else:
            gap_best_str = ""

        stats_tree.insert(
            "",
            tk.END,
            values=(
                s["position"],
                s["driver_number"],
                s["driver_name"],
                s.get("team_name", ""),
                laps_used,
                avg_str,
                std_str,
                best_str,
                gap_best_str,
            ),
        )

    plots_notebook.select(stats_tab_frame)

    status_var.set(
        "Statistiche lap time per pilota calcolate. "
        "I giri usati escludono le out-lap (is_pit_out_lap=True) e richiedono un lap_duration numerico."
    )


# --------------------- Analisi gomme & degrado --------------------- #

def populate_stints_combo():
    """Aggiorna la combo degli stint in base a current_stints_data."""
    global current_stints_for_combo, stints_combo

    current_stints_for_combo = []
    if stints_combo is None:
        return

    stints_combo["values"] = ()
    stints_combo.set("")

    if not current_stints_data:
        return

    try:
        sorted_stints = sorted(
            current_stints_data,
            key=lambda s: (
                s.get("stint_number", 9999)
                if isinstance(s.get("stint_number"), int)
                else s.get("lap_start", 9999)
            )
        )
    except Exception:
        sorted_stints = current_stints_data

    labels = []
    for s in sorted_stints:
        stint_num = s.get("stint_number", "")
        compound = s.get("compound", "")
        lap_start = s.get("lap_start", "")
        lap_end = s.get("lap_end", "")
        label = f"Stint {stint_num} - {compound} ({lap_start}-{lap_end})"
        labels.append(label)
        current_stints_for_combo.append(s)

    stints_combo["values"] = labels
    if labels:
        stints_combo.current(0)


def update_stints_map_plot():
    """Disegna la mappa stint (bar orizzontali) sul canvas stints_plot_canvas."""
    global stints_plot_canvas

    if stints_plot_canvas is not None:
        stints_plot_canvas.get_tk_widget().destroy()
        stints_plot_canvas = None

    if not current_stints_data:
        return

    try:
        stints_sorted = sorted(
            current_stints_data,
            key=lambda s: (
                s.get("stint_number", 9999)
                if isinstance(s.get("stint_number"), int)
                else s.get("lap_start", 9999)
            )
        )
    except Exception:
        stints_sorted = current_stints_data

    compound_colors = {
        "SOFT": "red",
        "MEDIUM": "yellow",
        "HARD": "white",
        "INTERMEDIATE": "green",
        "WET": "blue",
    }

    fig_stints = Figure(figsize=(6, 3.0))
    ax_stints = fig_stints.add_subplot(111)

    y_positions = []
    y_labels = []
    patches_used = {}

    for idx, stint in enumerate(stints_sorted, start=1):
        lap_start = stint.get("lap_start", None)
        lap_end = stint.get("lap_end", None)
        compound = stint.get("compound", "")

        if not isinstance(lap_start, int) or not isinstance(lap_end, int):
            continue

        width = lap_end - lap_start + 1
        if width <= 0:
            continue

        color = compound_colors.get(compound, None)
        y = idx
        y_positions.append(y)
        y_labels.append(f"Stint {stint.get('stint_number', idx)}")

        ax_stints.barh(
            y=y,
            width=width,
            left=lap_start,
            height=0.6,
            color=color,
            edgecolor="black"
        )

        if compound and compound in compound_colors and compound not in patches_used:
            patches_used[compound] = compound_colors[compound]

    if y_positions:
        ax_stints.set_yticks(y_positions)
        ax_stints.set_yticklabels(y_labels)
    ax_stints.set_xlabel("Giro")
    ax_stints.set_title("Mappa stint gomme")
    ax_stints.grid(True, axis="x", linestyle="--", linewidth=0.5)

    if patches_used:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=col, edgecolor="black", label=comp)
            for comp, col in patches_used.items()
        ]
        ax_stints.legend(handles=legend_handles, title="Compound")

    stints_plot_canvas_local = FigureCanvasTkAgg(fig_stints, master=stints_plot_canvas_frame)
    stints_plot_canvas_local.get_tk_widget().pack(fill="both", expand=True)
    stints_plot_canvas_local.draw()

    stints_plot_canvas = stints_plot_canvas_local


def update_stints_summary_table():
    """Calcola e mostra le statistiche per stint nella tabella riassuntiva."""
    clear_stints_summary_table()

    if not current_stints_data or not current_laps_data:
        return

    laps_by_number = {}
    for lap in current_laps_data:
        lap_num = lap.get("lap_number", None)
        if not isinstance(lap_num, int):
            continue
        ld = lap.get("lap_duration", None)
        is_out = bool(lap.get("is_pit_out_lap", False))
        laps_by_number[lap_num] = (ld, is_out)

    try:
        stints_sorted = sorted(
            current_stints_data,
            key=lambda s: (
                s.get("stint_number", 9999)
                if isinstance(s.get("stint_number"), int)
                else s.get("lap_start", 9999)
            )
        )
    except Exception:
        stints_sorted = current_stints_data

    prev_avg = None

    for stint in stints_sorted:
        stint_num = stint.get("stint_number", "")
        compound = stint.get("compound", "")
        lap_start = stint.get("lap_start", None)
        lap_end = stint.get("lap_end", None)

        if not isinstance(lap_start, int) or not isinstance(lap_end, int):
            continue

        lap_times = []
        for ln in range(lap_start, lap_end + 1):
            if ln not in laps_by_number:
                continue
            ld, is_out = laps_by_number[ln]
            if is_out:
                continue
            if isinstance(ld, (int, float)):
                lap_times.append(float(ld))

        if not lap_times:
            laps_used = 0
            avg_val = None
            best_val = None
        else:
            laps_used = len(lap_times)
            avg_val = sum(lap_times) / laps_used
            best_val = min(lap_times)

        avg_str = format_time_from_seconds(avg_val) if isinstance(avg_val, (int, float)) else ""
        best_str = format_time_from_seconds(best_val) if isinstance(best_val, (int, float)) else ""

        if isinstance(avg_val, (int, float)) and isinstance(prev_avg, (int, float)):
            delta = avg_val - prev_avg
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.3f}"
        else:
            delta_str = ""

        stints_summary_tree.insert(
            "",
            tk.END,
            values=(
                stint_num,
                compound,
                f"{lap_start}-{lap_end}",
                laps_used,
                avg_str,
                best_str,
                delta_str,
            ),
        )

        if isinstance(avg_val, (int, float)):
            prev_avg = avg_val


def on_stints_mode_changed():
    """Callback cambio radio button modalità stints (mappa vs analisi)."""
    global stints_plot_canvas

    mode = stints_mode_var.get()
    if mode == "map":
        update_stints_map_plot()
    else:
        # In modalità analisi mostriamo una figura "vuota" con messaggio
        if stints_plot_canvas is not None:
            stints_plot_canvas.get_tk_widget().destroy()
            stints_plot_canvas = None

        fig = Figure(figsize=(6, 3.0))
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Seleziona uno stint e usa i pulsanti\nper mostrare degrado o media per compound.",
            ha="center",
            va="center",
            wrap=True
        )
        ax.axis("off")

        stints_plot_canvas = FigureCanvasTkAgg(fig, master=stints_plot_canvas_frame)
        stints_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        stints_plot_canvas.draw()


def on_show_stint_degradation_click():
    """Mostra il grafico di degrado (lap vs lap_duration) per lo stint selezionato."""
    global stints_plot_canvas

    if not current_stints_for_combo or not current_laps_data:
        messagebox.showinfo(
            "Info",
            "Nessun dato di stint/giri disponibile per questo pilota."
        )
        return

    if stints_combo.current() < 0:
        messagebox.showinfo(
            "Info",
            "Seleziona uno stint dall'elenco prima di mostrare il degrado."
        )
        return

    stint = current_stints_for_combo[stints_combo.current()]
    lap_start = stint.get("lap_start", None)
    lap_end = stint.get("lap_end", None)
    compound = stint.get("compound", "")

    if not isinstance(lap_start, int) or not isinstance(lap_end, int):
        messagebox.showinfo(
            "Info",
            "Lo stint selezionato non ha un range di giri valido."
        )
        return

    laps = []
    times = []
    for lap in current_laps_data:
        ln = lap.get("lap_number", None)
        if not isinstance(ln, int):
            continue
        if ln < lap_start or ln > lap_end:
            continue
        is_out = bool(lap.get("is_pit_out_lap", False))
        if is_out:
            continue
        ld = lap.get("lap_duration", None)
        if isinstance(ld, (int, float)):
            laps.append(ln)
            times.append(float(ld))

    if not laps:
        messagebox.showinfo(
            "Info",
            "Nessun giro valido (non out-lap con lap_duration numerico) trovato per lo stint selezionato."
        )
        return

    if stints_plot_canvas is not None:
        stints_plot_canvas.get_tk_widget().destroy()
        stints_plot_canvas = None

    fig = Figure(figsize=(6, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(laps, times, marker="o")
    ax.set_xlabel("Giro")
    ax.set_ylabel("Lap time (s)")
    ax.set_title(f"Degrado stint {stint.get('stint_number', '')} - {compound}")
    ax.grid(True)

    stints_plot_canvas_local = FigureCanvasTkAgg(fig, master=stints_plot_canvas_frame)
    stints_plot_canvas_local.get_tk_widget().pack(fill="both", expand=True)
    stints_plot_canvas_local.draw()
    stints_plot_canvas = stints_plot_canvas_local


def on_show_compound_avg_click():
    """Mostra un bar chart con il tempo medio per compound per il pilota corrente."""
    global stints_plot_canvas

    if not current_stints_data or not current_laps_data:
        messagebox.showinfo(
            "Info",
            "Nessun dato di stint/giri disponibile per questo pilota."
        )
        return

    laps_by_number = {}
    for lap in current_laps_data:
        ln = lap.get("lap_number", None)
        if not isinstance(ln, int):
            continue
        laps_by_number[ln] = lap

    compound_times = {}
    for stint in current_stints_data:
        compound = stint.get("compound", "")
        lap_start = stint.get("lap_start", None)
        lap_end = stint.get("lap_end", None)
        if not isinstance(lap_start, int) or not isinstance(lap_end, int):
            continue
        if not compound:
            continue

        for ln in range(lap_start, lap_end + 1):
            lap = laps_by_number.get(ln)
            if not lap:
                continue
            is_out = bool(lap.get("is_pit_out_lap", False))
            if is_out:
                continue
            ld = lap.get("lap_duration", None)
            if isinstance(ld, (int, float)):
                compound_times.setdefault(compound, []).append(float(ld))

    if not compound_times:
        messagebox.showinfo(
            "Info",
            "Nessun giro valido per calcolare le medie per compound."
        )
        return

    compounds = []
    avgs = []
    for comp, times in compound_times.items():
        if not times:
            continue
        compounds.append(comp)
        avgs.append(sum(times) / len(times))

    if not compounds:
        messagebox.showinfo(
            "Info",
            "Nessun dato utilizzabile per il grafico di media per compound."
        )
        return

    if stints_plot_canvas is not None:
        stints_plot_canvas.get_tk_widget().destroy()
        stints_plot_canvas = None

    fig = Figure(figsize=(6, 3.0))
    ax = fig.add_subplot(111)
    x_positions = list(range(len(compounds)))
    ax.bar(x_positions, avgs)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(compounds)
    ax.set_ylabel("Lap time medio (s)")
    ax.set_title("Tempo medio per compound (giri non out-lap)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    stints_plot_canvas_local = FigureCanvasTkAgg(fig, master=stints_plot_canvas_frame)
    stints_plot_canvas_local.get_tk_widget().pack(fill="both", expand=True)
    stints_plot_canvas_local.draw()
    stints_plot_canvas = stints_plot_canvas_local


# --------------------- Analisi pit stop & strategia --------------------- #

def on_compute_pit_strategy_click():
    """
    Analizza i pit stop della sessione (Race/Sprint):
      - Tabella riassuntiva per pilota (num pit, pit medio, best, worst, giri).
      - Grafico numero pit stop per giro.
    """
    global current_pit_data, current_results_data
    global pit_stats_tree, pit_strategy_canvas, pit_strategy_fig

    session_key, session_type, meeting_key = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "L'analisi pit stop è disponibile solo per sessioni di tipo Race o Sprint."
        )
        return

    if not current_pit_data:
        messagebox.showinfo(
            "Info",
            "Nessun pit stop disponibile per questa sessione."
        )
        return

    if not current_results_data:
        messagebox.showinfo(
            "Info",
            "Prima carica i risultati della sessione (bottone 'Mostra risultati...')."
        )
        return

    clear_pit_strategy()

    # Mappa driver_number -> total_laps della sessione
    laps_by_driver = {}
    for r in current_results_data:
        dn = r.get("driver_number", None)
        tot_laps = r.get("number_of_laps", None)
        if isinstance(dn, int) and isinstance(tot_laps, int):
            laps_by_driver[dn] = tot_laps

    # Aggregazione per pilota
    per_driver = {}  # dn -> {"durations":[...], "laps":[...], "name":..., "total_laps":...}
    for p in current_pit_data:
        dnum = p.get("driver_number", None)
        lap_number = p.get("lap_number", None)
        pit_duration = p.get("pit_duration", None)
        if not isinstance(dnum, int) or not isinstance(lap_number, int):
            continue
        if not isinstance(pit_duration, (int, float)):
            continue

        per_driver.setdefault(
            dnum,
            {
                "durations": [],
                "laps": [],
                "name": "",
                "team_name": "",
                "total_laps": laps_by_driver.get(dnum, None),
            },
        )
        per_driver[dnum]["durations"].append(float(pit_duration))
        per_driver[dnum]["laps"].append(lap_number)

    # Popola tabella riassuntiva
    for dnum, info in per_driver.items():
        durations = info["durations"]
        laps_list = info["laps"]

        if not durations:
            continue

        avg_pit = sum(durations) / len(durations)
        best_pit = min(durations)
        worst_pit = max(durations)

        # Nome pilota
        name = info.get("name", "")
        if not name:
            name = fetch_driver_full_name(dnum, session_key)
            info["name"] = name
        team_name = info.get("team_name", "")
        if not team_name:
            team_name = fetch_driver_team_name(dnum, session_key)
            info["team_name"] = team_name

        laps_str = ", ".join(str(l) for l in sorted(laps_list))

        pit_stats_tree.insert(
            "",
            tk.END,
            values=(
                name,
                team_name,
                len(durations),
                f"{avg_pit:.3f}",
                f"{best_pit:.3f}",
                f"{worst_pit:.3f}",
                laps_str,
            ),
        )

    # Grafico pit stop per giro
    # Conta quanti pit ci sono stati in ogni giro
    pit_count_by_lap = {}
    max_lap_seen = 0

    for p in current_pit_data:
        lap_number = p.get("lap_number", None)
        if not isinstance(lap_number, int):
            continue
        pit_count_by_lap[lap_number] = pit_count_by_lap.get(lap_number, 0) + 1
        if lap_number > max_lap_seen:
            max_lap_seen = lap_number

    if max_lap_seen > 0:
        laps_x = list(range(1, max_lap_seen + 1))
        counts = [pit_count_by_lap.get(l, 0) for l in laps_x]

        pit_strategy_fig_local = Figure(figsize=(6, 3.0))
        ax = pit_strategy_fig_local.add_subplot(111)
        ax.bar(laps_x, counts)
        ax.set_xlabel("Giro")
        ax.set_ylabel("Numero pit stop")
        ax.set_title("Numero pit stop per giro (sessione)")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

        pit_strategy_canvas_local = FigureCanvasTkAgg(
            pit_strategy_fig_local, master=pit_strategy_plot_frame
        )
        pit_strategy_canvas_local.get_tk_widget().pack(fill="both", expand=True)
        pit_strategy_canvas_local.draw()

        pit_strategy_canvas = pit_strategy_canvas_local
        pit_strategy_fig = pit_strategy_fig_local

    plots_notebook.select(pit_strategy_tab_frame)

    status_var.set(
        "Analisi pit stop aggiornata: tabella riassuntiva per pilota e grafico pit stop per giro "
        "disponibili nella tab 'Pit stop & strategia'."
    )


# --------------------- Callback GUI principali --------------------- #

def on_fetch_sessions_click():
    """Carica l'elenco sessioni per l'anno inserito."""
    global DRIVER_CACHE, DRIVER_PROFILE_CACHE
    DRIVER_CACHE = {}
    DRIVER_PROFILE_CACHE = {}
    clear_driver_plots()
    clear_weather_plot()
    clear_stats_table()
    clear_pit_strategy()
    clear_race_control_table()

    year_str = year_entry.get().strip()
    if not year_str.isdigit() or len(year_str) != 4:
        messagebox.showerror("Errore", "Inserisci un anno valido a 4 cifre (es. 2023).")
        return

    year = int(year_str)
    status_var.set("Scarico le sessioni...")
    root.update_idletasks()

    try:
        sessions = fetch_sessions(year)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero delle sessioni.")
        messagebox.showerror("Errore", str(e))
        return

    for tree in (sessions_tree, results_tree, laps_tree, pits_tree):
        for item in tree.get_children():
            tree.delete(item)
    clear_stats_table()

    if not sessions:
        status_var.set(f"Nessuna sessione trovata per l'anno {year}.")
        return

    for s in sessions:
        date_time = s.get("date_start", "")
        date_part = date_time[:10] if len(date_time) >= 10 else ""
        time_part = date_time[11:16] if len(date_time) >= 16 else ""

        sessions_tree.insert(
            "",
            tk.END,
            values=(
                s.get("year", ""),
                f"{date_part} {time_part}",
                s.get("session_name", ""),
                s.get("session_type", ""),
                s.get("circuit_short_name", ""),
                s.get("location", ""),
                s.get("country_name", ""),
                s.get("session_key", ""),
                s.get("meeting_key", ""),
            ),
        )

    status_var.set(
        f"Trovate {len(sessions)} sessioni per l'anno {year}. "
        "Seleziona una sessione per vedere risultati, meteo e, se Race/Sprint, grafici, giri e pit stop."
    )


def on_fetch_results_click(event=None):
    """Carica risultati, pit stop (se Race/Sprint) e meteo sessione."""
    global DRIVER_CACHE, DRIVER_PROFILE_CACHE, current_pit_data, current_results_data
    DRIVER_CACHE = {}
    DRIVER_PROFILE_CACHE = {}
    clear_driver_plots()
    clear_weather_plot()
    clear_stats_table()
    clear_pit_strategy()
    clear_race_control_table()

    current_pit_data = []
    current_results_data = []

    session_key, session_type, meeting_key = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione nella tabella in alto.")
        return

    status_var.set(f"Scarico i risultati per la sessione {session_key}...")
    root.update_idletasks()

    try:
        results = fetch_session_results(session_key)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero dei risultati.")
        messagebox.showerror("Errore", str(e))
        return

    for tree in (results_tree, laps_tree, pits_tree):
        for item in tree.get_children():
            tree.delete(item)
    clear_stats_table()

    if not results:
        status_var.set(f"Nessun risultato trovato per la sessione {session_key}.")
        update_weather_for_session(session_key)
        return

    try:
        results_sorted = sorted(
            results,
            key=lambda r: (r.get("position", 9999)
                           if isinstance(r.get("position"), int) else 9999)
        )
    except Exception:
        results_sorted = results

    current_results_data = results_sorted

    for r in results_sorted:
        position = r.get("position", "")
        driver_number = r.get("driver_number", "")
        laps = r.get("number_of_laps", "")
        points = r.get("points", "")

        dnf = bool(r.get("dnf", False))
        dns = bool(r.get("dns", False))
        dsq = bool(r.get("dsq", False))

        if dsq:
            status_str = "DSQ"
        elif dns:
            status_str = "DNS"
        elif dnf:
            status_str = "DNF"
        else:
            status_str = "Finished"

        gap = r.get("gap_to_leader", "")

        duration_val = r.get("duration", "")
        if isinstance(duration_val, (int, float)):
            total_seconds = float(duration_val)
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            duration_str = f"{hours:d}:{minutes:02d}:{seconds:06.3f}"
        else:
            duration_str = ""

        full_name = fetch_driver_full_name(driver_number, session_key)
        team_name = fetch_driver_team_name(driver_number, session_key)

        results_tree.insert(
            "",
            tk.END,
            values=(
                position,
                driver_number,
                full_name,
                team_name,
                laps,
                points,
                status_str,
                gap,
                duration_str,
            ),
        )

    if is_race_like(session_type):
        status_var.set(
            f"Mostrati {len(results_sorted)} risultati (Race/Sprint). "
            "Scarico i pit stop della sessione..."
        )
        root.update_idletasks()

        try:
            pit_data = fetch_pit_stops(session_key)
        except RuntimeError as e:
            status_var.set(
                f"Risultati caricati. Errore nel recupero dei pit stop per la sessione {session_key}."
            )
            messagebox.showerror("Errore pit stop", str(e))
            pit_data = []

        current_pit_data = pit_data if isinstance(pit_data, list) else []

        if pit_data:
            try:
                pit_sorted = sorted(
                    pit_data,
                    key=lambda p: p.get("date", "") if isinstance(p.get("date", ""), str) else ""
                )
            except Exception:
                pit_sorted = pit_data

            for p in pit_sorted:
                dnum = p.get("driver_number", "")
                full_name = fetch_driver_full_name(dnum, session_key)
                team_name = fetch_driver_team_name(dnum, session_key)

                lap_number = p.get("lap_number", "")
                pit_duration = p.get("pit_duration", "")
                if isinstance(pit_duration, (int, float)):
                    pit_duration_str = f"{pit_duration:.3f}"
                else:
                    pit_duration_str = ""

                pits_tree.insert(
                    "",
                    tk.END,
                    values=(full_name, team_name, lap_number, pit_duration_str),
                )

            status_var.set(
                f"Mostrati {len(results_sorted)} risultati (Race/Sprint) e "
                f"{len(pit_sorted)} pit stop. "
                "Seleziona un pilota per vedere grafici e tabella giri, calcola le statistiche o analizza i pit."
            )
        else:
            status_var.set(
                f"Mostrati {len(results_sorted)} risultati (Race/Sprint), "
                "ma nessun pit stop disponibile. "
                "Seleziona comunque un pilota per grafici e giri o calcola le statistiche."
            )
    else:
        status_var.set(
            f"Mostrati {len(results_sorted)} risultati (tipo: {session_type}). "
            "Grafici, giri e pit stop sono disponibili solo per Race/Sprint; "
            "le statistiche lap time possono comunque essere calcolate."
        )

    update_weather_for_session(session_key)


def on_show_driver_plots_click():
    """
    Se Race/Sprint e un pilota è selezionato:
      - grafico distacchi
      - grafico stint gomme (mappa + analisi)
      - tabella giri
      - riassunto stint
    Il caricamento dei messaggi Race Control avviene su richiesta tramite pulsante dedicato.
    """
    global gap_plot_canvas, stints_plot_canvas
    global gap_fig, gap_ax, gap_click_cid, gap_click_data
    global current_stints_data, current_laps_data, current_laps_session_key
    global stints_mode_var

    session_key, session_type, meeting_key = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo("Info", "Funzione disponibile solo per sessioni di tipo Race/Sprint.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    title_driver = driver_name if driver_name else f"Driver {driver_number}"

    clear_driver_plots()
    for item in laps_tree.get_children():
        laps_tree.delete(item)

    # --- Grafico distacchi --- #
    status_var.set(
        f"Scarico dati di intervallo per sessione {session_key}, pilota {driver_number}..."
    )
    root.update_idletasks()

    try:
        intervals = fetch_intervals(session_key, driver_number)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero dei dati di intervallo.")
        messagebox.showerror("Errore", str(e))
        intervals = []

    if intervals:
        laps_idx = []
        gaps_leader = []
        gaps_prev = []

        for idx, entry in enumerate(intervals, start=1):
            laps_idx.append(idx)
            gap_leader = entry.get("gap_to_leader", None)
            gaps_leader.append(gap_leader if isinstance(gap_leader, (int, float)) else None)
            interval_val = entry.get("interval", None)
            gaps_prev.append(interval_val if isinstance(interval_val, (int, float)) else None)

        fig_gap = Figure(figsize=(6, 3.2))
        ax_gap = fig_gap.add_subplot(111)

        ax_gap.plot(laps_idx, gaps_leader, marker="o", label="Gap dal leader (s)")
        ax_gap.plot(
            laps_idx, gaps_prev, marker="o", linestyle="--", label="Gap dal pilota davanti (s)"
        )

        ax_gap.set_xlabel("Giro (indice dei dati intervals)")
        ax_gap.set_ylabel("Distacco (secondi)")
        ax_gap.set_title(f"Distacchi giro per giro - {title_driver}")
        ax_gap.grid(True)
        # Statistiche scia/aria pulita e tratti di pressione
        in_scia_pct, aria_pulita_pct = compute_slipstream_stats(gaps_prev)
        if gap_slipstream_var is not None:
            gap_slipstream_var.set(
                f"In scia: {in_scia_pct:.1f}% | Aria pulita: {aria_pulita_pct:.1f}%"
            )

        pressure_segments = find_pressure_stints(laps_idx, gaps_leader, gaps_prev)
        update_pressure_table(pressure_segments)

        marker_styles = {
            "Avvicinamento": {"color": "tab:green", "marker": "^"},
            "Allontanamento": {"color": "tab:red", "marker": "v"},
        }
        used_labels = set()
        for seg in pressure_segments:
            style = marker_styles.get(seg["trend"], {})
            y_val = seg.get("marker_y")
            if y_val is None:
                continue
            label_key = f"{seg['trend']} ({seg['metric']})"
            label = None if label_key in used_labels else label_key
            used_labels.add(label_key)
            ax_gap.scatter(
                seg["marker_x"],
                y_val,
                color=style.get("color", "black"),
                marker=style.get("marker", "o"),
                s=70,
                zorder=3,
                label=label,
            )

        handles, labels = ax_gap.get_legend_handles_labels()
        ax_gap.legend(handles, labels)

        global gap_plot_canvas, gap_fig, gap_ax, gap_click_cid, gap_click_data
        gap_plot_canvas = FigureCanvasTkAgg(fig_gap, master=gap_plot_frame)
        gap_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        gap_plot_canvas.draw()

        gap_fig = fig_gap
        gap_ax = ax_gap
        gap_click_data = {
            "laps": laps_idx,
            "gap_leader": gaps_leader,
            "gap_prev": gaps_prev,
        }
        gap_click_cid = fig_gap.canvas.mpl_connect("button_press_event", on_gap_plot_click)
    else:
        messagebox.showinfo(
            "Info",
            "Nessun dato di intervallo (gap/interval) disponibile per questo pilota in questa sessione."
        )

    # --- Dati stint gomme + laps per analisi gomme & degrado --- #
    status_var.set(
        f"Scarico dati stint gomme per sessione {session_key}, pilota {driver_number}..."
    )
    root.update_idletasks()

    try:
        stints = fetch_stints(session_key, driver_number)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero dei dati degli stint gomme.")
        messagebox.showerror("Errore", str(e))
        stints = []

    current_stints_data = stints if isinstance(stints, list) else []

    status_var.set(
        f"Scarico dati giri per sessione {session_key}, pilota {driver_number}..."
    )
    root.update_idletasks()

    try:
        laps_data = fetch_laps(session_key, driver_number)
    except RuntimeError as e:
        status_var.set("Errore durante il recupero dei dati giri.")
        messagebox.showerror("Errore", str(e))
        laps_data = []

    current_laps_data = laps_data if isinstance(laps_data, list) else []
    current_laps_session_key = session_key

    populate_stints_combo()
    update_stints_summary_table()

    if stints_mode_var is not None:
        stints_mode_var.set("map")
    update_stints_map_plot()
    update_weather_performance_plot(session_key, title_driver)

    # --- Tabella giri --- #
    if laps_data:
        try:
            laps_sorted = sorted(
                laps_data,
                key=lambda l: (l.get("lap_number", 9999)
                               if isinstance(l.get("lap_number"), int) else 9999)
            )
        except Exception:
            laps_sorted = laps_data

        for lap in laps_sorted:
            lap_number = lap.get("lap_number", "")

            lap_time_str = format_time_from_seconds(lap.get("lap_duration", None))
            s1_time_str = format_time_from_seconds(lap.get("duration_sector_1", None))
            s2_time_str = format_time_from_seconds(lap.get("duration_sector_2", None))
            s3_time_str = format_time_from_seconds(lap.get("duration_sector_3", None))

            i1_speed = lap.get("i1_speed", "")
            i2_speed = lap.get("i2_speed", "")
            st_speed = lap.get("st_speed", "")

            i1_speed_str = str(i1_speed) if isinstance(i1_speed, (int, float)) else ""
            i2_speed_str = str(i2_speed) if isinstance(i2_speed, (int, float)) else ""
            st_speed_str = str(st_speed) if isinstance(st_speed, (int, float)) else ""

            is_out = bool(lap.get("is_pit_out_lap", False))
            out_str = "Sì" if is_out else "No"

            tags = ("outlap",) if is_out else ()

            laps_tree.insert(
                "",
                tk.END,
                values=(
                    lap_number,
                    lap_time_str,
                    s1_time_str,
                    s2_time_str,
                    s3_time_str,
                    i1_speed_str,
                    i2_speed_str,
                    st_speed_str,
                    out_str,
                ),
                tags=tags
            )
    else:
        messagebox.showinfo(
            "Info",
            "Nessun dato giri disponibile per questo pilota in questa sessione."
        )

    plots_notebook.select(gap_tab_frame)

    status_var.set(
        f"Grafici, analisi gomme e tabella giri aggiornati per pilota {driver_number} ({title_driver}). "
        "La tabella pit stop mostra tutti i pit della sessione; la tab Meteo mostra l'evoluzione meteo. "
        "Premi 'Recupera messaggi Race Control' per caricare i messaggi della sessione."
    )


def on_fetch_race_control_click():
    """Handler per il pulsante di recupero dei messaggi Race Control."""

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "I messaggi Race Control sono disponibili solo per sessioni Race/Sprint.",
        )
        return

    status_var.set(
        f"Scarico messaggi Race Control per la sessione {session_key}..."
    )
    root.update_idletasks()

    update_race_control_messages(session_key)
    plots_notebook.select(race_control_tab_frame)

    status_var.set(
        f"Messaggi Race Control aggiornati per la sessione {session_key}."
    )


# --------------------- Costruzione GUI --------------------- #

root = tk.Tk()
root.title(
    "F1 OpenF1 Tool: Calendario, Risultati, Distacchi, Gomme, Giri, Pit, Meteo, Statistiche & Strategia"
)
root.geometry("1500x950")
root.minsize(1280, 850)

DARK_BG = "#0b1220"
DARK_PANEL = "#111827"
ACCENT = "#22d3ee"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#9ca3af"


def apply_dark_theme(app_root: tk.Tk):
    style = ttk.Style(app_root)
    style.theme_use("clam")

    style.configure(
        ".",
        background=DARK_BG,
        foreground=TEXT_COLOR,
        fieldbackground=DARK_PANEL,
        relief="flat",
    )
    style.configure("TFrame", background=DARK_BG)
    style.configure("Card.TFrame", background=DARK_PANEL)
    style.configure(
        "Card.TLabelframe",
        background=DARK_PANEL,
        foreground=TEXT_COLOR,
        bordercolor="#1f2937",
        relief="groove",
        padding=6,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=DARK_PANEL,
        foreground=TEXT_COLOR,
        font=("", 10, "bold"),
    )
    style.configure("TLabel", background=DARK_BG, foreground=TEXT_COLOR)
    style.configure("Header.TLabel", background=DARK_BG, foreground=TEXT_COLOR, font=("", 11, "bold"))
    style.configure(
        "Info.TLabel",
        background=DARK_PANEL,
        foreground=MUTED_TEXT,
        wraplength=1200,
    )
    style.configure(
        "Status.TLabel",
        background="#0f172a",
        foreground=TEXT_COLOR,
        padding=6,
    )
    style.configure(
        "TButton",
        background="#1f2937",
        foreground=TEXT_COLOR,
        borderwidth=0,
        padding=(10, 6),
    )
    style.map(
        "TButton",
        background=[("active", "#2563eb"), ("pressed", "#1d4ed8")],
        foreground=[("disabled", MUTED_TEXT)],
    )
    style.configure(
        "Treeview",
        background=DARK_PANEL,
        foreground=TEXT_COLOR,
        fieldbackground=DARK_PANEL,
        bordercolor="#1f2937",
    )
    style.map(
        "Treeview",
        background=[("selected", "#374151")],
        foreground=[("selected", TEXT_COLOR)],
    )
    style.configure(
        "Treeview.Heading",
        background="#111827",
        foreground=TEXT_COLOR,
        bordercolor="#1f2937",
    )
    style.configure(
        "TNotebook",
        background=DARK_BG,
        foreground=TEXT_COLOR,
        tabmargins=2,
    )
    style.configure(
        "TNotebook.Tab",
        background="#1f2937",
        foreground=TEXT_COLOR,
        padding=(10, 6),
    )
    style.map("TNotebook.Tab", background=[("selected", DARK_PANEL)], foreground=[("disabled", MUTED_TEXT)])
    style.configure("TEntry", fieldbackground=DARK_PANEL, foreground=TEXT_COLOR, insertcolor=TEXT_COLOR)
    style.configure("TCombobox", fieldbackground=DARK_PANEL, foreground=TEXT_COLOR)

    app_root.option_add("*TCombobox*Listbox*Background", DARK_PANEL)
    app_root.option_add("*TCombobox*Listbox*Foreground", TEXT_COLOR)


apply_dark_theme(root)
root.configure(bg=DARK_BG)

# --- Barra comandi e input --- #
input_frame = ttk.Frame(root, padding=(12, 10), style="Card.TFrame")
input_frame.pack(fill="x", padx=10, pady=(10, 6))

year_block = ttk.Frame(input_frame, style="Card.TFrame")
year_block.pack(side="left", fill="x", expand=True)

ttk.Label(year_block, text="Anno (es. 2023):", style="Header.TLabel").pack(side="left")
year_entry = ttk.Entry(year_block, width=10)
year_entry.pack(side="left", padx=6)
year_entry.insert(0, "2023")

fetch_sessions_button = ttk.Button(
    year_block,
    text="Recupera calendario",
    command=on_fetch_sessions_click,
)
fetch_sessions_button.pack(side="left", padx=6)

actions_frame = ttk.Frame(input_frame, style="Card.TFrame")
actions_frame.pack(side="right", fill="x")

fetch_results_button = ttk.Button(
    actions_frame,
    text="Mostra risultati sessione",
    command=on_fetch_results_click,
)
fetch_results_button.grid(row=0, column=0, padx=4, pady=2, sticky="ew")

plot_button = ttk.Button(
    actions_frame,
    text="Grafici & giri pilota",
    command=on_show_driver_plots_click,
)
plot_button.grid(row=0, column=1, padx=4, pady=2, sticky="ew")

stats_button = ttk.Button(
    actions_frame,
    text="Statistiche lap time",
    command=on_compute_session_stats_click,
)
stats_button.grid(row=0, column=2, padx=4, pady=2, sticky="ew")

pit_strategy_button = ttk.Button(
    actions_frame,
    text="Pit stop & strategia",
    command=on_compute_pit_strategy_click,
)
pit_strategy_button.grid(row=0, column=3, padx=4, pady=2, sticky="ew")

race_control_action = ttk.Button(
    actions_frame,
    text="Race Control pilota",
    command=on_fetch_race_control_click,
)
race_control_action.grid(row=0, column=4, padx=4, pady=2, sticky="ew")

actions_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)

# --- Layout principale con paned window per mostrare tutte le sezioni --- #
main_paned = ttk.Panedwindow(root, orient=tk.VERTICAL)
main_paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

top_paned = ttk.Panedwindow(main_paned, orient=tk.HORIZONTAL)
main_paned.add(top_paned, weight=3)

sessions_panel = ttk.Labelframe(
    top_paned,
    text="Calendario sessioni",
    style="Card.TLabelframe",
)
top_paned.add(sessions_panel, weight=2)

ttk.Label(
    sessions_panel,
    text="Doppio click per caricare i risultati della sessione.",
    style="Info.TLabel",
    anchor="w",
).pack(fill="x", pady=(0, 4))

sessions_frame = ttk.Frame(sessions_panel, style="Card.TFrame")
sessions_frame.pack(fill="both", expand=True)

sessions_columns = (
    "year",
    "datetime",
    "session_name",
    "session_type",
    "circuit",
    "location",
    "country",
    "session_key",
    "meeting_key",
)

sessions_tree = ttk.Treeview(
    sessions_frame,
    columns=sessions_columns,
    show="headings",
    height=10,
)

sessions_tree.heading("year", text="Anno")
sessions_tree.heading("datetime", text="Data / Ora (UTC)")
sessions_tree.heading("session_name", text="Sessione")
sessions_tree.heading("session_type", text="Tipo")
sessions_tree.heading("circuit", text="Circuito")
sessions_tree.heading("location", text="Località")
sessions_tree.heading("country", text="Paese")
sessions_tree.heading("session_key", text="Session Key")
sessions_tree.heading("meeting_key", text="Meeting Key")

sessions_tree.column("year", width=60, anchor="center")
sessions_tree.column("datetime", width=170, anchor="center")
sessions_tree.column("session_name", width=190)
sessions_tree.column("session_type", width=90, anchor="center")
sessions_tree.column("circuit", width=160)
sessions_tree.column("location", width=160)
sessions_tree.column("country", width=150)

sessions_tree.column("session_key", width=100, anchor="center")
sessions_tree.column("meeting_key", width=110, anchor="center")

sessions_vsb = ttk.Scrollbar(
    sessions_frame,
    orient="vertical",
    command=sessions_tree.yview,
)
sessions_tree.configure(yscrollcommand=sessions_vsb.set)

sessions_tree.grid(row=0, column=0, sticky="nsew")
sessions_vsb.grid(row=0, column=1, sticky="ns")

sessions_frame.rowconfigure(0, weight=1)
sessions_frame.columnconfigure(0, weight=1)

sessions_tree.bind("<Double-1>", on_fetch_results_click)

right_paned = ttk.Panedwindow(top_paned, orient=tk.VERTICAL)
top_paned.add(right_paned, weight=3)

results_panel = ttk.Labelframe(
    right_paned, text="Risultati sessione", style="Card.TLabelframe"
)
right_paned.add(results_panel, weight=2)

results_frame = ttk.Frame(results_panel, style="Card.TFrame")
results_frame.pack(fill="both", expand=True)

results_columns = (
    "position",
    "driver_number",
    "driver_name",
    "team_name",
    "laps",
    "points",
    "status",
    "gap",
    "duration",
)

results_tree = ttk.Treeview(
    results_frame,
    columns=results_columns,
    show="headings",
    height=8,
)

results_tree.heading("position", text="Pos.")
results_tree.heading("driver_number", text="N°")
results_tree.heading("driver_name", text="Pilota")
results_tree.heading("team_name", text="Team")
results_tree.heading("laps", text="Giri")
results_tree.heading("points", text="Punti")
results_tree.heading("status", text="Stato")
results_tree.heading("gap", text="Gap")
results_tree.heading("duration", text="Durata")

results_tree.column("position", width=50, anchor="center")
results_tree.column("driver_number", width=50, anchor="center")
results_tree.column("driver_name", width=170, anchor="w")
results_tree.column("team_name", width=170, anchor="w")
results_tree.column("laps", width=70, anchor="center")
results_tree.column("points", width=70, anchor="center")
results_tree.column("status", width=130, anchor="w")
results_tree.column("gap", width=110, anchor="center")
results_tree.column("duration", width=130, anchor="center")

results_vsb = ttk.Scrollbar(
    results_frame,
    orient="vertical",
    command=results_tree.yview,
)
results_tree.configure(yscrollcommand=results_vsb.set)

results_tree.grid(row=0, column=0, sticky="nsew")
results_vsb.grid(row=0, column=1, sticky="ns")

results_frame.rowconfigure(0, weight=1)
results_frame.columnconfigure(0, weight=1)

pilot_panel = ttk.Labelframe(
    right_paned,
    text="Dettaglio pilota selezionato (giri + pit)",
    style="Card.TLabelframe",
)
right_paned.add(pilot_panel, weight=3)

pilot_paned = ttk.Panedwindow(pilot_panel, orient=tk.HORIZONTAL)
pilot_paned.pack(fill="both", expand=True)

laps_frame = ttk.Labelframe(
    pilot_paned, text="Giri del pilota", style="Card.TLabelframe"
)
pilot_paned.add(laps_frame, weight=3)

laps_table_frame = ttk.Frame(laps_frame, style="Card.TFrame")
laps_table_frame.pack(fill="both", expand=True)

laps_columns = (
    "lap_number",
    "lap_time",
    "s1_time",
    "s2_time",
    "s3_time",
    "s1_speed",
    "s2_speed",
    "st_speed",
    "out_lap",
)

laps_tree = ttk.Treeview(
    laps_table_frame,
    columns=laps_columns,
    show="headings",
    height=8,
)

laps_tree.heading("lap_number", text="Giro")
laps_tree.heading("lap_time", text="Lap Time")
laps_tree.heading("s1_time", text="Settore 1")
laps_tree.heading("s2_time", text="Settore 2")
laps_tree.heading("s3_time", text="Settore 3")
laps_tree.heading("s1_speed", text="S1 Speed")
laps_tree.heading("s2_speed", text="S2 Speed")
laps_tree.heading("st_speed", text="Speed Trap")
laps_tree.heading("out_lap", text="Out Lap")

laps_tree.column("lap_number", width=55, anchor="center")
laps_tree.column("lap_time", width=95, anchor="center")
laps_tree.column("s1_time", width=95, anchor="center")
laps_tree.column("s2_time", width=95, anchor="center")
laps_tree.column("s3_time", width=95, anchor="center")
laps_tree.column("s1_speed", width=90, anchor="center")
laps_tree.column("s2_speed", width=90, anchor="center")
laps_tree.column("st_speed", width=90, anchor="center")
laps_tree.column("out_lap", width=70, anchor="center")

laps_tree.tag_configure("outlap", background="#312e81", foreground=TEXT_COLOR)

laps_vsb = ttk.Scrollbar(
    laps_table_frame,
    orient="vertical",
    command=laps_tree.yview,
)
laps_tree.configure(yscrollcommand=laps_vsb.set)

laps_tree.grid(row=0, column=0, sticky="nsew")
laps_vsb.grid(row=0, column=1, sticky="ns")

laps_table_frame.rowconfigure(0, weight=1)
laps_table_frame.columnconfigure(0, weight=1)

pits_frame = ttk.Labelframe(
    pilot_paned, text="Pit stop sessione", style="Card.TLabelframe"
)
pilot_paned.add(pits_frame, weight=2)

pits_table_frame = ttk.Frame(pits_frame, style="Card.TFrame")
pits_table_frame.pack(fill="both", expand=True)

pits_columns = (
    "pit_driver_name",
    "pit_team_name",
    "pit_lap_number",
    "pit_duration",
)

pits_tree = ttk.Treeview(
    pits_table_frame,
    columns=pits_columns,
    show="headings",
    height=6,
)

pits_tree.heading("pit_driver_name", text="Pilota")
pits_tree.heading("pit_team_name", text="Team")
pits_tree.heading("pit_lap_number", text="Giro")
pits_tree.heading("pit_duration", text="Pit Time (s)")

pits_tree.column("pit_driver_name", width=170, anchor="w")
pits_tree.column("pit_team_name", width=170, anchor="w")
pits_tree.column("pit_lap_number", width=70, anchor="center")
pits_tree.column("pit_duration", width=90, anchor="center")

pits_vsb = ttk.Scrollbar(
    pits_table_frame,
    orient="vertical",
    command=pits_tree.yview,
)
pits_tree.configure(yscrollcommand=pits_vsb.set)

pits_tree.grid(row=0, column=0, sticky="nsew")
pits_vsb.grid(row=0, column=1, sticky="ns")

pits_table_frame.rowconfigure(0, weight=1)
pits_table_frame.columnconfigure(0, weight=1)

# --- Notebook inferiore per grafici e analisi --- #
plots_shell = ttk.Labelframe(
    main_paned,
    text="Analisi grafica e output completi",
    style="Card.TLabelframe",
)
main_paned.add(plots_shell, weight=4)

plots_notebook = ttk.Notebook(plots_shell)
plots_notebook.pack(fill="both", expand=True, padx=4, pady=4)

gap_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")
stints_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")
race_control_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")
weather_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")
stats_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")
pit_strategy_tab_frame = ttk.Frame(plots_notebook, padding=6, style="Card.TFrame")

plots_notebook.add(gap_tab_frame, text="Grafico distacchi")
plots_notebook.add(stints_tab_frame, text="Gomme: mappa & analisi")
plots_notebook.add(race_control_tab_frame, text="Race Control")
plots_notebook.add(weather_tab_frame, text="Meteo sessione")
plots_notebook.add(stats_tab_frame, text="Statistiche piloti")
plots_notebook.add(pit_strategy_tab_frame, text="Pit stop & strategia")

# --- Contenuto tab Grafico distacchi --- #
gap_info_frame = ttk.Frame(gap_tab_frame, style="Card.TFrame")
gap_info_frame.pack(fill="x", padx=5, pady=(5, 2))

gap_slipstream_var = tk.StringVar(value="In scia: -- | Aria pulita: --")
gap_slipstream_label = ttk.Label(
    gap_info_frame,
    textvariable=gap_slipstream_var,
    font=("", 10, "bold"),
    anchor="w",
)
gap_slipstream_label.pack(side="left")

ttk.Label(
    gap_info_frame,
    text="Soglie: interval < 1.0s = scia, interval > 2.5s = aria pulita",
).pack(side="left", padx=(10, 0))

gap_plot_frame = ttk.Frame(gap_tab_frame, style="Card.TFrame")
gap_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 4))

gap_pressure_frame = ttk.LabelFrame(
    gap_tab_frame,
    text="Stints di pressione (variazioni rapide di gap)",
    style="Card.TLabelframe",
)
gap_pressure_frame.pack(fill="both", expand=False, padx=5, pady=(0, 5))

pressure_columns = (
    "metric",
    "trend",
    "start",
    "end",
    "delta",
)

gap_pressure_tree = ttk.Treeview(
    gap_pressure_frame, columns=pressure_columns, show="headings", height=6
)
gap_pressure_tree.heading("metric", text="Metrica")
gap_pressure_tree.heading("trend", text="Trend")
gap_pressure_tree.heading("start", text="Inizio")
gap_pressure_tree.heading("end", text="Fine")
gap_pressure_tree.heading("delta", text="Δ gap (s)")

gap_pressure_tree.column("metric", width=120, anchor="w")
gap_pressure_tree.column("trend", width=120, anchor="center")
gap_pressure_tree.column("start", width=80, anchor="center")
gap_pressure_tree.column("end", width=80, anchor="center")
gap_pressure_tree.column("delta", width=100, anchor="center")

gap_pressure_tree.pack(fill="both", expand=True, padx=5, pady=4)

# --- Contenuto tab Race Control --- #
race_control_info_var = tk.StringVar(
    value=(
        "Messaggi Race Control: seleziona una sessione Race/Sprint, poi premi "
        "'Recupera messaggi Race Control' per visualizzare i dettagli."
    )
)
race_control_info_label = ttk.Label(
    race_control_tab_frame,
    textvariable=race_control_info_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
)
race_control_info_label.pack(fill="x", padx=5, pady=(5, 2))

race_control_buttons_frame = ttk.Frame(race_control_tab_frame, style="Card.TFrame")
race_control_buttons_frame.pack(fill="x", padx=5, pady=(0, 4))

race_control_fetch_button = ttk.Button(
    race_control_buttons_frame,
    text="Recupera messaggi Race Control",
    command=on_fetch_race_control_click,
)
race_control_fetch_button.pack(side="left")

race_control_frame = ttk.Frame(race_control_tab_frame)
race_control_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

race_control_columns = (
    "date_time",
    "lap",
    "category",
    "flag",
    "scope",
    "message",
)

race_control_tree = ttk.Treeview(
    race_control_frame,
    columns=race_control_columns,
    show="headings",
    height=12,
)

race_control_tree.heading("date_time", text="Data / Ora")
race_control_tree.heading("lap", text="Giro")
race_control_tree.heading("category", text="Categoria")
race_control_tree.heading("flag", text="Segnale")
race_control_tree.heading("scope", text="Ambito")
race_control_tree.heading("message", text="Messaggio")

race_control_tree.column("date_time", width=150, anchor="center")
race_control_tree.column("lap", width=70, anchor="center")
race_control_tree.column("category", width=120, anchor="center")
race_control_tree.column("flag", width=140, anchor="center")
race_control_tree.column("scope", width=100, anchor="center")
race_control_tree.column("message", width=800, anchor="w")

race_control_vsb = ttk.Scrollbar(
    race_control_frame, orient="vertical", command=race_control_tree.yview
)
race_control_tree.configure(yscrollcommand=race_control_vsb.set)

race_control_tree.grid(row=0, column=0, sticky="nsew")
race_control_vsb.grid(row=0, column=1, sticky="ns")

race_control_frame.rowconfigure(0, weight=1)
race_control_frame.columnconfigure(0, weight=1)

# --- Contenuto tab Gomme (stints_tab_frame) --- #
stints_mode_var = tk.StringVar(value="map")

stints_mode_frame = ttk.Frame(stints_tab_frame)
stints_mode_frame.pack(fill="x", pady=(4, 2), padx=5)

ttk.Label(stints_mode_frame, text="Vista:", font=("", 9, "bold")).pack(side="left")
rb_map = ttk.Radiobutton(
    stints_mode_frame,
    text="Mappa stint",
    variable=stints_mode_var,
    value="map",
    command=on_stints_mode_changed
)
rb_map.pack(side="left", padx=(5, 0))

rb_analysis = ttk.Radiobutton(
    stints_mode_frame,
    text="Analisi stint/compound",
    variable=stints_mode_var,
    value="analysis",
    command=on_stints_mode_changed
)
rb_analysis.pack(side="left", padx=(5, 0))

# Controlli analisi stint/compound
stints_controls_frame = ttk.Frame(stints_tab_frame)
stints_controls_frame.pack(fill="x", padx=5, pady=(0, 4))

ttk.Label(stints_controls_frame, text="Stint per grafico degrado:").pack(side="left")
stints_combo = ttk.Combobox(stints_controls_frame, width=35, state="readonly")
stints_combo.pack(side="left", padx=(5, 5))

btn_stint_degr = ttk.Button(
    stints_controls_frame,
    text="Mostra degrado stint selezionato",
    command=on_show_stint_degradation_click
)
btn_stint_degr.pack(side="left", padx=(5, 5))

btn_compound_avg = ttk.Button(
    stints_controls_frame,
    text="Mostra media per compound",
    command=on_show_compound_avg_click
)
btn_compound_avg.pack(side="left")

# Frame per canvas grafico gomme
stints_plot_canvas_frame = ttk.Frame(stints_tab_frame)
stints_plot_canvas_frame.pack(fill="both", expand=True, padx=5, pady=(2, 4))

# Tabella riassunto stint
ttk.Label(
    stints_tab_frame,
    text="Riassunto stint (giri non out-lap):",
    font=("", 9, "bold")
).pack(anchor="w", padx=5, pady=(2, 0))

stints_summary_frame = ttk.Frame(stints_tab_frame)
stints_summary_frame.pack(fill="x", expand=False, padx=5, pady=(0, 5))

stints_summary_columns = (
    "stint_num",
    "compound",
    "laps_range",
    "laps_used",
    "avg_lap",
    "best_lap",
    "delta_prev",
)

stints_summary_tree = ttk.Treeview(
    stints_summary_frame,
    columns=stints_summary_columns,
    show="headings",
    height=4
)

stints_summary_tree.heading("stint_num", text="Stint")
stints_summary_tree.heading("compound", text="Compound")
stints_summary_tree.heading("laps_range", text="Giri (start-end)")
stints_summary_tree.heading("laps_used", text="Giri usati")
stints_summary_tree.heading("avg_lap", text="Lap medio")
stints_summary_tree.heading("best_lap", text="Best lap")
stints_summary_tree.heading("delta_prev", text="Δ avg vs stint prec. (s)")

stints_summary_tree.column("stint_num", width=60, anchor="center")
stints_summary_tree.column("compound", width=90, anchor="center")
stints_summary_tree.column("laps_range", width=120, anchor="center")
stints_summary_tree.column("laps_used", width=80, anchor="center")
stints_summary_tree.column("avg_lap", width=110, anchor="center")
stints_summary_tree.column("best_lap", width=110, anchor="center")
stints_summary_tree.column("delta_prev", width=150, anchor="center")

stints_summary_vsb = ttk.Scrollbar(
    stints_summary_frame,
    orient="vertical",
    command=stints_summary_tree.yview
)
stints_summary_tree.configure(yscrollcommand=stints_summary_vsb.set)

stints_summary_tree.grid(row=0, column=0, sticky="nsew")
stints_summary_vsb.grid(row=0, column=1, sticky="ns")

stints_summary_frame.rowconfigure(0, weight=1)
stints_summary_frame.columnconfigure(0, weight=1)

# --- Contenuto tab Meteo --- #
weather_info_var = tk.StringVar(
    value="Meteo sessione: in attesa di una sessione selezionata."
)
weather_info_label = ttk.Label(weather_tab_frame, textvariable=weather_info_var, anchor="w")
weather_info_label.pack(fill="x", pady=(4, 2), padx=5)

weather_summary_frame = ttk.LabelFrame(weather_tab_frame, text="Riepilogo meteo sessione")
weather_summary_frame.pack(fill="x", padx=5, pady=(4, 2))

weather_stats_vars = {
    "track": tk.StringVar(value="Track temp --"),
    "air": tk.StringVar(value="Air temp --"),
    "humidity": tk.StringVar(value="Umidità media: --"),
    "wind": tk.StringVar(value="Wind speed media: --"),
}

ttk.Label(weather_summary_frame, textvariable=weather_stats_vars["track"], anchor="w").grid(
    row=0, column=0, sticky="w", padx=5, pady=2
)
ttk.Label(weather_summary_frame, textvariable=weather_stats_vars["air"], anchor="w").grid(
    row=0, column=1, sticky="w", padx=5, pady=2
)
ttk.Label(weather_summary_frame, textvariable=weather_stats_vars["humidity"], anchor="w").grid(
    row=1, column=0, sticky="w", padx=5, pady=2
)
ttk.Label(weather_summary_frame, textvariable=weather_stats_vars["wind"], anchor="w").grid(
    row=1, column=1, sticky="w", padx=5, pady=2
)
weather_summary_frame.columnconfigure(0, weight=1)
weather_summary_frame.columnconfigure(1, weight=1)

weather_charts_container = ttk.Frame(weather_tab_frame)
weather_charts_container.pack(fill="both", expand=True, padx=5, pady=(2, 5))

weather_evolution_section = ttk.LabelFrame(
    weather_charts_container,
    text="Evoluzione temperatura asfalto / aria",
)
weather_evolution_section.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)

weather_plot_frame = ttk.Frame(weather_evolution_section)
weather_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

weather_perf_section = ttk.LabelFrame(
    weather_charts_container,
    text="Meteo & prestazioni: correlazione track temp vs lap time",
)
weather_perf_section.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)

weather_perf_info_var = tk.StringVar(
    value="Carica una sessione e seleziona un pilota per vedere la correlazione track temp vs lap time."
)
ttk.Label(
    weather_perf_section,
    textvariable=weather_perf_info_var,
    anchor="w",
    wraplength=600,
).pack(fill="x", padx=5, pady=(4, 2))

weather_perf_plot_frame = ttk.Frame(weather_perf_section)
weather_perf_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

weather_charts_container.columnconfigure(0, weight=1)
weather_charts_container.columnconfigure(1, weight=1)
weather_charts_container.rowconfigure(0, weight=1)

# --- Contenuto tab Statistiche piloti --- #
stats_frame_inner = ttk.Frame(stats_tab_frame)
stats_frame_inner.pack(fill="both", expand=True, padx=5, pady=5)

stats_columns = (
    "position",
    "driver_number",
    "driver_name",
    "team_name",
    "laps_used",
    "avg_lap",
    "std_lap",
    "best_lap",
    "gap_best",
)

stats_tree = ttk.Treeview(
    stats_frame_inner,
    columns=stats_columns,
    show="headings",
    height=8
)

stats_tree.heading("position", text="Pos.")
stats_tree.heading("driver_number", text="N°")
stats_tree.heading("driver_name", text="Pilota")
stats_tree.heading("team_name", text="Team")
stats_tree.heading("laps_used", text="Giri usati")
stats_tree.heading("avg_lap", text="Lap medio")
stats_tree.heading("std_lap", text="σ Lap Time")
stats_tree.heading("best_lap", text="Best lap")
stats_tree.heading("gap_best", text="Gap dal best sessione (s)")

stats_tree.column("position", width=45, anchor="center")
stats_tree.column("driver_number", width=45, anchor="center")
stats_tree.column("driver_name", width=180, anchor="w")
stats_tree.column("team_name", width=180, anchor="w")
stats_tree.column("laps_used", width=80, anchor="center")
stats_tree.column("avg_lap", width=110, anchor="center")
stats_tree.column("std_lap", width=110, anchor="center")
stats_tree.column("best_lap", width=110, anchor="center")
stats_tree.column("gap_best", width=150, anchor="center")

stats_vsb = ttk.Scrollbar(
    stats_frame_inner,
    orient="vertical",
    command=stats_tree.yview
)
stats_tree.configure(yscrollcommand=stats_vsb.set)

stats_tree.grid(row=0, column=0, sticky="nsew")
stats_vsb.grid(row=0, column=1, sticky="ns")

stats_frame_inner.rowconfigure(0, weight=1)
stats_frame_inner.columnconfigure(0, weight=1)

# --- Contenuto tab Pit stop & strategia --- #
pit_stats_label = ttk.Label(
    pit_strategy_tab_frame,
    text="Riassunto pit stop per pilota (Race/Sprint):",
    font=("", 9, "bold")
)
pit_stats_label.pack(anchor="w", padx=5, pady=(4, 0))

pit_stats_frame = ttk.Frame(pit_strategy_tab_frame)
pit_stats_frame.pack(fill="x", expand=False, padx=5, pady=(0, 4))

pit_stats_columns = (
    "driver_name",
    "team_name",
    "num_pits",
    "avg_pit",
    "best_pit",
    "worst_pit",
    "pit_laps",
)

pit_stats_tree = ttk.Treeview(
    pit_stats_frame,
    columns=pit_stats_columns,
    show="headings",
    height=6
)

pit_stats_tree.heading("driver_name", text="Pilota")
pit_stats_tree.heading("team_name", text="Team")
pit_stats_tree.heading("num_pits", text="# Pit")
pit_stats_tree.heading("avg_pit", text="Pit medio (s)")
pit_stats_tree.heading("best_pit", text="Miglior pit (s)")
pit_stats_tree.heading("worst_pit", text="Peggior pit (s)")
pit_stats_tree.heading("pit_laps", text="Giri pit")

pit_stats_tree.column("driver_name", width=180, anchor="w")
pit_stats_tree.column("team_name", width=180, anchor="w")
pit_stats_tree.column("num_pits", width=60, anchor="center")
pit_stats_tree.column("avg_pit", width=100, anchor="center")
pit_stats_tree.column("best_pit", width=100, anchor="center")
pit_stats_tree.column("worst_pit", width=110, anchor="center")
pit_stats_tree.column("pit_laps", width=260, anchor="w")

pit_stats_vsb = ttk.Scrollbar(
    pit_stats_frame,
    orient="vertical",
    command=pit_stats_tree.yview
)
pit_stats_tree.configure(yscrollcommand=pit_stats_vsb.set)

pit_stats_tree.grid(row=0, column=0, sticky="nsew")
pit_stats_vsb.grid(row=0, column=1, sticky="ns")

pit_stats_frame.rowconfigure(0, weight=1)
pit_stats_frame.columnconfigure(0, weight=1)

ttk.Label(
    pit_strategy_tab_frame,
    text="Distribuzione pit stop per giro:",
    font=("", 9, "bold")
).pack(anchor="w", padx=5, pady=(0, 2))

pit_strategy_plot_frame = ttk.Frame(pit_strategy_tab_frame, style="Card.TFrame")
pit_strategy_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

# --- Label info click distacchi --- #
gap_point_info_var = tk.StringVar(
    value="Clicca un punto sul grafico distacchi per vedere gap_to_leader e interval."
)
gap_point_info_label = ttk.Label(
    plots_shell, textvariable=gap_point_info_var, anchor="w", style="Info.TLabel"
)
gap_point_info_label.pack(fill="x", padx=5, pady=(0, 4))

# Barra di stato
status_var = tk.StringVar(
    value="Inserisci un anno, premi 'Recupera calendario', poi seleziona una sessione."
)
status_label = ttk.Label(
    root, textvariable=status_var, anchor="w", padding=5, style="Status.TLabel"
)
status_label.pack(fill="x")

# Avvio GUI
root.mainloop()
