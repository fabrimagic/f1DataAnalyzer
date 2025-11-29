import csv
import os
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import urllib.request
import urllib.error
import urllib.parse
import json
import math
from datetime import datetime, timedelta
from shutil import which

import pandas as pd
import numpy as np

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
API_TEAM_RADIO_URL = "https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_number}"
API_OVERTAKES_URL = "https://api.openf1.org/v1/overtakes?session_key={session_key}"
API_CAR_DATA_URL = (
    "https://api.openf1.org/v1/car_data?session_key={session_key}&driver_number={driver_number}"
)

# Cache per le informazioni pilota
DRIVER_CACHE = {}
DRIVER_PROFILE_CACHE = {}

# Canvas matplotlib per i grafici (uno per tab)
gap_plot_canvas = None
stints_plot_canvas = None
gap_plot_frame = None
battle_pressure_tab_frame = None

# Dati e handler per il click sul grafico distacchi
gap_fig = None
gap_ax = None
gap_click_cid = None
gap_click_data = {"laps": [], "gap_leader": [], "gap_prev": []}
gap_point_info_var = None  # inizializzato dopo la creazione della GUI
gap_slipstream_var = None
gap_pressure_tree = None
battle_pressure_tree = None
battle_pressure_canvas = None
battle_pressure_fig = None
battle_pressure_plot_frame = None
battle_pressure_info_var = None
battle_pressure_cache = {}
race_control_tree = None
race_control_info_var = None
team_radio_tree = None
team_radio_info_var = None
team_radio_play_thread = None
team_radio_play_process = None
race_timeline_tree = None
race_timeline_detail_var = None
race_timeline_last_events = []
lift_coast_per_lap_tree = None
lift_coast_segments_tree = None
lift_coast_info_var = None
lift_coast_lap_listbox = None
lift_coast_selected_laps_var = None
lift_coast_selection_label_var = None
lift_coast_available_laps = []

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
weather_phase_tree = None
weather_phase_canvas = None
weather_phase_fig = None
weather_phase_plot_frame = None
weather_phase_info_var = None
compound_weather_tree = None
compound_weather_canvas = None
compound_weather_fig = None
compound_weather_plot_frame = None
compound_weather_info_var = None

# Statistiche piloti (lap time)
stats_tree = None

# Dati correnti per stint/laps del pilota selezionato
current_stints_data = []
current_laps_data = []
current_laps_session_key = None
current_laps_driver = None
current_results_session_key = None
current_pit_session_key = None

# Riferimenti GUI per sezione gomme
stints_mode_var = None
stints_summary_tree = None
stints_combo = None
stints_plot_canvas_frame = None
current_stints_for_combo = []

# Dati correnti per pit & risultati (per analisi strategia)
current_pit_data = []
current_results_data = []

# Cache per export e dati caricati
intervals_cache = {}
stints_cache = {}
laps_cache = {}
pit_cache = {}
overtakes_cache = {}
race_control_cache = {}
race_control_session_key = None
battle_pressure_last_results = []
battle_pressure_session_key = None
session_stats_last_results = []
session_stats_session_key = None
race_timeline_last_session_key = None
car_data_cache = {}
car_data_lap_cache = {}

# Degrado gomme (Practice)
tyre_wear_laps_tree = None
tyre_wear_info_var = None
tyre_wear_results_vars = {}
tyre_wear_level_label = None
tyre_wear_canvas = None
tyre_wear_fig = None
tyre_wear_plot_frame = None

# Pit stop & strategia
pit_stats_tree = None
pit_strategy_canvas = None
pit_strategy_fig = None
pit_strategy_plot_frame = None
pit_undercut_tree = None
pit_window_tree = None
pit_window_canvas = None
pit_window_fig = None
pit_window_plot_frame = None


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


def http_get_bytes(url: str):
    """GET semplice che ritorna il contenuto binario o solleva RuntimeError."""
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} per URL: {url}")
            return response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Errore di rete: {e}")


def fetch_list_from_api(url: str, error_context: str):
    """Esegue una GET e verifica che la risposta sia una lista."""
    data = http_get_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Formato JSON inatteso {error_context}.")
    return data


def _cache_driver_dataset(cache: dict, session_key: int, driver_number, data_list):
    """Salva i dati per (session_key, driver_number) in una cache dizionario."""
    if not isinstance(data_list, list):
        return
    try:
        skey = int(session_key)
        dnum = int(driver_number)
    except (ValueError, TypeError):
        return

    cache.setdefault(skey, {})[dnum] = data_list


def parse_driver_number(driver_number, error_context: str):
    """Converte il driver_number in intero o solleva RuntimeError con contesto."""
    try:
        return int(driver_number)
    except (ValueError, TypeError):
        raise RuntimeError(f"driver_number non valido per la chiamata {error_context}.")


def fetch_sessions(year: int):
    url = f"{API_SESSIONS_URL}{year}"
    return fetch_list_from_api(url, "per le sessioni")


def fetch_session_results(session_key: int):
    url = f"{API_RESULTS_URL}{session_key}"
    return fetch_list_from_api(url, "per i risultati di sessione")


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
    dn_key = parse_driver_number(driver_number, "intervals")

    url = API_INTERVALS_URL.format(session_key=session_key, driver_number=dn_key)
    data = fetch_list_from_api(url, "per i dati intervals")
    _cache_driver_dataset(intervals_cache, session_key, dn_key, data)
    return data


def fetch_stints(session_key: int, driver_number):
    """Stint gomme per pilota nella sessione."""
    dn_key = parse_driver_number(driver_number, "stints")

    url = API_STINTS_URL.format(session_key=session_key, driver_number=dn_key)
    data = fetch_list_from_api(url, "per i dati stints")
    _cache_driver_dataset(stints_cache, session_key, dn_key, data)
    return data


def fetch_laps(session_key: int, driver_number):
    """Dati dei giri per pilota nella sessione."""
    dn_key = parse_driver_number(driver_number, "laps")

    url = API_LAPS_URL.format(session_key=session_key, driver_number=dn_key)
    data = fetch_list_from_api(url, "per i dati laps")
    _cache_driver_dataset(laps_cache, session_key, dn_key, data)
    return data


def fetch_car_data(session_key: int, driver_number):
    """Telemetria car_data per pilota e sessione."""
    dn_key = parse_driver_number(driver_number, "car data")

    url = API_CAR_DATA_URL.format(session_key=session_key, driver_number=dn_key)
    data = fetch_list_from_api(url, "per i dati car_data")
    _cache_driver_dataset(car_data_cache, session_key, dn_key, data)
    return data


def fetch_car_data_for_lap(session_key: int, driver_number, lap_number, lap_start_dt, lap_end_dt=None):
    """Telemetria car_data filtrata per singolo giro usando intervalli temporali."""

    dn_key = parse_driver_number(driver_number, "car data")

    cache_session = car_data_lap_cache.setdefault(int(session_key), {})
    cache_driver = cache_session.setdefault(dn_key, {})
    if lap_number in cache_driver:
        return cache_driver[lap_number]

    if lap_start_dt is None:
        raise RuntimeError("Data di inizio giro non disponibile per la chiamata car_data.")

    start_iso = lap_start_dt.isoformat()
    if lap_start_dt.tzinfo is None:
        start_iso += "+00:00"

    params = [
        f"session_key={session_key}",
        f"driver_number={dn_key}",
        f"date>{urllib.parse.quote(start_iso)}",
    ]

    if lap_end_dt is not None:
        end_iso = lap_end_dt.isoformat()
        if lap_end_dt.tzinfo is None:
            end_iso += "+00:00"
        params.append(f"date<{urllib.parse.quote(end_iso)}")

    url = "https://api.openf1.org/v1/car_data?" + "&".join(params)
    data = fetch_list_from_api(url, f"per i dati car_data del giro {lap_number}")
    cache_driver[lap_number] = data if isinstance(data, list) else []
    return cache_driver[lap_number]


def fetch_pit_stops(session_key: int):
    """Elenco pit stop per la sessione (tutti i piloti)."""
    url = API_PIT_URL.format(session_key=session_key)
    data = fetch_list_from_api(url, "per i dati pit")
    pit_cache[session_key] = data if isinstance(data, list) else []
    return data


def fetch_weather(session_key: int):
    """Dati meteo per la singola sessione (session_key)."""
    url = API_WEATHER_URL.format(session_key=session_key)
    return fetch_list_from_api(url, "per i dati weather")


def fetch_race_control_messages(session_key: int):
    """Messaggi Race Control per l'intera sessione."""
    url = API_RACE_CONTROL_URL.format(session_key=session_key)
    data = fetch_list_from_api(url, "per i dati Race Control")
    race_control_cache[session_key] = data if isinstance(data, list) else []
    return data


def fetch_team_radio(session_key: int, driver_number):
    """Team radio per pilota e sessione selezionati."""
    dn_key = parse_driver_number(driver_number, "team radio")

    url = API_TEAM_RADIO_URL.format(session_key=session_key, driver_number=dn_key)
    return fetch_list_from_api(url, "per i team radio")


def fetch_overtakes(session_key: int):
    """Sorpassi rilevati automaticamente per la sessione."""
    url = API_OVERTAKES_URL.format(session_key=session_key)
    data = fetch_list_from_api(url, "per i dati sui sorpassi")
    overtakes_cache[session_key] = data if isinstance(data, list) else []
    return data


def set_team_radio_status(message: str):
    if team_radio_info_var is not None:
        root.after(0, lambda m=message: team_radio_info_var.set(m))


def stop_team_radio_playback():
    """Termina un'eventuale riproduzione audio in corso."""
    global team_radio_play_process
    if team_radio_play_process is not None:
        try:
            if team_radio_play_process.poll() is None:
                team_radio_play_process.terminate()
        except Exception:
            pass
    team_radio_play_process = None


def build_audio_player_command(audio_path: str):
    """Restituisce il comando per riprodurre l'audio usando un player disponibile."""
    candidate_players = [
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("mpv", ["--no-video", "--really-quiet"]),
        ("mpg123", []),
        ("cvlc", ["--play-and-exit", "--intf", "dummy"]),
        ("afplay", []),
    ]

    for exe, extra_args in candidate_players:
        full_path = which(exe)
        if full_path:
            return [full_path, *extra_args, audio_path]
    return None


def play_team_radio_from_url(url: str):
    """Scarica l'audio e lo riproduce senza aprire il browser."""
    global team_radio_play_thread, team_radio_play_process

    if not url:
        messagebox.showinfo("Info", "Nessun URL disponibile per questo team radio.")
        return

    stop_team_radio_playback()

    def worker():
        tmp_path = None
        try:
            set_team_radio_status("Download audio del team radio in corso...")
            audio_bytes = http_get_bytes(url)

            suffix = os.path.splitext(url)[1] or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            cmd = build_audio_player_command(tmp_path)
            if cmd is None:
                raise RuntimeError(
                    "Nessun lettore audio trovato (prova ad installare ffplay, mpv, mpg123 o vlc)."
                )

            set_team_radio_status("Riproduzione team radio in corso...")
            team_radio_play_process = subprocess.Popen(cmd)
            team_radio_play_process.wait()
            set_team_radio_status("Riproduzione team radio completata.")
        except Exception as e:
            set_team_radio_status("Errore nella riproduzione del team radio.")
            root.after(0, lambda: messagebox.showerror("Riproduzione team radio", str(e)))
        finally:
            stop_team_radio_playback()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    team_radio_play_thread = threading.Thread(target=worker, daemon=True)
    team_radio_play_thread.start()


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


def build_lap_intervals(laps_data, selected_laps):
    """Restituisce intervalli (start/end) per i giri selezionati, ordinati per data inizio."""

    if not laps_data:
        return []

    selected_set = set()
    for lap in selected_laps or []:
        try:
            selected_set.add(int(lap))
        except (TypeError, ValueError):
            continue

    if not selected_set:
        return []

    lap_intervals = []
    for lap in laps_data:
        lap_start = parse_iso_datetime(lap.get("date_start", ""))
        lap_number = lap.get("lap_number")

        if lap_start is None or lap_number is None:
            continue

        try:
            lap_number_int = int(lap_number)
        except (TypeError, ValueError):
            continue

        if lap_number_int not in selected_set:
            continue

        duration_raw = lap.get("lap_duration")
        duration = None
        if isinstance(duration_raw, (int, float)):
            duration = float(duration_raw)
        else:
            try:
                duration = float(duration_raw)
            except (TypeError, ValueError):
                duration = None

        lap_intervals.append(
            {
                "lap_number": lap_number_int,
                "start": lap_start,
                "duration": duration,
                "end": None,
            }
        )

    lap_intervals.sort(key=lambda x: (x.get("start") or datetime.max, x.get("lap_number")))

    if not lap_intervals:
        return []

    for idx, lap in enumerate(lap_intervals):
        next_start = lap_intervals[idx + 1]["start"] if idx + 1 < len(lap_intervals) else None
        duration = lap.get("duration")
        candidate_end = None
        if duration is not None:
            candidate_end = lap["start"] + timedelta(seconds=duration)

        if next_start and candidate_end:
            lap["end"] = min(candidate_end, next_start)
        elif next_start:
            lap["end"] = next_start
        else:
            lap["end"] = candidate_end

        if lap.get("end") is None and lap.get("start") is not None:
            # Fallback prudente per l'ultimo giro senza durata nota
            lap["end"] = lap["start"] + timedelta(seconds=180)

    return lap_intervals


def compute_lift_and_coast(lap_intervals, laps_car_data):
    """Calcola i segmenti di lift & coast per giro limitando ai dati per-lap."""

    per_lap = {}
    summary = {
        "total_lnc_sec": 0.0,
        "avg_lnc_pct": 0.0,
        "laps_with_lnc": 0,
        "laps_without_lnc": 0,
    }

    if not lap_intervals or not laps_car_data:
        return {"per_lap": per_lap, "summary": summary}

    lap_lookup = {lap["lap_number"]: lap for lap in lap_intervals}

    def ensure_lap_entry(lap_info):
        lap_number = lap_info["lap_number"]
        if lap_number not in per_lap:
            per_lap[lap_number] = {
                "total_lnc_sec": 0.0,
                "lnc_pct": 0.0,
                "segments": [],
                "lap_duration": lap_info.get("duration"),
                "lap_start": lap_info.get("start"),
                "lap_end": lap_info.get("end"),
                "first_sample": None,
                "last_sample": None,
            }
        return per_lap[lap_number]

    def normalize_pedal(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for lap_info in lap_intervals:
        lap_number = lap_info.get("lap_number")
        lap_data = laps_car_data.get(lap_number, [])

        if not lap_data:
            continue

        car_samples = []
        for item in lap_data:
            sample_dt = parse_iso_datetime(item.get("date", ""))
            if sample_dt is None:
                continue
            car_samples.append(
                {
                    "date": sample_dt,
                    "throttle": normalize_pedal(item.get("throttle")),
                    "brake": normalize_pedal(item.get("brake")),
                }
            )

        car_samples.sort(key=lambda s: s["date"])

        if not car_samples:
            continue

        lap_entry = ensure_lap_entry(lap_info)

        current_segment_start = None

        def close_segment(end_ts):
            nonlocal current_segment_start
            if current_segment_start is None:
                return

            boundary = lap_info.get("end")
            if boundary is None and lap_entry.get("last_sample"):
                boundary = lap_entry["last_sample"]
            if end_ts is None:
                end_ts = boundary
            if end_ts is None:
                current_segment_start = None
                return

            duration_sec = (end_ts - current_segment_start).total_seconds()
            if duration_sec <= 0:
                current_segment_start = None
                return

            start_offset = None
            end_offset = None
            if lap_info.get("start"):
                start_offset = (current_segment_start - lap_info["start"]).total_seconds()
                end_offset = (end_ts - lap_info["start"]).total_seconds()

            segment = {
                "lap_number": lap_number,
                "start_time": current_segment_start,
                "end_time": end_ts,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "duration_sec": round(duration_sec, 3),
            }

            lap_entry["segments"].append(segment)
            lap_entry["total_lnc_sec"] += round(duration_sec, 3)
            current_segment_start = None

        for sample in car_samples:
            sample_dt = sample["date"]
            throttle = sample.get("throttle")
            brake = sample.get("brake")

            if lap_entry["first_sample"] is None:
                lap_entry["first_sample"] = sample_dt
            lap_entry["last_sample"] = sample_dt

            if throttle is None and brake is None:
                continue

            if throttle is None:
                throttle = 0.0
            if brake is None:
                brake = 0.0

            in_lnc = throttle == 0 and brake == 0

            if in_lnc:
                if current_segment_start is None:
                    current_segment_start = sample_dt
            elif current_segment_start is not None:
                close_segment(sample_dt)

        if current_segment_start is not None:
            close_segment(lap_info.get("end") or car_samples[-1]["date"])

    total_pct_values = []
    total_lnc = 0.0
    for lap_number, lap_entry in per_lap.items():
        lap_duration = lap_entry.get("lap_duration")
        if lap_duration is None and lap_entry.get("first_sample") and lap_entry.get("last_sample"):
            lap_duration = (
                lap_entry["last_sample"] - lap_entry["first_sample"]
            ).total_seconds()

        if isinstance(lap_duration, (int, float)) and lap_duration > 0:
            lap_entry["lap_duration"] = float(lap_duration)
            lap_entry["lnc_pct"] = round(
                (lap_entry["total_lnc_sec"] / lap_entry["lap_duration"]) * 100, 2
            )
            total_pct_values.append(lap_entry["lnc_pct"])
        else:
            lap_entry["lnc_pct"] = 0.0

        total_lnc += lap_entry.get("total_lnc_sec", 0.0)

    summary["total_lnc_sec"] = round(total_lnc, 3)
    if total_pct_values:
        summary["avg_lnc_pct"] = round(sum(total_pct_values) / len(total_pct_values), 2)
    summary["laps_with_lnc"] = len([lp for lp in per_lap.values() if lp.get("segments")])
    summary["laps_without_lnc"] = max(0, len(per_lap) - summary["laps_with_lnc"])

    return {"per_lap": per_lap, "summary": summary}


def update_lift_coast_lap_selection(laps_data):
    """Popola la lista dei giri selezionabili per l'analisi Lift & Coast."""

    global lift_coast_lap_listbox, lift_coast_selected_laps_var, lift_coast_available_laps

    lift_coast_available_laps = []

    if lift_coast_lap_listbox is None:
        return

    lift_coast_lap_listbox.delete(0, tk.END)

    if not isinstance(laps_data, list):
        if lift_coast_selected_laps_var is not None:
            lift_coast_selected_laps_var.set("Giri selezionati: 0 / 5")
        return

    laps_entries = []
    for lap in laps_data:
        lap_number = lap.get("lap_number")
        lap_duration = lap.get("lap_duration")
        try:
            lap_number_int = int(lap_number)
        except (TypeError, ValueError):
            continue

        lap_time_txt = format_time_from_seconds(lap_duration) or "--"
        laps_entries.append((lap_number_int, lap_time_txt))

    laps_entries.sort(key=lambda x: x[0])
    lift_coast_available_laps = [lap for lap, _ in laps_entries]

    for lap_number_int, lap_time_txt in laps_entries:
        lift_coast_lap_listbox.insert(tk.END, f"Giro {lap_number_int} - {lap_time_txt}")

    if lift_coast_selected_laps_var is not None:
        lift_coast_selected_laps_var.set("Giri selezionati: 0 / 5")


def get_selected_lift_coast_laps():
    if lift_coast_lap_listbox is None:
        return []

    indices = lift_coast_lap_listbox.curselection()
    selected = []
    for idx in indices:
        if 0 <= idx < len(lift_coast_available_laps):
            selected.append(lift_coast_available_laps[idx])
    if lift_coast_selected_laps_var is not None:
        lift_coast_selected_laps_var.set(f"Giri selezionati: {len(selected)} / 5")
    return selected


def on_lift_coast_selection_change(event):
    widget = event.widget
    if widget is None:
        return

    selected_indices = list(widget.curselection())
    if len(selected_indices) > 5:
        for idx in selected_indices[5:]:
            widget.selection_clear(idx)
        messagebox.showerror("Errore", "Puoi analizzare al massimo 5 giri per volta.")

    get_selected_lift_coast_laps()


def is_practice_session(session_name: str = "", session_type: str = "") -> bool:
    """Riconosce le sessioni Practice (FP1/FP2/FP3)."""
    name = (session_name or "").strip().lower()
    stype = (session_type or "").strip().lower()
    practice_tokens = ("practice", "fp1", "fp2", "fp3")
    return any(tok in name for tok in practice_tokens) or stype in practice_tokens or any(
        stype.startswith(tok) for tok in practice_tokens
    )


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


def get_selected_session_name():
    selection = sessions_tree.selection()
    if not selection:
        return None

    item_id = selection[0]
    values = sessions_tree.item(item_id, "values")
    if not values or len(values) < 3:
        return None
    return values[2]


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
    global current_stints_data, current_laps_data, current_laps_session_key, current_stints_for_combo, current_laps_driver

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
    current_laps_driver = None
    current_stints_for_combo = []
    if stints_combo is not None:
        stints_combo["values"] = ()
        stints_combo.set("")

    clear_stints_summary_table()
    clear_race_control_table()
    clear_team_radio_table()
    clear_rain_impact_outputs()
    clear_compound_weather_outputs()
    clear_tyre_wear_view()
    clear_lift_and_coast_view()


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
    clear_rain_impact_outputs()
    clear_compound_weather_outputs()


def clear_rain_impact_outputs():
    """Ripulisce tabella e grafico dell'analisi pioggia."""
    global weather_phase_tree, weather_phase_canvas, weather_phase_fig, weather_phase_info_var
    if weather_phase_tree is not None:
        for item in weather_phase_tree.get_children():
            weather_phase_tree.delete(item)
    if weather_phase_canvas is not None:
        weather_phase_canvas.get_tk_widget().destroy()
        weather_phase_canvas = None
    weather_phase_fig = None
    if weather_phase_info_var is not None:
        weather_phase_info_var.set(
            "Analizza l'impatto della pioggia sul passo gara dopo aver caricato meteo e giri del pilota."
        )


def clear_compound_weather_outputs():
    """Ripulisce tabella e grafico dell'analisi compound vs meteo."""
    global compound_weather_tree, compound_weather_canvas, compound_weather_fig, compound_weather_info_var
    if compound_weather_tree is not None:
        for item in compound_weather_tree.get_children():
            compound_weather_tree.delete(item)
    if compound_weather_canvas is not None:
        compound_weather_canvas.get_tk_widget().destroy()
        compound_weather_canvas = None
    compound_weather_fig = None
    if compound_weather_info_var is not None:
        compound_weather_info_var.set(
            "Analizza i compound rispetto alla temperatura pista dopo aver caricato giri e stint."
        )


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


def clear_team_radio_table():
    """Svuota la tabella Team Radio e ripristina il messaggio di guida."""
    global team_radio_tree, team_radio_info_var
    stop_team_radio_playback()
    if team_radio_tree is not None:
        for item in team_radio_tree.get_children():
            team_radio_tree.delete(item)

    if team_radio_info_var is not None:
        team_radio_info_var.set(
            "Team radio: seleziona una sessione e un pilota dai risultati, poi premi "
            "'Team radio pilota' per caricare i messaggi disponibili."
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


def clear_battle_pressure_table():
    """Svuota tabella e grafico del Battle / Pressure Index."""
    global battle_pressure_tree, battle_pressure_canvas, battle_pressure_fig
    global battle_pressure_info_var, battle_pressure_cache

    battle_pressure_cache = {}

    if battle_pressure_tree is not None:
        for item in battle_pressure_tree.get_children():
            battle_pressure_tree.delete(item)

    if battle_pressure_canvas is not None:
        try:
            battle_pressure_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        battle_pressure_canvas = None
    battle_pressure_fig = None

    if battle_pressure_info_var is not None:
        battle_pressure_info_var.set(
            "Calcola il Battle/Pressure Index per vedere un riepilogo dei duelli dei piloti."
        )


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


def update_battle_pressure_plot(driver_number):
    """Aggiorna il mini grafico del Battle/Pressure Index per il pilota selezionato."""
    global battle_pressure_canvas, battle_pressure_fig, battle_pressure_plot_frame
    global battle_pressure_cache, battle_pressure_info_var

    if battle_pressure_plot_frame is None:
        return

    if battle_pressure_canvas is not None:
        try:
            battle_pressure_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        battle_pressure_canvas = None
    battle_pressure_fig = None

    try:
        dnum_int = int(driver_number)
    except (ValueError, TypeError):
        dnum_int = None

    if dnum_int is None or dnum_int not in battle_pressure_cache:
        if battle_pressure_info_var is not None:
            battle_pressure_info_var.set(
                "Seleziona un pilota dalla tabella per vedere il dettaglio dei giri in attacco/difesa."
            )
        return

    detail = battle_pressure_cache.get(dnum_int, {})
    laps = detail.get("laps", [])
    intervals = detail.get("intervals", [])

    if not laps or not any(isinstance(v, (int, float)) for v in intervals):
        if battle_pressure_info_var is not None:
            battle_pressure_info_var.set("Nessun dato interval valido per questo pilota.")
        return

    attack_laps = set(detail.get("attack_laps", []))
    clean_laps = set(detail.get("clean_laps", []))
    segments = detail.get("pressure_segments", [])

    fig = Figure(figsize=(6, 2.6))
    ax = fig.add_subplot(111)

    plot_x = []
    plot_y = []
    colors = []
    labels_used = set()

    for lap, val in zip(laps, intervals):
        if not isinstance(val, (int, float)):
            continue
        plot_x.append(lap)
        plot_y.append(val)
        if lap in attack_laps:
            colors.append("tab:red")
        elif lap in clean_laps:
            colors.append("tab:blue")
        else:
            colors.append("#9ca3af")

    ax.scatter(plot_x, plot_y, c=colors, s=26, label="Interval")

    marker_styles = {
        ("interval", "Avvicinamento"): {"color": "tab:green", "marker": "^"},
        ("interval", "Allontanamento"): {"color": "tab:orange", "marker": "v"},
        ("gap_to_leader", "Avvicinamento"): {"color": "tab:purple", "marker": "^"},
        ("gap_to_leader", "Allontanamento"): {"color": "tab:blue", "marker": "v"},
    }

    for seg in segments:
        trend = seg.get("trend")
        metric = seg.get("metric")
        style = marker_styles.get((metric, trend), {})
        marker_x = seg.get("marker_x")
        marker_y = seg.get("marker_y")
        if marker_x is None or marker_y is None:
            continue
        label_key = f"{trend} ({metric})"
        label = None if label_key in labels_used else label_key
        labels_used.add(label_key)
        ax.scatter(
            marker_x,
            marker_y,
            color=style.get("color", "black"),
            marker=style.get("marker", "o"),
            s=70,
            zorder=3,
            label=label,
        )

    ax.set_xlabel("Giro")
    ax.set_ylabel("Interval (s)")
    ax.set_title("Giri in attacco (rosso) vs aria pulita (blu)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="upper right")

    battle_pressure_canvas = FigureCanvasTkAgg(fig, master=battle_pressure_plot_frame)
    battle_pressure_canvas.get_tk_widget().pack(fill="both", expand=True)
    battle_pressure_canvas.draw()

    battle_pressure_fig = fig

    if battle_pressure_info_var is not None:
        battle_pressure_info_var.set(
            "Rosso = giri <1s dal pilota davanti, Blu = aria pulita, marker = tratti di pressione."
        )


def update_battle_pressure_table(results):
    """Popola la tabella Battle/Pressure Index con i valori per pilota."""
    global battle_pressure_tree, battle_pressure_cache

    if battle_pressure_tree is None:
        return

    for item in battle_pressure_tree.get_children():
        battle_pressure_tree.delete(item)

    battle_pressure_cache = {}

    try:
        sorted_results = sorted(results, key=lambda r: r.get("attack_pct", 0), reverse=True)
    except Exception:
        sorted_results = results

    for res in sorted_results:
        driver_number = res.get("driver_number")
        driver_name = res.get("driver_name") or f"Driver {driver_number}"
        team_name = res.get("team_name", "")

        laps_total = res.get("laps_total", 0) or 0
        attack_pct = res.get("attack_pct", 0.0)
        clean_pct = res.get("clean_pct", 0.0)
        pressure_stints = res.get("pressure_stints", 0)
        pressure_delta = res.get("pressure_delta", 0.0)
        defense_segments = res.get("defense_segments", 0)
        overtakes_suffered = res.get("overtakes_suffered", 0)
        overtakes_made = res.get("overtakes_made", 0)

        battle_pressure_tree.insert(
            "",
            "end",
            values=(
                driver_name,
                team_name,
                f"{res.get('attack_laps', 0)}/{laps_total} ({attack_pct:.1f}%)",
                f"{res.get('clean_laps', 0)}/{laps_total} ({clean_pct:.1f}%)",
                pressure_stints,
                f"{pressure_delta:+.2f}",
                defense_segments,
                overtakes_suffered,
                overtakes_made,
                driver_number,
            ),
        )

        try:
            dnum_int = int(driver_number)
        except (ValueError, TypeError):
            continue

        battle_pressure_cache[dnum_int] = res.get("detail", {})


def on_battle_pressure_select(event=None):
    """Handler per la selezione di una riga nella tabella Battle/Pressure."""
    global battle_pressure_tree
    if battle_pressure_tree is None:
        return

    selection = battle_pressure_tree.selection()
    if not selection:
        return

    values = battle_pressure_tree.item(selection[0], "values")
    if not values or len(values) < 10:
        return

    driver_number = values[9]
    update_battle_pressure_plot(driver_number)



def update_race_control_messages(session_key: int):
    """Scarica e popola i messaggi Race Control per l'intera sessione."""
    global race_control_tree, race_control_info_var, race_control_session_key

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

    race_control_cache[session_key] = sorted_msgs if isinstance(sorted_msgs, list) else []
    race_control_session_key = session_key

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


def update_team_radio_table(session_key: int, driver_number, driver_name: str):
    """Scarica e mostra i team radio per il pilota selezionato."""
    global team_radio_tree, team_radio_info_var

    clear_team_radio_table()

    if team_radio_tree is None:
        return

    if team_radio_info_var is not None:
        team_radio_info_var.set(
            f"Scarico team radio per sessione {session_key}, pilota {driver_number}..."
        )

    try:
        messages = fetch_team_radio(session_key, driver_number)
    except RuntimeError as e:
        if team_radio_info_var is not None:
            team_radio_info_var.set("Errore nel recupero dei team radio.")
        messagebox.showerror("Errore Team Radio", str(e))
        return

    if not messages:
        if team_radio_info_var is not None:
            team_radio_info_var.set(
                f"Nessun team radio disponibile per il pilota {driver_number} in questa sessione."
            )
        return

    try:
        sorted_msgs = sorted(
            messages,
            key=lambda m: m.get("date", ""),
        )
    except Exception:
        sorted_msgs = messages

    for msg in sorted_msgs:
        raw_date = msg.get("date", "")
        parsed = parse_iso_datetime(raw_date)
        date_str = parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else str(raw_date)

        lap_number = msg.get("lap_number", "")
        lap_str = lap_number if isinstance(lap_number, int) else ""
        recording_url = msg.get("recording_url", "")

        team_radio_tree.insert(
            "",
            "end",
            values=(date_str, lap_str, recording_url),
        )

    if team_radio_info_var is not None:
        driver_label = driver_name if driver_name else f"Driver {driver_number}"
        team_radio_info_var.set(
            f"{len(sorted_msgs)} team radio per {driver_label}. Seleziona una riga e premi 'Riproduci' oppure fai doppio click."
        )


def on_play_team_radio_click(event=None):
    """Riproduce l'audio del team radio selezionato direttamente dall'interfaccia."""
    global team_radio_tree

    if team_radio_tree is None:
        return

    selection = team_radio_tree.selection()
    if not selection:
        messagebox.showinfo("Info", "Seleziona prima un team radio da riprodurre.")
        return

    item_id = selection[0]
    values = team_radio_tree.item(item_id, "values")
    if not values or len(values) < 3:
        messagebox.showinfo("Info", "Nessun URL disponibile per questo team radio.")
        return

    url = values[2]
    if not url:
        messagebox.showinfo("Info", "Nessun URL disponibile per questo team radio.")
        return

    play_team_radio_from_url(url)


def on_fetch_team_radio_click():
    """Handler per caricare i team radio del pilota selezionato."""
    session_key, _, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    update_team_radio_table(session_key, driver_number, driver_name)


# --------------------- Analisi Lift & Coast --------------------- #


def clear_lift_and_coast_view():
    global lift_coast_per_lap_tree, lift_coast_segments_tree, lift_coast_info_var
    global lift_coast_lap_listbox, lift_coast_selected_laps_var, lift_coast_available_laps
    global lift_coast_selection_label_var

    if lift_coast_per_lap_tree is not None:
        for item in lift_coast_per_lap_tree.get_children():
            lift_coast_per_lap_tree.delete(item)

    if lift_coast_segments_tree is not None:
        for item in lift_coast_segments_tree.get_children():
            lift_coast_segments_tree.delete(item)

    if lift_coast_info_var is not None:
        lift_coast_info_var.set(
            "Calcola il Lift & Coast per il pilota selezionato in una sessione di gara."
        )

    if lift_coast_selected_laps_var is not None:
        lift_coast_selected_laps_var.set("Giri selezionati: 0 / 5")

    if lift_coast_selection_label_var is not None:
        lift_coast_selection_label_var.set(
            "Carica i giri del pilota selezionato. Puoi scegliere manualmente fino a 5 giri da analizzare; se non selezioni nulla, verranno analizzati tutti i giri."
        )

    lift_coast_available_laps = []

    if lift_coast_lap_listbox is not None:
        lift_coast_lap_listbox.delete(0, tk.END)


def populate_lift_and_coast_tables(analysis_results, driver_name, selected_laps=None):
    if lift_coast_per_lap_tree is None or lift_coast_segments_tree is None:
        return

    clear_lift_and_coast_view()

    per_lap = analysis_results.get("per_lap", {}) if isinstance(analysis_results, dict) else {}
    summary = analysis_results.get("summary", {}) if isinstance(analysis_results, dict) else {}

    def format_time_value(timestamp, offset):
        if offset is not None:
            return f"{offset:.3f}s"
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%H:%M:%S.%f")[:-3]
        return "--"

    def lap_sort_key(item):
        try:
            return int(item[0])
        except (TypeError, ValueError):
            return item[0]

    for lap_number, info in sorted(per_lap.items(), key=lap_sort_key):
        total_lnc = info.get("total_lnc_sec", 0.0) or 0.0
        pct = info.get("lnc_pct", 0.0) or 0.0

        lift_coast_per_lap_tree.insert(
            "",
            "end",
            values=(lap_number, f"{total_lnc:.3f}", f"{pct:.2f}"),
        )

        segments = info.get("segments", []) or []
        for idx, seg in enumerate(segments, start=1):
            start_txt = format_time_value(seg.get("start_time"), seg.get("start_offset"))
            end_txt = format_time_value(seg.get("end_time"), seg.get("end_offset"))
            duration_txt = f"{seg.get('duration_sec', 0):.3f}"
            lift_coast_segments_tree.insert(
                "",
                "end",
                values=(lap_number, idx, start_txt, end_txt, duration_txt),
            )

    total_lnc = summary.get("total_lnc_sec", 0.0) or 0.0
    avg_pct = summary.get("avg_lnc_pct", 0.0) or 0.0
    laps_with = summary.get("laps_with_lnc", 0) or 0
    laps_without = summary.get("laps_without_lnc", 0) or 0

    if lift_coast_info_var is not None:
        driver_label = driver_name if driver_name else "pilota selezionato"
        selected_list = selected_laps or []
        if selected_list:
            laps_txt = f"Giri analizzati: {', '.join(map(str, selected_list))}"
        else:
            laps_txt = "Tutti i giri disponibili analizzati"
        lift_coast_info_var.set(
            " | ".join(
                [
                    laps_txt,
                    f"Totale L&C {driver_label}: {total_lnc:.3f}s",
                    f"Media per giro: {avg_pct:.2f}%",
                    f"Giri con L&C: {laps_with}",
                    f"Giri senza L&C: {laps_without}",
                ]
            )
        )


def on_refresh_lift_coast_laps_click():
    global current_laps_data, current_laps_driver, current_laps_session_key

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "Il Lift & Coast è disponibile solo per sessioni Race o Sprint.",
        )
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    if (
        current_laps_session_key != session_key
        or current_laps_driver != driver_number
        or not current_laps_data
    ):
        try:
            current_laps_data = fetch_laps(session_key, driver_number)
            current_laps_session_key = session_key
            current_laps_driver = driver_number
        except RuntimeError as e:
            messagebox.showerror("Errore", str(e))
            return

    update_lift_coast_lap_selection(current_laps_data)
    if lift_coast_selection_label_var is not None:
        lift_coast_selection_label_var.set(
            f"Giri disponibili per {driver_name}: seleziona manualmente fino a 5 giri oppure lascia vuoto per analizzare tutti i giri."
        )
    if lift_coast_info_var is not None:
        lift_coast_info_var.set(
            "Seleziona manualmente fino a 5 giri e premi 'Lift & Coast pilota' per l'analisi. Se non selezioni nessun giro, verranno analizzati tutti i giri del pilota."
        )


def on_compute_lift_and_coast_click():
    global current_laps_data, current_laps_driver, current_laps_session_key

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "Il Lift & Coast è disponibile solo per sessioni Race o Sprint.",
        )
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    if (
        current_laps_session_key != session_key
        or current_laps_driver != driver_number
        or not current_laps_data
    ):
        try:
            current_laps_data = fetch_laps(session_key, driver_number)
            current_laps_session_key = session_key
            current_laps_driver = driver_number
            update_lift_coast_lap_selection(current_laps_data)
        except RuntimeError as e:
            messagebox.showerror("Errore", str(e))
            return
    elif lift_coast_available_laps and lift_coast_selected_laps_var is not None:
        selected_preview = get_selected_lift_coast_laps()
        lift_coast_selected_laps_var.set(
            f"Giri selezionati: {len(selected_preview)} / 5"
        )

    selected_laps_ui = get_selected_lift_coast_laps()

    if selected_laps_ui and len(selected_laps_ui) > 5:
        messagebox.showerror("Errore", "Puoi analizzare al massimo 5 giri per volta.")
        return

    if selected_laps_ui:
        laps_to_analyze = selected_laps_ui
    else:
        all_laps = []
        for lap in current_laps_data:
            num = lap.get("lap_number")
            try:
                num_int = int(num)
            except (TypeError, ValueError):
                continue
            all_laps.append(num_int)

        laps_to_analyze = sorted(set(all_laps))

    lap_intervals = build_lap_intervals(current_laps_data, laps_to_analyze)
    if not lap_intervals:
        messagebox.showerror(
            "Errore",
            "Impossibile determinare gli intervalli temporali dei giri selezionati.",
        )
        return

    laps_car_data = {}
    for lap in lap_intervals:
        lap_num = lap.get("lap_number")
        if lap.get("start") is None:
            messagebox.showerror(
                "Errore",
                f"Data/ora di inizio non disponibile per il giro {lap_num}. Giro saltato.",
            )
            continue

        try:
            lap_data = fetch_car_data_for_lap(
                session_key,
                driver_number,
                lap_num,
                lap.get("start"),
                lap.get("end"),
            )
            laps_car_data[lap_num] = lap_data
        except RuntimeError as e:
            err_msg = str(e)
            if "HTTP 422" in err_msg:
                messagebox.showerror(
                    "Errore", f"Errore nel recupero telemetria per il giro {lap_num} (HTTP 422). Questo giro non verrà incluso nell'analisi."
                )
            else:
                messagebox.showerror(
                    "Errore",
                    f"Errore nel recupero telemetria per il giro {lap_num}: {err_msg}\nIl giro non verrà incluso nell'analisi.",
                )
            continue

    if not laps_car_data:
        messagebox.showerror(
            "Errore",
            "Non sono stati recuperati dati telemetrici per i giri selezionati.",
        )
        return

    status_var.set("Calcolo Lift & Coast in corso...")
    root.update_idletasks()

    analysis = compute_lift_and_coast(lap_intervals, laps_car_data)
    populate_lift_and_coast_tables(analysis, driver_name, laps_to_analyze)
    plots_notebook.select(lift_coast_tab)

    status_var.set(
        f"Lift & Coast calcolato per il pilota {driver_name or driver_number} in questa sessione."
    )


def clear_race_timeline_table():
    """Svuota la timeline di gara."""
    global race_timeline_tree, race_timeline_detail_var, race_timeline_last_events, race_timeline_last_session_key
    race_timeline_last_events = []
    race_timeline_last_session_key = None
    if race_timeline_tree is not None:
        for item in race_timeline_tree.get_children():
            race_timeline_tree.delete(item)
    if race_timeline_detail_var is not None:
        race_timeline_detail_var.set(
            "Timeline pronta: premi 'Genera Timeline Gara' per creare una sequenza di eventi."
        )


def on_export_race_timeline_click():
    """Esporta la timeline di gara in un file di testo."""
    global race_timeline_last_events

    if not race_timeline_last_events:
        messagebox.showinfo(
            "Export Timeline", "Genera prima la timeline di gara da esportare."
        )
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("File di testo", "*.txt"), ("Tutti i file", "*.*")],
        title="Esporta Race Timeline",
    )

    if not filepath:
        return

    try:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["Race Timeline"])
            writer.writerow(["Timestamp", "Giro", "Tipo", "Descrizione", "Pilota/i"])

            for ev in race_timeline_last_events:
                ts = ev.get("timestamp")
                ts_str = (
                    ts.strftime("%Y-%m-%d %H:%M:%S")
                    if isinstance(ts, datetime)
                    else (str(ts) if ts else "")
                )
                lap_val = ev.get("lap")
                lap_str = str(lap_val) if isinstance(lap_val, int) else ""

                description = ev.get("description", "") or ""
                drivers = ev.get("drivers", "") or ""
                writer.writerow([ts_str, lap_str, ev.get("type", ""), description, drivers])

        messagebox.showinfo(
            "Export Timeline",
            f"Timeline esportata in:\n{filepath}",
        )
    except OSError as e:
        messagebox.showerror("Export Timeline", f"Impossibile salvare il file: {e}")


def update_race_timeline_table(events):
    """Popola la tabella timeline con gli eventi ordinati."""
    global race_timeline_tree, race_timeline_detail_var, race_timeline_last_events, race_timeline_last_session_key
    if race_timeline_tree is None:
        return

    race_timeline_last_events = events or []
    session_key, _, _ = get_selected_session_info()
    race_timeline_last_session_key = session_key

    for item in race_timeline_tree.get_children():
        race_timeline_tree.delete(item)

    for ev in events:
        ts = ev.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "--"
        lap_val = ev.get("lap")
        lap_str = lap_val if isinstance(lap_val, int) else ""
        race_timeline_tree.insert(
            "",
            "end",
            values=(ts_str, lap_str, ev.get("type", ""), ev.get("description", ""), ev.get("drivers", "")),
        )

    if race_timeline_detail_var is not None:
        race_timeline_detail_var.set(
            f"Timeline generata con {len(events)} eventi." if events else "Nessun evento disponibile per la timeline."
        )


def on_race_timeline_select(event=None):
    """Mostra dettagli dell'evento selezionato nella timeline."""
    global race_timeline_tree, race_timeline_detail_var
    if race_timeline_tree is None or race_timeline_detail_var is None:
        return

    selection = race_timeline_tree.selection()
    if not selection:
        race_timeline_detail_var.set(
            "Seleziona un evento della timeline per vedere i dettagli."
        )
        return

    values = race_timeline_tree.item(selection[0], "values")
    if not values or len(values) < 5:
        return

    ts_str, lap_str, ev_type, description, drivers = values
    details_lines = [f"[{ev_type}] {description}"]
    if lap_str:
        details_lines.append(f"Giro: {lap_str}")
    if ts_str:
        details_lines.append(f"Timestamp: {ts_str}")
    if drivers:
        details_lines.append(f"Coinvolti: {drivers}")

    race_timeline_detail_var.set(" | ".join(details_lines))


def on_generate_race_timeline_click():
    """Handler per costruire la Race Timeline unificata."""
    global weather_last_data, weather_last_session_key

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info", "La timeline di gara è disponibile solo per sessioni Race o Sprint."
        )
        return

    status_var.set("Genero la timeline di gara unificando eventi...")
    root.update_idletasks()

    laps_data = current_laps_data if current_laps_session_key == session_key else []
    weather_data = weather_last_data if weather_last_session_key == session_key else []

    try:
        race_control_messages = fetch_race_control_messages(session_key)
    except RuntimeError as e:
        race_control_messages = []
        messagebox.showerror("Race Control", str(e))

    driver_number, _ = get_selected_driver_info()
    team_radio_messages = []
    if driver_number is not None:
        try:
            team_radio_messages = fetch_team_radio(session_key, driver_number)
        except RuntimeError as e:
            team_radio_messages = []
            messagebox.showerror("Team Radio", str(e))

    events = compute_race_timeline(
        session_key,
        current_results_data,
        current_pit_data,
        laps_data,
        weather_data,
        race_control_messages,
        team_radio_messages,
    )

    update_race_timeline_table(events)
    plots_notebook.select(race_timeline_tab)

    if events:
        status_var.set(f"Timeline di gara generata con {len(events)} eventi.")
    else:
        status_var.set("Nessun evento disponibile per la timeline.")

def clear_stints_summary_table():
    """Svuota la tabella riassuntiva degli stint."""
    global stints_summary_tree
    if stints_summary_tree is not None:
        for item in stints_summary_tree.get_children():
            stints_summary_tree.delete(item)


def clear_pit_strategy():
    """Svuota tabella e grafico di 'Pit stop & strategia'."""
    global pit_stats_tree, pit_strategy_canvas, pit_strategy_fig, pit_undercut_tree
    global pit_window_tree, pit_window_canvas, pit_window_fig

    if pit_stats_tree is not None:
        for item in pit_stats_tree.get_children():
            pit_stats_tree.delete(item)

    if pit_undercut_tree is not None:
        for item in pit_undercut_tree.get_children():
            pit_undercut_tree.delete(item)

    if pit_strategy_canvas is not None:
        pit_strategy_canvas.get_tk_widget().destroy()
        pit_strategy_canvas = None

    pit_strategy_fig = None

    if pit_window_tree is not None:
        for item in pit_window_tree.get_children():
            pit_window_tree.delete(item)

    if pit_window_canvas is not None:
        pit_window_canvas.get_tk_widget().destroy()
        pit_window_canvas = None

    pit_window_fig = None


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


def build_overtake_index(overtakes):
    """Costruisce dizionari con sorpassi effettuati e subiti per pilota."""
    made = {}
    suffered = {}
    if not isinstance(overtakes, list):
        return made, suffered

    for ot in overtakes:
        overtaking = ot.get("overtaking_driver_number")
        overtaken = ot.get("overtaken_driver_number")

        try:
            ovk_int = int(overtaking) if overtaking is not None else None
            ovn_int = int(overtaken) if overtaken is not None else None
        except (ValueError, TypeError):
            continue

        if ovk_int is not None:
            made[ovk_int] = made.get(ovk_int, 0) + 1
        if ovn_int is not None:
            suffered[ovn_int] = suffered.get(ovn_int, 0) + 1

    return made, suffered


def compute_battle_pressure_for_driver(session_key, driver_number, overtake_index=None):
    """Calcola indicatori di pressione/duello per un singolo pilota."""

    overtake_index = overtake_index or {"made": {}, "suffered": {}}

    try:
        intervals = fetch_intervals(session_key, driver_number)
    except RuntimeError:
        return None

    if not intervals:
        return None

    laps_idx = []
    gaps_leader = []
    gaps_prev = []

    for idx, entry in enumerate(intervals, start=1):
        lap_number = entry.get("lap_number")
        if not isinstance(lap_number, int):
            lap_number = idx

        laps_idx.append(lap_number)

        gap_leader = entry.get("gap_to_leader", None)
        gaps_leader.append(gap_leader if isinstance(gap_leader, (int, float)) else None)

        interval_val = entry.get("interval", None)
        if isinstance(interval_val, (int, float)):
            gaps_prev.append(interval_val)
        else:
            gaps_prev.append(None)

    total_valid = len([v for v in gaps_prev if isinstance(v, (int, float))])
    attack_laps = [lap for lap, val in zip(laps_idx, gaps_prev) if isinstance(val, (int, float)) and val < SLIPSTREAM_THRESHOLD]
    clean_laps = [lap for lap, val in zip(laps_idx, gaps_prev) if isinstance(val, (int, float)) and val > CLEAN_AIR_THRESHOLD]

    attack_pct = (len(attack_laps) / total_valid * 100.0) if total_valid else 0.0
    clean_pct = (len(clean_laps) / total_valid * 100.0) if total_valid else 0.0

    pressure_segments = find_pressure_stints(laps_idx, gaps_leader, gaps_prev)
    offensive = [
        s for s in pressure_segments if s.get("metric") == "interval" and s.get("trend") == "Avvicinamento"
    ]
    defensive = [
        s for s in pressure_segments if s.get("metric") == "interval" and s.get("trend") == "Allontanamento"
    ]

    pressure_delta_sum = sum(s.get("delta", 0) for s in offensive)
    defense_delta_sum = sum(s.get("delta", 0) for s in defensive)

    made = overtake_index.get("made", {}).get(int(driver_number), 0)
    suffered = overtake_index.get("suffered", {}).get(int(driver_number), 0)

    return {
        "driver_number": driver_number,
        "laps_total": total_valid,
        "attack_laps": len(attack_laps),
        "clean_laps": len(clean_laps),
        "attack_pct": attack_pct,
        "clean_pct": clean_pct,
        "pressure_stints": len(offensive),
        "pressure_delta": pressure_delta_sum,
        "defense_segments": len(defensive),
        "defense_delta": defense_delta_sum,
        "overtakes_made": made,
        "overtakes_suffered": suffered,
        "detail": {
            "laps": laps_idx,
            "intervals": gaps_prev,
            "attack_laps": attack_laps,
            "clean_laps": clean_laps,
            "pressure_segments": pressure_segments,
        },
    }


def compute_race_timeline(
    session_key,
    current_results_data,
    current_pit_data,
    current_laps_data,
    weather_last_data,
    race_control_messages,
    team_radio_messages,
):
    """Costruisce una timeline unificata degli eventi di gara."""

    events = []

    lap_time_index = {}
    driver_from_laps = None
    for lap in current_laps_data or []:
        lap_num = lap.get("lap_number")
        lap_dt = parse_iso_datetime(lap.get("date_start", ""))
        if isinstance(lap_num, int) and lap_dt is not None:
            lap_time_index[lap_num] = lap_dt
        if driver_from_laps is None and isinstance(lap.get("driver_number"), (int, float, str)):
            driver_from_laps = lap.get("driver_number")

    driver_name_cache = {}

    for res in current_results_data or []:
        dnum = res.get("driver_number")
        if dnum is None:
            continue
        try:
            d_int = int(dnum)
        except (ValueError, TypeError):
            continue
        driver_name_cache[d_int] = res.get("full_name") or fetch_driver_full_name(d_int, session_key)

    def resolve_driver_label(dnum):
        if dnum is None:
            return ""
        try:
            d_int = int(dnum)
        except (ValueError, TypeError):
            return str(dnum)
        if d_int in driver_name_cache:
            return driver_name_cache[d_int]
        name = fetch_driver_full_name(d_int, session_key)
        driver_name_cache[d_int] = name
        return name or f"Driver {d_int}"

    def resolve_timestamp(raw_ts, lap_number=None):
        parsed = parse_iso_datetime(raw_ts) if raw_ts else None
        if parsed is None and isinstance(lap_number, int):
            parsed = lap_time_index.get(lap_number)
        return parsed

    def add_event(ts, lap, ev_type, description, drivers_text=""):
        events.append(
            {
                "timestamp": ts,
                "lap": lap if isinstance(lap, int) else None,
                "type": ev_type,
                "description": description,
                "drivers": drivers_text,
            }
        )

    # Sorpassi
    overtakes = []
    try:
        overtakes = fetch_overtakes(session_key)
    except RuntimeError:
        overtakes = []

    for ot in overtakes:
        ts = resolve_timestamp(ot.get("date"))
        overtaking = resolve_driver_label(ot.get("overtaking_driver_number"))
        overtaken = resolve_driver_label(ot.get("overtaken_driver_number"))
        position = ot.get("position")
        pos_txt = f" per P{position}" if isinstance(position, int) else ""
        add_event(
            ts,
            None,
            "Sorpasso",
            f"{overtaking} supera {overtaken}{pos_txt}",
            f"{overtaking} vs {overtaken}",
        )

    # Pit stop
    for pit in current_pit_data or []:
        lap_num = pit.get("lap_number")
        ts = resolve_timestamp(pit.get("date"), lap_num)
        driver_label = resolve_driver_label(pit.get("driver_number"))
        pit_dur = pit.get("pit_duration")
        dur_txt = f" in {pit_dur:.3f}s" if isinstance(pit_dur, (int, float)) else ""
        add_event(
            ts,
            lap_num,
            "Pit stop",
            f"{driver_label} effettua un pit{dur_txt}",
            driver_label,
        )

    # Race control (inseriti sempre in timeline, ordinati per timestamp JSON)
    for msg in race_control_messages or []:
        lap_num = msg.get("lap_number")
        ts = resolve_timestamp(msg.get("date"), lap_num)
        category = msg.get("category") or "Race Control"
        flag_label = msg.get("flag") or ""
        scope = msg.get("scope") or ""
        message_text = msg.get("message", "")

        description_parts = []
        if flag_label:
            description_parts.append(str(flag_label))
        description_parts.append(message_text)
        if scope:
            description_parts.append(f"(scope: {scope})")

        description = " ".join(part for part in description_parts if part).strip()
        add_event(ts, lap_num, category, description)

    # Meteo: pioggia o variazioni track temp
    prev_rain = None
    prev_track_temp = None
    for w in weather_last_data or []:
        ts = resolve_timestamp(w.get("date"))
        rain = w.get("rainfall")
        track_temp = w.get("track_temperature")
        if isinstance(rain, (int, float)):
            if (prev_rain is None or prev_rain <= 0) and rain > 0:
                add_event(ts, None, "Meteo", f"Pioggia rilevata ({rain} mm/h)")
            prev_rain = rain
        if isinstance(track_temp, (int, float)) and isinstance(prev_track_temp, (int, float)):
            delta = track_temp - prev_track_temp
            if abs(delta) >= 5:
                trend = "aumenta" if delta > 0 else "diminuisce"
                add_event(ts, None, "Meteo", f"Track temp {trend} di {abs(delta):.1f}°C")
        if isinstance(track_temp, (int, float)):
            prev_track_temp = track_temp

    # Analisi gap su giri correnti
    laps_sorted = []
    try:
        laps_sorted = sorted(
            [l for l in current_laps_data or [] if isinstance(l.get("lap_number"), int)],
            key=lambda l: l.get("lap_number"),
        )
    except Exception:
        laps_sorted = current_laps_data or []

    laps_numbers = [l.get("lap_number") for l in laps_sorted]
    gap_leader_series = [
        l.get("gap_to_leader") if isinstance(l.get("gap_to_leader"), (int, float)) else None
        for l in laps_sorted
    ]
    interval_series = [
        l.get("interval") if isinstance(l.get("interval"), (int, float)) else None
        for l in laps_sorted
    ]

    if laps_numbers:
        pressure_segments = find_pressure_stints(
            laps_numbers,
            gap_leader_series,
            interval_series,
        )
        driver_label = resolve_driver_label(driver_from_laps)
        for seg in pressure_segments:
            ts = lap_time_index.get(seg.get("start_lap")) or lap_time_index.get(seg.get("end_lap"))
            trend_txt = "guadagna" if seg.get("trend") == "Avvicinamento" else "perde"
            metric = seg.get("metric")
            add_event(
                ts,
                seg.get("start_lap"),
                "Analisi gap",
                (
                    f"{driver_label} {trend_txt} rapidamente ({metric}, Δ {seg.get('delta', 0):+.2f}s) "
                    f"tra i giri {seg.get('start_lap')} e {seg.get('end_lap')}"
                ),
                driver_label,
            )

    # Team radio
    for tr in team_radio_messages or []:
        lap_num = tr.get("lap_number")
        ts = resolve_timestamp(tr.get("date"), lap_num)
        driver_label = resolve_driver_label(tr.get("driver_number") or driver_from_laps)
        lap_txt = f" (giro {lap_num})" if isinstance(lap_num, int) else ""
        add_event(ts, lap_num, "Team Radio", f"Team radio {driver_label}{lap_txt}", driver_label)

    def sort_key(ev):
        ts = ev.get("timestamp")
        if ts is None:
            ts = lap_time_index.get(ev.get("lap"))
        if ts is None:
            ts = datetime.max
        return ts

    return sorted(events, key=sort_key)


def on_compute_battle_pressure_click():
    """Callback per calcolare il Battle / Pressure Index per i piloti della sessione."""
    global current_results_data, battle_pressure_last_results, battle_pressure_session_key

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info", "L'analisi Battle/Pressure è disponibile solo per Race o Sprint.",
        )
        return

    if not current_results_data:
        messagebox.showinfo(
            "Info", "Carica prima i risultati della sessione per ottenere l'elenco piloti.",
        )
        return

    status_var.set("Calcolo Battle/Pressure Index in corso...")
    root.update_idletasks()

    overtakes = []
    try:
        overtakes = fetch_overtakes(session_key)
    except RuntimeError:
        overtakes = []

    made_idx, suffered_idx = build_overtake_index(overtakes)
    overtake_index = {"made": made_idx, "suffered": suffered_idx}

    results = []
    skipped = 0

    for res in current_results_data:
        driver_number = res.get("driver_number")
        if driver_number is None:
            continue

        driver_name = res.get("full_name") or fetch_driver_full_name(driver_number, session_key)
        team_name = res.get("team_name") or fetch_driver_team_name(driver_number, session_key)

        try:
            metrics = compute_battle_pressure_for_driver(session_key, driver_number, overtake_index)
        except Exception:
            metrics = None

        if not metrics:
            skipped += 1
            continue

        metrics.update({"driver_name": driver_name, "team_name": team_name})
        results.append(metrics)

    battle_pressure_last_results = results
    battle_pressure_session_key = session_key

    update_battle_pressure_table(results)
    plots_notebook.select(battle_pressure_tab)

    if not results:
        status_var.set("Nessun dato intervals disponibile per calcolare il Battle/Pressure Index.")
    else:
        status_msg = (
            f"Battle/Pressure Index calcolato per {len(results)} piloti."
        )
        if skipped:
            status_msg += f" Nessun dato disponibile per {skipped} piloti."
        status_var.set(status_msg)


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


# --------------------- Analisi meteo-strategia avanzata --------------------- #

def _build_weather_points_with_phase_markers():
    """Ritorna lista di campioni meteo con info utili e indici ordinati."""
    weather_points = []
    for idx, w in enumerate(weather_last_data or []):
        dt = parse_iso_datetime(w.get("date", ""))
        if dt is None:
            continue
        rainfall = w.get("rainfall")
        rainfall_val = float(rainfall) if isinstance(rainfall, (int, float)) else 0.0
        weather_points.append(
            {
                "idx": idx,
                "dt": dt,
                "rainfall": rainfall_val,
                "track_temp": w.get("track_temperature"),
                "humidity": w.get("humidity"),
            }
        )
    return weather_points


def _classify_weather_phase(sample_idx, rainfall_val, humidity_val, first_rain_idx, transition_range, first_rain_humidity):
    """Ritorna etichetta fase meteo per un campione meteo indicizzato."""
    if first_rain_idx is None:
        return "DRY"

    start_tr, end_tr = transition_range
    if start_tr <= sample_idx <= end_tr:
        return "TRANSITION"

    humidity_jump = False
    if isinstance(humidity_val, (int, float)) and isinstance(first_rain_humidity, (int, float)):
        humidity_jump = (humidity_val - first_rain_humidity) >= 8

    if rainfall_val > 0:
        return "WET"
    if humidity_jump:
        return "TRANSITION"
    return "DRY"


def on_compute_rain_impact_click():
    """Calcola impatto della pioggia sul passo gara del pilota selezionato."""
    session_key, _, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella risultati.")
        return

    if not weather_last_data or weather_last_session_key != session_key:
        messagebox.showinfo("Info", "Scarica prima i dati meteo della sessione.")
        return

    if not current_laps_data or current_laps_session_key != session_key:
        messagebox.showinfo("Info", "Carica prima i giri del pilota selezionato.")
        return

    clear_rain_impact_outputs()

    weather_points = _build_weather_points_with_phase_markers()
    if not weather_points:
        if weather_phase_info_var is not None:
            weather_phase_info_var.set("Nessun campione meteo valido per l'analisi.")
        return

    first_rain_idx = None
    for wp in weather_points:
        if wp.get("rainfall", 0) > 0:
            first_rain_idx = wp["idx"]
            break

    transition_range = (0, -1)
    first_rain_humidity = None
    if first_rain_idx is not None:
        transition_range = (max(0, first_rain_idx - 2), first_rain_idx + 2)
        first_match = next((w for w in weather_points if w["idx"] == first_rain_idx), None)
        first_rain_humidity = first_match.get("humidity") if first_match else None

    phases_data = {"DRY": [], "TRANSITION": [], "WET": []}

    for lap in current_laps_data:
        if lap.get("is_pit_out_lap", False):
            continue
        lap_time = lap.get("lap_duration")
        if not isinstance(lap_time, (int, float)):
            continue
        lap_dt = parse_iso_datetime(lap.get("date_start", ""))
        if lap_dt is None:
            continue

        nearest = min(
            weather_points,
            key=lambda wp: abs((lap_dt - wp["dt"]).total_seconds()),
        )

        phase_label = _classify_weather_phase(
            nearest["idx"],
            nearest.get("rainfall", 0),
            nearest.get("humidity"),
            first_rain_idx,
            transition_range,
            first_rain_humidity,
        )

        phases_data.setdefault(phase_label, []).append(
            {
                "lap_number": lap.get("lap_number"),
                "lap_time": float(lap_time),
                "rainfall": nearest.get("rainfall", 0),
            }
        )

    if all(not lst for lst in phases_data.values()):
        if weather_phase_info_var is not None:
            weather_phase_info_var.set("Nessun giro valido per calcolare l'impatto della pioggia.")
        return

    phase_rows = []
    colors = {"DRY": "tab:green", "TRANSITION": "tab:orange", "WET": "tab:blue"}
    markers = {"DRY": "o", "TRANSITION": "s", "WET": "^"}

    for phase_name, laps_list in phases_data.items():
        if not laps_list:
            continue
        times = [l["lap_time"] for l in laps_list if isinstance(l.get("lap_time"), float)]
        if not times:
            continue
        count = len(times)
        mean_val = sum(times) / count
        var = sum((t - mean_val) ** 2 for t in times) / count if count else 0
        std_val = math.sqrt(var) if count else None
        best_val = min(times)

        avg_str = format_time_from_seconds(mean_val)
        best_str = format_time_from_seconds(best_val)
        std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) else ""

        avg_rain = compute_mean([l.get("rainfall") for l in laps_list])
        if isinstance(avg_rain, (int, float)) and avg_rain > 0:
            note = "Pioggia leggera" if avg_rain < 1 else "Full wet/Inter"
        else:
            note = ""

        phase_rows.append(
            {
                "phase": phase_name,
                "laps": count,
                "avg": avg_str,
                "best": best_str,
                "std": std_str,
                "note": note,
                "plot_data": laps_list,
                "color": colors.get(phase_name, "tab:gray"),
                "marker": markers.get(phase_name, "o"),
            }
        )

    if weather_phase_tree is not None:
        for row in phase_rows:
            weather_phase_tree.insert(
                "",
                tk.END,
                values=(row["phase"], row["laps"], row["avg"], row["best"], row["std"], row["note"]),
            )

    if phase_rows:
        fig_local = Figure(figsize=(6, 3.2))
        ax = fig_local.add_subplot(111)
        for row in phase_rows:
            xs = [d.get("lap_number") for d in row["plot_data"] if isinstance(d.get("lap_number"), int)]
            ys = [d.get("lap_time") for d in row["plot_data"] if isinstance(d.get("lap_time"), float)]
            if not xs or not ys:
                continue
            ax.scatter(xs, ys, label=row["phase"], color=row["color"], marker=row["marker"], alpha=0.8)

        ax.set_xlabel("Giro")
        ax.set_ylabel("Lap time (s)")
        ax.set_title(f"Impatto pioggia - {driver_name}")
        ax.grid(True)
        ax.legend()

        global weather_phase_canvas, weather_phase_fig
        weather_phase_fig = fig_local
        weather_phase_canvas = FigureCanvasTkAgg(fig_local, master=weather_phase_plot_frame)
        weather_phase_canvas.get_tk_widget().pack(fill="both", expand=True)
        weather_phase_canvas.draw()

    if weather_phase_info_var is not None:
        laps_used = sum(row.get("laps", 0) for row in phase_rows)
        phase_names = [row.get("phase", "") for row in phase_rows]
        weather_phase_info_var.set(
            f"Analisi completata su {laps_used} giri. Fasi rilevate: {', '.join(phase_names)}."
        )


def on_compute_compound_weather_click():
    """Analizza performance dei compound rispetto alla temperatura pista."""
    session_key, _, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella risultati.")
        return

    if not weather_last_data or weather_last_session_key != session_key:
        messagebox.showinfo("Info", "Scarica prima i dati meteo della sessione.")
        return

    if not current_laps_data or current_laps_session_key != session_key:
        messagebox.showinfo("Info", "Carica prima i giri del pilota selezionato.")
        return

    if not current_stints_data:
        messagebox.showinfo("Info", "Nessun dato stint disponibile per il pilota selezionato.")
        return

    clear_compound_weather_outputs()

    weather_points = _build_weather_points_with_phase_markers()
    if not weather_points:
        if compound_weather_info_var is not None:
            compound_weather_info_var.set("Nessun campione meteo valido per associare track temp.")
        return

    stint_spans = []
    for stint in current_stints_data:
        lap_start = stint.get("lap_start")
        lap_end = stint.get("lap_end")
        compound = (stint.get("compound") or "").upper()
        if not isinstance(lap_start, int) or not isinstance(lap_end, int):
            continue
        stint_spans.append({"lap_start": lap_start, "lap_end": lap_end, "compound": compound})

    compound_data = {}

    for lap in current_laps_data:
        if lap.get("is_pit_out_lap", False):
            continue
        lap_time = lap.get("lap_duration")
        lap_number = lap.get("lap_number")
        if not isinstance(lap_time, (int, float)) or not isinstance(lap_number, int):
            continue
        lap_dt = parse_iso_datetime(lap.get("date_start", ""))
        if lap_dt is None:
            continue

        stint_for_lap = next(
            (s for s in stint_spans if s["lap_start"] <= lap_number <= s["lap_end"]),
            None,
        )
        if not stint_for_lap or not stint_for_lap.get("compound"):
            continue

        nearest_weather = min(
            weather_points,
            key=lambda wp: abs((lap_dt - wp["dt"]).total_seconds()),
        )
        track_temp = nearest_weather.get("track_temp")
        if not isinstance(track_temp, (int, float)):
            continue

        comp_key = stint_for_lap["compound"]
        compound_data.setdefault(comp_key, []).append(
            {
                "track_temp": float(track_temp),
                "lap_time": float(lap_time),
                "lap_number": lap_number,
            }
        )

    if not compound_data:
        if compound_weather_info_var is not None:
            compound_weather_info_var.set("Nessun giro valido per associare compound e meteo.")
        return

    rows = []
    default_colors = {
        "SOFT": "tab:red",
        "MEDIUM": "tab:orange",
        "HARD": "tab:gray",
        "INTERMEDIATE": "tab:green",
        "WET": "tab:blue",
    }

    for comp, samples in compound_data.items():
        temps = [s["track_temp"] for s in samples if isinstance(s.get("track_temp"), float)]
        times = [s["lap_time"] for s in samples if isinstance(s.get("lap_time"), float)]
        if not temps or not times:
            continue
        avg_temp = compute_mean(temps)
        avg_time = compute_mean(times)

        rows.append(
            {
                "compound": comp,
                "avg_temp": avg_temp,
                "avg_time": avg_time,
                "laps": len(times),
                "color": default_colors.get(comp, "tab:purple"),
                "samples": samples,
            }
        )

    if compound_weather_tree is not None:
        for row in rows:
            compound_weather_tree.insert(
                "",
                tk.END,
                values=(
                    row["compound"],
                    f"{row['avg_temp']:.1f}°C" if isinstance(row.get("avg_temp"), (int, float)) else "",
                    format_time_from_seconds(row.get("avg_time")),
                    row.get("laps", 0),
                ),
            )

    if rows:
        fig_local = Figure(figsize=(6, 3.2))
        ax = fig_local.add_subplot(111)

        for row in rows:
            temps = [s.get("track_temp") for s in row["samples"] if isinstance(s.get("track_temp"), float)]
            times = [s.get("lap_time") for s in row["samples"] if isinstance(s.get("lap_time"), float)]
            if not temps or not times:
                continue
            ax.scatter(temps, times, label=row["compound"], color=row["color"], alpha=0.8)

        ax.set_xlabel("Track temperature (°C)")
        ax.set_ylabel("Lap time (s)")
        ax.set_title(f"Compound vs track temp - {driver_name}")
        ax.grid(True)
        ax.legend()

        global compound_weather_canvas, compound_weather_fig
        compound_weather_fig = fig_local
        compound_weather_canvas = FigureCanvasTkAgg(fig_local, master=compound_weather_plot_frame)
        compound_weather_canvas.get_tk_widget().pack(fill="both", expand=True)
        compound_weather_canvas.draw()

    if compound_weather_info_var is not None:
        compound_weather_info_var.set(
            f"Analisi completata per {len(rows)} compound. Seleziona altri piloti per confrontare i dati."
        )


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
    global stats_tree, session_stats_last_results, session_stats_session_key

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

    session_stats_last_results = stats_per_driver
    session_stats_session_key = session_key

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

    plots_notebook.select(stats_tab)

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


# --------------------- Analisi degrado gomme (Practice) --------------------- #


def clear_tyre_wear_view():
    global tyre_wear_canvas, tyre_wear_fig, tyre_wear_level_label

    if tyre_wear_laps_tree is not None:
        for item in tyre_wear_laps_tree.get_children():
            tyre_wear_laps_tree.delete(item)

    if tyre_wear_canvas is not None:
        try:
            tyre_wear_canvas.get_tk_widget().destroy()
        except Exception:
            pass
        tyre_wear_canvas = None
        tyre_wear_fig = None

    if tyre_wear_info_var is not None:
        tyre_wear_info_var.set(
            "Analisi degrado gomme: seleziona una sessione Practice e un pilota."
        )

    default_texts = {
        "slope": "Degrado medio: --",
        "intercept": "Tempo stimato all'inizio: --",
        "r2": "Qualità del fit (R²): --",
        "diagnosis": "Diagnosi: --",
        "degradation_level": "Livello degrado: --",
    }
    for key, var in tyre_wear_results_vars.items():
        if var is not None:
            var.set(default_texts.get(key, "--"))

    if tyre_wear_level_label is not None:
        try:
            tyre_wear_level_label.config(fg="black")
        except Exception:
            pass


def get_pit_lap_numbers_for_driver(session_key, driver_number):
    try:
        dnum = int(driver_number)
    except (TypeError, ValueError):
        return set()

    pit_data = pit_cache.get(session_key)
    if pit_data is None:
        try:
            pit_data = fetch_pit_stops(session_key)
        except RuntimeError:
            pit_data = []

    lap_numbers = set()
    if isinstance(pit_data, list):
        for p in pit_data:
            dn = p.get("driver_number")
            lap_num = p.get("lap_number")
            if dn == dnum and isinstance(lap_num, int):
                lap_numbers.add(lap_num)
    return lap_numbers


def update_tyre_wear_laps_list(session_key, driver_number, driver_name):
    if tyre_wear_laps_tree is None:
        return

    if not current_laps_data or current_laps_session_key != session_key or current_laps_driver != driver_number:
        return

    for item in tyre_wear_laps_tree.get_children():
        tyre_wear_laps_tree.delete(item)

    try:
        laps_sorted = sorted(
            current_laps_data,
            key=lambda l: (l.get("lap_number", 9999) if isinstance(l.get("lap_number"), int) else 9999),
        )
    except Exception:
        laps_sorted = current_laps_data

    pit_laps = get_pit_lap_numbers_for_driver(session_key, driver_number)

    for lap in laps_sorted:
        lap_num = lap.get("lap_number", "")
        lap_time_str = format_time_from_seconds(lap.get("lap_duration", None))
        flags = []
        if lap.get("is_pit_out_lap", False):
            flags.append("Out-lap")
        if bool(lap.get("is_pit_in_lap", False)):
            flags.append("In-lap")
        if isinstance(lap_num, int) and lap_num in pit_laps:
            flags.append("Pit stop")
        flag_str = ", ".join(flags) if flags else ""

        tyre_wear_laps_tree.insert(
            "",
            tk.END,
            values=(lap_num, lap_time_str, flag_str),
        )

    if tyre_wear_info_var is not None:
        tyre_wear_info_var.set(
            f"Giri di {driver_name} caricati: seleziona manualmente i giri da analizzare."
        )


def classify_tyre_degradation(m: float) -> str:
    """
    Classifica il degrado gomme in base alla pendenza m (s/giro).
    Restituisce una stringa tra: 'Basso', 'Medio', 'Alto', 'Critico'.
    """
    try:
        slope = float(m)
    except (TypeError, ValueError):
        slope = 0.0

    if math.isnan(slope) or math.isinf(slope):
        slope = 0.0

    # Pendenze negative o molto piccole vengono considerate stabili.
    slope = max(0.0, slope)

    if slope < 0.05:
        return "Basso"
    if slope < 0.12:
        return "Medio"
    if slope < 0.20:
        return "Alto"
    return "Critico"


def compute_tyre_wear_linear_regression(lap_points):
    """Calcola smoothing leggero + regressione lineare e R²."""
    if not lap_points:
        raise ValueError("Nessun giro valido per il calcolo del degrado.")

    lap_points_sorted = sorted(lap_points, key=lambda x: x[0])
    x_vals = [lp[0] for lp in lap_points_sorted]
    y_vals = [lp[1] for lp in lap_points_sorted]

    y_smooth = []
    for idx, _ in enumerate(y_vals):
        neighbors = []
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(y_vals):
                neighbors.append(y_vals[j])
        y_smooth.append(sum(neighbors) / len(neighbors))

    coeffs = np.polyfit(x_vals, y_smooth, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    y_pred = [slope * x + intercept for x in x_vals]
    y_mean = sum(y_smooth) / len(y_smooth)
    ss_res = sum((ys - yp) ** 2 for ys, yp in zip(y_smooth, y_pred))
    ss_tot = sum((ys - y_mean) ** 2 for ys in y_smooth)
    r2 = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "x": x_vals,
        "y_raw": y_vals,
        "y_smooth": y_smooth,
        "y_pred": y_pred,
    }


def on_prepare_tyre_wear_click():
    global current_laps_data, current_laps_session_key, current_laps_driver

    session_key, session_type, _ = get_selected_session_info()
    session_name = get_selected_session_name()

    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_practice_session(session_name, session_type):
        messagebox.showinfo("Info", "Analisi degrado disponibile solo per sessioni Practice.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    if current_laps_session_key != session_key or current_laps_driver != driver_number:
        status_var.set(
            f"Scarico i giri del pilota {driver_number} per la sessione {session_key}..."
        )
        root.update_idletasks()
        try:
            laps_data = fetch_laps(session_key, driver_number)
        except RuntimeError as e:
            messagebox.showerror("Errore", str(e))
            status_var.set("Errore durante il recupero dei dati giri.")
            return
        current_laps_data = laps_data if isinstance(laps_data, list) else []
        current_laps_session_key = session_key
        current_laps_driver = driver_number

    clear_tyre_wear_view()
    update_tyre_wear_laps_list(session_key, driver_number, driver_name)
    plots_notebook.select(tyre_wear_tab)
    status_var.set(
        "Seleziona almeno 3 giri validi (no out-lap/in-lap/pit) e premi 'Calcola degrado gomme'."
    )


def on_compute_tyre_wear_click():
    global current_laps_data, current_laps_session_key, current_laps_driver
    global tyre_wear_canvas, tyre_wear_fig, tyre_wear_level_label

    session_key, session_type, _ = get_selected_session_info()
    session_name = get_selected_session_name()

    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_practice_session(session_name, session_type):
        messagebox.showinfo("Info", "Disponibile solo per sessioni Practice.")
        return

    driver_number, driver_name = get_selected_driver_info()
    if driver_number is None:
        messagebox.showinfo("Info", "Seleziona un pilota nella tabella dei risultati.")
        return

    if current_laps_session_key != session_key or current_laps_driver != driver_number:
        try:
            laps_data = fetch_laps(session_key, driver_number)
        except RuntimeError as e:
            messagebox.showerror("Errore", str(e))
            return
        current_laps_data = laps_data if isinstance(laps_data, list) else []
        current_laps_session_key = session_key
        current_laps_driver = driver_number
        update_tyre_wear_laps_list(session_key, driver_number, driver_name)

    if not current_laps_data:
        messagebox.showinfo("Info", "Nessun dato giri disponibile per questo pilota.")
        return

    if tyre_wear_laps_tree is None:
        messagebox.showinfo("Info", "La lista dei giri non è pronta.")
        return

    selection = tyre_wear_laps_tree.selection()
    if not selection:
        messagebox.showinfo("Info", "Seleziona uno o più giri da analizzare.")
        return

    lap_map = {}
    for lap in current_laps_data:
        ln = lap.get("lap_number")
        if isinstance(ln, int):
            lap_map[ln] = lap

    selected_laps = []
    for item_id in selection:
        values = tyre_wear_laps_tree.item(item_id, "values")
        if not values:
            continue
        lap_num = values[0]
        try:
            lap_num_int = int(lap_num)
        except (TypeError, ValueError):
            continue
        lap_data = lap_map.get(lap_num_int)
        if lap_data:
            selected_laps.append(lap_data)

    if not selected_laps:
        messagebox.showinfo("Info", "Selezione non valida: nessun giro trovato.")
        return

    pit_laps = get_pit_lap_numbers_for_driver(session_key, driver_number)

    valid_points = []
    for lap in selected_laps:
        ln = lap.get("lap_number")
        ld = lap.get("lap_duration")
        if not (isinstance(ln, int) and isinstance(ld, (int, float))):
            continue
        if lap.get("is_pit_out_lap", False):
            continue
        if lap.get("is_pit_in_lap", False):
            continue
        if ln in pit_laps:
            continue
        valid_points.append((ln, float(ld)))

    if not valid_points:
        messagebox.showinfo(
            "Info", "Nessun giro valido (no out-lap/in-lap/pit) nella selezione."
        )
        return

    times = [p[1] for p in valid_points]
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_dev = math.sqrt(variance)
    threshold = mean_time + 2 * std_dev
    filtered_points = [p for p in valid_points if p[1] <= threshold]

    if len(filtered_points) < 3:
        messagebox.showinfo(
            "Info",
            "Seleziona almeno 3 giri validi (dopo aver escluso outlier/out-lap/in-lap/pit).",
        )
        return

    try:
        results = compute_tyre_wear_linear_regression(filtered_points)
    except ValueError as e:
        messagebox.showinfo("Info", str(e))
        return

    slope = results["slope"]
    intercept = results["intercept"]
    r2 = results["r2"]

    slope_display = max(0.0, slope)
    degradation_level = classify_tyre_degradation(slope)
    if slope_display < 0.05:
        diagnosis = "Gomme stabili"
    elif slope_display < 0.15:
        diagnosis = "Degrado normale"
    else:
        diagnosis = "Degrado elevato"

    if tyre_wear_results_vars.get("slope") is not None:
        tyre_wear_results_vars["slope"].set(f"Degrado medio: +{slope_display:.3f} s/giro")
    if tyre_wear_results_vars.get("intercept") is not None:
        tyre_wear_results_vars["intercept"].set(
            f"Tempo stimato all'inizio: {format_time_from_seconds(max(0, intercept))}"
        )
    if tyre_wear_results_vars.get("r2") is not None:
        tyre_wear_results_vars["r2"].set(f"Qualità del fit (R²): {r2:.2f}")
    if tyre_wear_results_vars.get("diagnosis") is not None:
        tyre_wear_results_vars["diagnosis"].set(f"Diagnosi: {diagnosis}")
    if tyre_wear_results_vars.get("degradation_level") is not None:
        tyre_wear_results_vars["degradation_level"].set(
            f"Livello degrado: {degradation_level} (+{slope_display:.3f} s/giro)"
        )

    level_colors = {
        "Basso": "green",
        "Medio": "goldenrod",
        "Alto": "darkorange",
        "Critico": "red",
    }
    if tyre_wear_level_label is not None:
        try:
            tyre_wear_level_label.config(fg=level_colors.get(degradation_level, "black"))
        except Exception:
            pass

    if tyre_wear_info_var is not None:
        tyre_wear_info_var.set(
            "Analisi completata: curva smussata e retta di regressione disegnate."
        )

    if tyre_wear_canvas is not None:
        try:
            tyre_wear_canvas.get_tk_widget().destroy()
        except Exception:
            pass

    fig = Figure(figsize=(6.5, 3.5))
    ax = fig.add_subplot(111)
    ax.scatter(results["x"], results["y_raw"], color="tab:blue", label="Dati reali")
    ax.plot(results["x"], results["y_smooth"], color="tab:orange", label="Smussati")
    ax.plot(results["x"], results["y_pred"], color="tab:green", label="Regressione")
    ax.set_xlabel("Giro")
    ax.set_ylabel("Tempo sul giro (s)")
    ax.set_title(
        f"Degrado gomme – {driver_name} – Sessione Practice"
    )
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()

    tyre_wear_fig_local = fig
    tyre_wear_fig = tyre_wear_fig_local
    tyre_wear_canvas = FigureCanvasTkAgg(tyre_wear_fig_local, master=tyre_wear_plot_frame)
    tyre_wear_canvas.get_tk_widget().pack(fill="both", expand=True)
    tyre_wear_canvas.draw()

    plots_notebook.select(tyre_wear_tab)
    status_var.set(
        "Analisi degrado gomme aggiornata: verifica grafico e indicatori testuali nella tab dedicata."
    )

# --------------------- Analisi pit stop & strategia --------------------- #

def compute_undercut_overcut_events(session_key, session_type, pit_data, results_data):
    """
    Identifica eventi di undercut/overcut tra piloti vicini in classifica.

    Ritorna una lista di dict con chiavi:
        driver_a, driver_b, type ("Undercut"/"Overcut"), pit_a, pit_b, laps, delta
    """
    if not is_race_like(session_type):
        return []

    if not pit_data or not results_data:
        return []

    # Ordina i risultati per posizione per considerare solo piloti ravvicinati
    ordered_results = []
    for r in results_data:
        pos = r.get("position", None)
        dnum = r.get("driver_number", None)
        if isinstance(pos, int) and isinstance(dnum, int):
            ordered_results.append((pos, dnum))
    ordered_results.sort(key=lambda x: x[0])

    if not ordered_results:
        return []

    # Mappa driver -> lista giri pit ordinati
    pits_by_driver = {}
    for p in pit_data:
        dnum = p.get("driver_number", None)
        lap = p.get("lap_number", None)
        if not isinstance(dnum, int) or not isinstance(lap, int):
            continue
        pits_by_driver.setdefault(dnum, []).append(lap)

    for laps in pits_by_driver.values():
        laps.sort()

    if not pits_by_driver:
        return []

    laps_cache = {}
    name_cache = {}

    def get_driver_name(dnum):
        if dnum not in name_cache:
            name_cache[dnum] = fetch_driver_full_name(dnum, session_key)
        return name_cache.get(dnum, "")

    def get_lap_map(dnum):
        """Restituisce lap_number -> lap_duration per giri validi (no out-lap)."""
        if dnum in laps_cache:
            return laps_cache[dnum]

        try:
            laps_data = fetch_laps(session_key, dnum)
        except RuntimeError:
            laps_data = []

        lap_map = {}
        if isinstance(laps_data, list):
            for lap in laps_data:
                ln = lap.get("lap_number", None)
                ld = lap.get("lap_duration", None)
                if not isinstance(ln, int) or not isinstance(ld, (int, float)):
                    continue
                if lap.get("is_pit_out_lap", False):
                    continue
                lap_map[ln] = float(ld)

        laps_cache[dnum] = lap_map
        return lap_map

    events = []
    MAX_POSITION_GAP = 3
    MIN_GAIN_SECONDS = 0.7

    def analyze_pair(first_driver, second_driver):
        """
        first_driver: pilota che pitta per primo
        second_driver: pilota che rimane fuori almeno un giro in più
        """
        results = []
        first_pits = pits_by_driver.get(first_driver, [])
        second_pits = pits_by_driver.get(second_driver, [])
        if not first_pits or not second_pits:
            return results

        for pit_first in first_pits:
            later_pits = [lp for lp in second_pits if lp > pit_first]
            if not later_pits:
                continue
            pit_second = min(later_pits)

            laps_first = get_lap_map(first_driver)
            laps_second = get_lap_map(second_driver)
            if not laps_first or not laps_second:
                continue

            comparison = []
            for lap_idx in range(pit_first + 1, pit_second):
                t_first = laps_first.get(lap_idx, None)
                t_second = laps_second.get(lap_idx, None)
                if not isinstance(t_first, (int, float)) or not isinstance(t_second, (int, float)):
                    continue
                comparison.append((lap_idx, t_second - t_first))

            if not comparison:
                continue

            total_gain = sum(delta for _, delta in comparison)
            if abs(total_gain) < MIN_GAIN_SECONDS:
                continue

            laps_range = (
                f"{comparison[0][0]}-{comparison[-1][0]}"
                if len(comparison) > 1
                else str(comparison[0][0])
            )

            if total_gain >= MIN_GAIN_SECONDS:
                results.append(
                    {
                        "driver_a": get_driver_name(first_driver),
                        "driver_b": get_driver_name(second_driver),
                        "type": "Undercut",
                        "pit_a": pit_first,
                        "pit_b": pit_second,
                        "laps": laps_range,
                        "delta": total_gain,
                    }
                )
            elif total_gain <= -MIN_GAIN_SECONDS:
                results.append(
                    {
                        "driver_a": get_driver_name(second_driver),
                        "driver_b": get_driver_name(first_driver),
                        "type": "Overcut",
                        "pit_a": pit_second,
                        "pit_b": pit_first,
                        "laps": laps_range,
                        "delta": abs(total_gain),
                    }
                )
            # Considera solo il primo pit utile della coppia per evitare duplicati
            if results:
                break

        return results

    for idx, (pos_a, d_a) in enumerate(ordered_results):
        for pos_b, d_b in ordered_results[idx + 1 :]:
            if pos_b - pos_a > MAX_POSITION_GAP:
                break
            # Primo scenario: d_a pitta prima di d_b
            events.extend(analyze_pair(d_a, d_b))
            # Secondo scenario: d_b pitta prima di d_a
            events.extend(analyze_pair(d_b, d_a))

    return events


def update_pit_undercut_table(events):
    """Aggiorna la tabella degli undercut/overcut nella sezione pit."""
    global pit_undercut_tree

    if pit_undercut_tree is None:
        return

    for item in pit_undercut_tree.get_children():
        pit_undercut_tree.delete(item)

    for ev in events:
        pit_undercut_tree.insert(
            "",
            tk.END,
            values=(
                ev.get("driver_a", ""),
                ev.get("driver_b", ""),
                ev.get("type", ""),
                ev.get("pit_a", ""),
                ev.get("pit_b", ""),
                ev.get("laps", ""),
                f"{ev.get('delta', 0):.3f}",
            ),
        )


def compute_pit_window_analysis(session_key, results_data, pit_data):
    """Calcola pit window e posizione virtuale dopo un pit per ogni pilota/giro."""

    def build_cumulative_times(laps_list):
        """Restituisce lista ordinata (lap, cumulative_time)."""
        try:
            ordered = sorted(
                [
                    (l.get("lap_number"), l.get("lap_duration"))
                    for l in laps_list
                    if isinstance(l.get("lap_number"), int)
                ],
                key=lambda x: x[0],
            )
        except Exception:
            ordered = []

        cumulative = []
        total = 0.0
        for lap_num, lap_dur in ordered:
            if not isinstance(lap_dur, (int, float)):
                continue
            total += float(lap_dur)
            cumulative.append((lap_num, total))
        return cumulative

    def time_at_or_before(cum_list, lap):
        last_time = None
        for ln, t in cum_list:
            if ln > lap:
                break
            last_time = t
        return last_time

    if not results_data:
        return [], None, set()

    # Stima tempo perso al pit
    pit_durations = [
        float(p.get("pit_duration"))
        for p in pit_data
        if isinstance(p.get("pit_duration"), (int, float))
    ]
    pit_loss_estimate = sum(pit_durations) / len(pit_durations) if pit_durations else 20.0

    # Dati per ogni pilota
    drivers = []
    for r in results_data:
        dnum = r.get("driver_number")
        if isinstance(dnum, int):
            drivers.append(dnum)
        else:
            try:
                drivers.append(int(dnum))
            except (ValueError, TypeError):
                continue

    laps_by_driver = {}
    for dnum in drivers:
        try:
            laps = fetch_laps(session_key, dnum)
        except RuntimeError:
            laps = []
        cumulative = build_cumulative_times(laps)
        if cumulative:
            laps_by_driver[dnum] = cumulative

    if not laps_by_driver:
        return [], pit_loss_estimate, set()

    # Identifica giri con SC/VSC
    sc_vsc_laps = set()
    try:
        rc_messages = fetch_race_control_messages(session_key)
    except RuntimeError:
        rc_messages = []

    for msg in rc_messages:
        flag_val = str(msg.get("flag", "")).lower()
        text_val = str(msg.get("message", "")).lower()
        if any(term in flag_val or term in text_val for term in ["vsc", "virtual safety", "sc", "safety car"]):
            lap_num = msg.get("lap_number")
            if isinstance(lap_num, int):
                sc_vsc_laps.add(lap_num)

    entries = []

    for dnum, cumulative in laps_by_driver.items():
        driver_name = fetch_driver_full_name(dnum, session_key)

        for lap_num, cum_time in cumulative:
            # Posizione reale stimata al lap
            current_pos = 1
            opponent_times = []
            for other_num, other_cum in laps_by_driver.items():
                if other_num == dnum:
                    continue
                other_time = time_at_or_before(other_cum, lap_num)
                opponent_times.append(other_time)
                if other_time is not None and other_time <= cum_time:
                    current_pos += 1

            predicted_time = cum_time + pit_loss_estimate
            virtual_pos = 1
            for other_time in opponent_times:
                if other_time is not None and other_time <= predicted_time:
                    virtual_pos += 1

            gaps_ahead = [
                other_time - predicted_time
                for other_time in opponent_times
                if other_time is not None and other_time > predicted_time
            ]
            gap_relevant = min(gaps_ahead) if gaps_ahead else None

            delta_pos = virtual_pos - current_pos
            if gap_relevant is not None and gap_relevant >= 1.5:
                comment = "Safe window"
                comment_tag = "safe"
            elif gap_relevant is not None and gap_relevant >= 0.5:
                comment = "Rischio traffico"
                comment_tag = "risky"
            else:
                comment = "Traffico elevato"
                comment_tag = "traffic"

            if delta_pos > 0:
                comment += f" (perderebbe {delta_pos} posizioni)"
            elif delta_pos < 0:
                comment += f" (guadagno stimato {abs(delta_pos)} posizioni)"

            if lap_num in sc_vsc_laps:
                comment += " | Possibile SC/VSC"

            entries.append(
                {
                    "driver_number": dnum,
                    "driver_name": driver_name,
                    "lap": lap_num,
                    "virtual_position": virtual_pos,
                    "gap_relevant": gap_relevant,
                    "pit_loss": pit_loss_estimate,
                    "comment": comment,
                    "comment_tag": comment_tag,
                }
            )

    return entries, pit_loss_estimate, sc_vsc_laps


def update_pit_window_table(entries):
    """Aggiorna la tabella della pit window nella sezione strategia."""
    global pit_window_tree

    if pit_window_tree is None:
        return

    for item in pit_window_tree.get_children():
        pit_window_tree.delete(item)

    for e in entries:
        gap_val = e.get("gap_relevant")
        if isinstance(gap_val, (int, float)):
            gap_str = f"{gap_val:+.2f}"
        else:
            gap_str = "--"

        pit_window_tree.insert(
            "",
            tk.END,
            values=(
                e.get("driver_name", ""),
                e.get("lap", ""),
                e.get("virtual_position", ""),
                gap_str,
                f"{e.get('pit_loss', 0):.2f}",
                e.get("comment", ""),
            ),
        )


def update_pit_window_plot(entries, sc_vsc_laps):
    """Disegna il grafico di posizione virtuale per pit window."""
    global pit_window_canvas, pit_window_fig

    if pit_window_canvas is not None:
        pit_window_canvas.get_tk_widget().destroy()
        pit_window_canvas = None

    pit_window_fig = None

    if not entries:
        return

    grouped = {}
    for e in entries:
        grouped.setdefault(e.get("driver_name", ""), []).append(e)

    fig = Figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)

    for driver_name, data in grouped.items():
        try:
            ordered = sorted(data, key=lambda x: x.get("lap", 0))
        except Exception:
            ordered = data

        valid_points = [
            (
                d.get("lap"),
                d.get("virtual_position"),
                d.get("comment_tag"),
            )
            for d in ordered
            if isinstance(d.get("lap"), int) and isinstance(d.get("virtual_position"), int)
        ]

        if valid_points:
            laps, positions, _tags = zip(*valid_points)
            ax.plot(laps, positions, label=driver_name, linewidth=1.3)

            safe_laps = [l for l, _, t in valid_points if t == "safe"]
            safe_pos = [p for (l, p, t) in valid_points if t == "safe"]
            risky_laps = [l for l, _, t in valid_points if t != "safe"]
            risky_pos = [p for (l, p, t) in valid_points if t != "safe"]

            if safe_laps and safe_pos:
                ax.scatter(safe_laps, safe_pos, color="green", s=18, marker="o", alpha=0.8)
            if risky_laps and risky_pos:
                ax.scatter(risky_laps, risky_pos, color="orange", s=14, marker="x", alpha=0.7)

    for lap in sc_vsc_laps:
        ax.axvspan(lap - 0.5, lap + 0.5, color="yellow", alpha=0.15)

    ax.set_xlabel("Giro")
    ax.set_ylabel("Posizione virtuale dopo pit")
    ax.set_title("Pit window & posizione virtuale")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()
    if grouped:
        ax.legend(fontsize=8, ncol=2)

    pit_window_fig = fig

    pit_window_canvas_local = FigureCanvasTkAgg(fig, master=pit_window_plot_frame)
    pit_window_canvas_local.get_tk_widget().pack(fill="both", expand=True)
    pit_window_canvas_local.draw()

    pit_window_canvas = pit_window_canvas_local

def on_compute_undercut_analysis_click():
    """Callback per il bottone di analisi undercut/overcut."""
    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "L'analisi undercut/overcut è disponibile solo per sessioni Race o Sprint.",
        )
        return

    if not current_pit_data:
        messagebox.showinfo("Info", "Nessun pit stop disponibile per questa sessione.")
        return

    if not current_results_data:
        messagebox.showinfo(
            "Info", "Prima carica i risultati della sessione con il relativo bottone."
        )
        return

    status_var.set("Calcolo analisi undercut/overcut in corso...")
    root.update_idletasks()

    try:
        events = compute_undercut_overcut_events(
            session_key, session_type, current_pit_data, current_results_data
        )
    except Exception as e:
        status_var.set("Errore durante l'analisi undercut/overcut.")
        messagebox.showerror("Analisi undercut/overcut", str(e))
        return

    update_pit_undercut_table(events)
    plots_notebook.select(pit_strategy_tab)

    if events:
        status_var.set(
            f"Trovati {len(events)} scenari di undercut/overcut. Seleziona la tab 'Pit stop & strategia'."
        )
    else:
        status_var.set("Nessun undercut/overcut rilevato con i dati disponibili.")


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

    plots_notebook.select(pit_strategy_tab)

    status_var.set(
        "Analisi pit stop aggiornata: tabella riassuntiva per pilota e grafico pit stop per giro "
        "disponibili nella tab 'Pit stop & strategia'."
    )


def on_compute_pit_window_click():
    """Callback per calcolare pit window e posizione virtuale dopo pit."""
    global current_pit_data, current_results_data

    session_key, session_type, _ = get_selected_session_info()
    if session_key is None:
        messagebox.showinfo("Info", "Seleziona prima una sessione.")
        return

    if not is_race_like(session_type):
        messagebox.showinfo(
            "Info",
            "L'analisi pit window è disponibile solo per sessioni Race o Sprint.",
        )
        return

    if not current_results_data:
        messagebox.showinfo(
            "Info", "Prima carica i risultati della sessione con il relativo bottone."
        )
        return

    status_var.set("Calcolo pit window e posizione virtuale in corso...")
    root.update_idletasks()

    try:
        entries, pit_loss, sc_vsc_laps = compute_pit_window_analysis(
            session_key, current_results_data, current_pit_data
        )
    except Exception as e:
        status_var.set("Errore durante il calcolo della pit window.")
        messagebox.showerror("Pit window", str(e))
        return

    update_pit_window_table(entries)
    update_pit_window_plot(entries, sc_vsc_laps)

    plots_notebook.select(pit_strategy_tab)

    if not entries:
        status_var.set(
            "Nessuna stima disponibile: dati giri o pit insufficienti per calcolare la pit window."
        )
    else:
        pit_loss_str = f"{pit_loss:.2f}s" if pit_loss is not None else "--"
        status_var.set(
            f"Pit window calcolata (pit loss stimato {pit_loss_str}). Controlla tabella e grafico nella tab 'Pit stop & strategia'."
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
    clear_battle_pressure_table()
    clear_battle_pressure_table()

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
    global current_results_session_key, current_pit_session_key
    DRIVER_CACHE = {}
    DRIVER_PROFILE_CACHE = {}
    clear_driver_plots()
    clear_weather_plot()
    clear_stats_table()
    clear_pit_strategy()
    clear_race_control_table()

    current_pit_data = []
    current_results_data = []
    current_results_session_key = None
    current_pit_session_key = None

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
    current_results_session_key = session_key

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
        current_pit_session_key = session_key

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
    global current_stints_data, current_laps_data, current_laps_session_key, current_laps_driver
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
    current_laps_driver = driver_number

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

    update_team_radio_table(session_key, driver_number, title_driver)

    plots_notebook.select(gap_tab)

    status_var.set(
        f"Grafici, analisi gomme e tabella giri aggiornati per pilota {driver_number} ({title_driver}). "
        "La tabella pit stop mostra tutti i pit della sessione; la tab Meteo mostra l'evoluzione meteo. "
        "Premi 'Recupera messaggi Race Control' per caricare i messaggi della sessione. "
        "La tab Team Radio contiene le registrazioni scaricate per il pilota selezionato."
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
    plots_notebook.select(race_control_tab)

    status_var.set(
        f"Messaggi Race Control aggiornati per la sessione {session_key}."
    )


# --------------------- Costruzione GUI --------------------- #

root = tk.Tk()
root.title(
    "f1DataAnalyzer"
)
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
initial_w = min(1500, max(1100, int(screen_w * 0.9)))
initial_h = min(950, max(780, int(screen_h * 0.9)))
root.geometry(f"{initial_w}x{initial_h}")
root.minsize(max(900, min(screen_w - 140, initial_w)), max(680, min(screen_h - 160, initial_h)))
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

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


def create_scrollable_tab(parent, padding=6, style="Card.TFrame"):
    """Create a scrollable container for notebook tabs to avoid overflow."""

    container = ttk.Frame(parent)
    canvas = tk.Canvas(container, highlightthickness=0, background=DARK_BG)
    vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    hsb = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    content = ttk.Frame(canvas, padding=padding, style=style)
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _resize_content(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(window_id, width=canvas.winfo_width())

    content.bind("<Configure>", _resize_content)
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

    return container, content


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

actions = [
    ("Mostra risultati sessione", on_fetch_results_click),
    ("Grafici & giri pilota", on_show_driver_plots_click),
    ("Statistiche lap time", on_compute_session_stats_click),
    ("Pit stop & strategia", on_compute_pit_strategy_click),
    ("Analisi undercut/overcut", on_compute_undercut_analysis_click),
    ("Pit window & virtual position", on_compute_pit_window_click),
    ("Race Control pilota", on_fetch_race_control_click),
    ("Team radio pilota", on_fetch_team_radio_click),
    ("Lift & Coast pilota", on_compute_lift_and_coast_click),
    ("Degrado gomme (Practice)", on_prepare_tyre_wear_click),
]

for idx, (text, command) in enumerate(actions):
    row, col = divmod(idx, 4)
    btn = ttk.Button(actions_frame, text=text, command=command)
    btn.grid(row=row, column=col, padx=4, pady=2, sticky="ew")

actions_frame.columnconfigure((0, 1, 2, 3), weight=1)

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

gap_tab, gap_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
battle_pressure_tab, battle_pressure_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
stints_tab, stints_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
tyre_wear_tab, tyre_wear_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
race_control_tab, race_control_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
team_radio_tab, team_radio_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
lift_coast_tab, lift_coast_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
weather_tab, weather_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
stats_tab, stats_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
pit_strategy_tab, pit_strategy_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")
race_timeline_tab, race_timeline_tab_frame = create_scrollable_tab(plots_notebook, padding=6, style="Card.TFrame")

plots_notebook.add(gap_tab, text="Grafico distacchi")
plots_notebook.add(battle_pressure_tab, text="Battaglie & Pressure Index")
plots_notebook.add(stints_tab, text="Gomme: mappa & analisi")
plots_notebook.add(tyre_wear_tab, text="Degrado Gomme")
plots_notebook.add(race_control_tab, text="Race Control")
plots_notebook.add(team_radio_tab, text="Team Radio")
plots_notebook.add(lift_coast_tab, text="Lift & Coast")
plots_notebook.add(weather_tab, text="Meteo sessione")
plots_notebook.add(stats_tab, text="Statistiche piloti")
plots_notebook.add(pit_strategy_tab, text="Pit stop & strategia")
plots_notebook.add(race_timeline_tab, text="Race Timeline")

# --- Contenuto tab Battaglie & Pressure Index --- #
battle_pressure_info_var = tk.StringVar(
    value=(
        "Analizza quanto un pilota è vicino all'auto davanti (attacco) o in aria pulita. "
        "Premi 'Calcola Battle/Pressure Index' dopo aver caricato una sessione Race/Sprint."
    )
)

battle_pressure_info_label = ttk.Label(
    battle_pressure_tab_frame,
    textvariable=battle_pressure_info_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
)
battle_pressure_info_label.pack(fill="x", padx=5, pady=(5, 2))

battle_pressure_buttons_frame = ttk.Frame(
    battle_pressure_tab_frame, style="Card.TFrame"
)
battle_pressure_buttons_frame.pack(fill="x", padx=5, pady=(0, 4))

battle_pressure_button = ttk.Button(
    battle_pressure_buttons_frame,
    text="Calcola Battle/Pressure Index",
    command=on_compute_battle_pressure_click,
)
battle_pressure_button.pack(side="left")

battle_pressure_table_frame = ttk.Frame(
    battle_pressure_tab_frame, style="Card.TFrame"
)
battle_pressure_table_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

battle_pressure_columns = (
    "driver",
    "team",
    "attack_laps",
    "clean_laps",
    "pressure_stints",
    "pressure_delta",
    "defense_segments",
    "overtakes_suffered",
    "overtakes_made",
    "driver_id_hidden",
)

battle_pressure_tree = ttk.Treeview(
    battle_pressure_table_frame,
    columns=battle_pressure_columns,
    show="headings",
    displaycolumns=battle_pressure_columns[:-1],
    height=9,
)

battle_pressure_tree.heading("driver", text="Pilota")
battle_pressure_tree.heading("team", text="Team")
battle_pressure_tree.heading("attack_laps", text="Giri in attacco")
battle_pressure_tree.heading("clean_laps", text="Aria pulita")
battle_pressure_tree.heading("pressure_stints", text="Pressure stint")
battle_pressure_tree.heading("pressure_delta", text="Δ totale (s)")
battle_pressure_tree.heading("defense_segments", text="Segmenti allontanamento")
battle_pressure_tree.heading("overtakes_suffered", text="Sorpassi subiti")
battle_pressure_tree.heading("overtakes_made", text="Sorpassi fatti")

battle_pressure_tree.column("driver", width=200, anchor="w")
battle_pressure_tree.column("team", width=180, anchor="w")
battle_pressure_tree.column("attack_laps", width=170, anchor="center")
battle_pressure_tree.column("clean_laps", width=150, anchor="center")
battle_pressure_tree.column("pressure_stints", width=120, anchor="center")
battle_pressure_tree.column("pressure_delta", width=120, anchor="center")
battle_pressure_tree.column("defense_segments", width=170, anchor="center")
battle_pressure_tree.column("overtakes_suffered", width=130, anchor="center")
battle_pressure_tree.column("overtakes_made", width=120, anchor="center")
battle_pressure_tree.column("driver_id_hidden", width=1, anchor="center")

battle_pressure_tree.bind("<<TreeviewSelect>>", on_battle_pressure_select)

battle_pressure_vsb = ttk.Scrollbar(
    battle_pressure_table_frame,
    orient="vertical",
    command=battle_pressure_tree.yview,
)
battle_pressure_hsb = ttk.Scrollbar(
    battle_pressure_table_frame,
    orient="horizontal",
    command=battle_pressure_tree.xview,
)
battle_pressure_tree.configure(
    yscrollcommand=battle_pressure_vsb.set, xscrollcommand=battle_pressure_hsb.set
)

battle_pressure_tree.grid(row=0, column=0, sticky="nsew")
battle_pressure_vsb.grid(row=0, column=1, sticky="ns")
battle_pressure_hsb.grid(row=1, column=0, sticky="ew")

battle_pressure_table_frame.rowconfigure(0, weight=1)
battle_pressure_table_frame.columnconfigure(0, weight=1)

ttk.Label(
    battle_pressure_tab_frame,
    text="Timeline giri attacco/difesa (pilota selezionato):",
    font=("", 9, "bold"),
).pack(anchor="w", padx=5, pady=(0, 2))

battle_pressure_plot_frame = ttk.Frame(
    battle_pressure_tab_frame, style="Card.TFrame"
)
battle_pressure_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

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
race_control_hsb = ttk.Scrollbar(
    race_control_frame, orient="horizontal", command=race_control_tree.xview
)
race_control_tree.configure(
    yscrollcommand=race_control_vsb.set, xscrollcommand=race_control_hsb.set
)

race_control_tree.grid(row=0, column=0, sticky="nsew")
race_control_vsb.grid(row=0, column=1, sticky="ns")
race_control_hsb.grid(row=1, column=0, sticky="ew")

race_control_frame.rowconfigure(0, weight=1)
race_control_frame.columnconfigure(0, weight=1)

# --- Contenuto tab Team Radio --- #
team_radio_info_var = tk.StringVar(
    value=(
        "Team radio: seleziona una sessione e un pilota dai risultati, poi premi "
        "'Team radio pilota' per caricare i messaggi disponibili."
    )
)
team_radio_info_label = ttk.Label(
    team_radio_tab_frame,
    textvariable=team_radio_info_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
)
team_radio_info_label.pack(fill="x", padx=5, pady=(5, 2))

team_radio_columns = ("timestamp", "lap_number", "recording_url")
team_radio_tree = ttk.Treeview(
    team_radio_tab_frame,
    columns=team_radio_columns,
    show="headings",
    displaycolumns=("timestamp", "lap_number"),
    height=12,
)

team_radio_tree.heading("timestamp", text="Data / Ora (UTC)")
team_radio_tree.heading("lap_number", text="Giro")

team_radio_tree.column("timestamp", width=170, anchor="center")
team_radio_tree.column("lap_number", width=70, anchor="center")
team_radio_tree.column("recording_url", width=0, minwidth=0, stretch=False)

team_radio_vsb = ttk.Scrollbar(
    team_radio_tab_frame, orient="vertical", command=team_radio_tree.yview
)
team_radio_tree.configure(yscrollcommand=team_radio_vsb.set)

team_radio_tree.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
team_radio_vsb.pack(side="right", fill="y", padx=(0, 5), pady=5)

team_radio_tree.bind("<Double-1>", on_play_team_radio_click)

team_radio_buttons = ttk.Frame(team_radio_tab_frame, style="Card.TFrame")
team_radio_buttons.pack(fill="x", padx=5, pady=(0, 5))

ttk.Button(
    team_radio_buttons,
    text="Riproduci selezione",
    command=on_play_team_radio_click,
).pack(side="left", padx=(0, 6))

# --- Contenuto tab Lift & Coast --- #
lift_coast_info_var = tk.StringVar(
    value=(
        "Lift & Coast: seleziona sessione Race/Sprint e pilota, carica i giri, scegli manualmente fino a 5 giri "
        "(multi-selezione) e premi 'Lift & Coast pilota' per calcolare i segmenti di rilascio e veleggio."
    )
)
lift_coast_info_label = ttk.Label(
    lift_coast_tab_frame,
    textvariable=lift_coast_info_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
)
lift_coast_info_label.pack(fill="x", padx=5, pady=(5, 2))

lift_coast_selected_laps_var = tk.StringVar(value="Giri selezionati: 0 / 5")
lift_coast_selection_label_var = tk.StringVar(
    value="Carica i giri del pilota selezionato. Puoi scegliere manualmente fino a 5 giri da analizzare; se non selezioni nulla, verranno analizzati tutti i giri."
)

lift_coast_selection_frame = ttk.LabelFrame(
    lift_coast_tab_frame,
    text="Selezione giri Lift & Coast (max 5)",
    style="Card.TLabelframe",
)
lift_coast_selection_frame.pack(fill="x", padx=5, pady=(0, 5))

selection_header = ttk.Frame(lift_coast_selection_frame, style="Card.TFrame")
selection_header.pack(fill="x", padx=4, pady=4)
ttk.Label(
    selection_header,
    textvariable=lift_coast_selection_label_var,
    anchor="w",
    wraplength=1200,
).pack(fill="x", side="left", expand=True)
ttk.Button(
    selection_header,
    text="Aggiorna giri disponibili",
    command=on_refresh_lift_coast_laps_click,
).pack(side="right", padx=(6, 0))

selection_body = ttk.Frame(lift_coast_selection_frame, style="Card.TFrame")
selection_body.pack(fill="x", padx=4, pady=(0, 6))

lift_coast_lap_listbox = tk.Listbox(
    selection_body,
    selectmode="extended",
    height=6,
    exportselection=False,
)
lift_coast_lap_listbox.pack(side="left", fill="x", expand=True)
lift_coast_lap_listbox.bind("<<ListboxSelect>>", on_lift_coast_selection_change)
lift_coast_lap_scroll = ttk.Scrollbar(
    selection_body, orient="vertical", command=lift_coast_lap_listbox.yview
)
lift_coast_lap_listbox.configure(yscrollcommand=lift_coast_lap_scroll.set)
lift_coast_lap_scroll.pack(side="left", fill="y", padx=(2, 0))

ttk.Label(
    lift_coast_selection_frame,
    textvariable=lift_coast_selected_laps_var,
    anchor="w",
    style="Info.TLabel",
).pack(fill="x", padx=4, pady=(0, 4))

lift_coast_summary_frame = ttk.LabelFrame(
    lift_coast_tab_frame,
    text="Riepilogo per giro",
    style="Card.TLabelframe",
)
lift_coast_summary_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

lift_coast_per_lap_container = ttk.Frame(
    lift_coast_summary_frame, style="Card.TFrame"
)
lift_coast_per_lap_container.pack(fill="both", expand=True)

lift_coast_columns = ("lap", "total", "pct")
lift_coast_per_lap_tree = ttk.Treeview(
    lift_coast_per_lap_container,
    columns=lift_coast_columns,
    show="headings",
    height=8,
)
lift_coast_per_lap_tree.heading("lap", text="Giro")
lift_coast_per_lap_tree.heading("total", text="Durata totale L&C (s)")
lift_coast_per_lap_tree.heading("pct", text="% sul giro")

lift_coast_per_lap_tree.column("lap", width=80, anchor="center")
lift_coast_per_lap_tree.column("total", width=200, anchor="center")
lift_coast_per_lap_tree.column("pct", width=150, anchor="center")

lift_coast_per_lap_vsb = ttk.Scrollbar(
    lift_coast_per_lap_container,
    orient="vertical",
    command=lift_coast_per_lap_tree.yview,
)
lift_coast_per_lap_hsb = ttk.Scrollbar(
    lift_coast_per_lap_container,
    orient="horizontal",
    command=lift_coast_per_lap_tree.xview,
)
lift_coast_per_lap_tree.configure(
    yscrollcommand=lift_coast_per_lap_vsb.set,
    xscrollcommand=lift_coast_per_lap_hsb.set,
)

lift_coast_per_lap_tree.grid(row=0, column=0, sticky="nsew")
lift_coast_per_lap_vsb.grid(row=0, column=1, sticky="ns")
lift_coast_per_lap_hsb.grid(row=1, column=0, sticky="ew")

lift_coast_per_lap_container.rowconfigure(0, weight=1)
lift_coast_per_lap_container.columnconfigure(0, weight=1)

lift_coast_segments_frame = ttk.LabelFrame(
    lift_coast_tab_frame,
    text="Segmenti Lift & Coast",
    style="Card.TLabelframe",
)
lift_coast_segments_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

lift_coast_segments_container = ttk.Frame(
    lift_coast_segments_frame, style="Card.TFrame"
)
lift_coast_segments_container.pack(fill="both", expand=True)

lift_coast_segments_columns = (
    "lap",
    "segment",
    "start",
    "end",
    "duration",
)
lift_coast_segments_tree = ttk.Treeview(
    lift_coast_segments_container,
    columns=lift_coast_segments_columns,
    show="headings",
    height=10,
)

lift_coast_segments_tree.heading("lap", text="Giro")
lift_coast_segments_tree.heading("segment", text="# Segmento")
lift_coast_segments_tree.heading("start", text="Inizio")
lift_coast_segments_tree.heading("end", text="Fine")
lift_coast_segments_tree.heading("duration", text="Durata (s)")

lift_coast_segments_tree.column("lap", width=80, anchor="center")
lift_coast_segments_tree.column("segment", width=110, anchor="center")
lift_coast_segments_tree.column("start", width=200, anchor="center")
lift_coast_segments_tree.column("end", width=200, anchor="center")
lift_coast_segments_tree.column("duration", width=130, anchor="center")

lift_coast_segments_vsb = ttk.Scrollbar(
    lift_coast_segments_container,
    orient="vertical",
    command=lift_coast_segments_tree.yview,
)
lift_coast_segments_hsb = ttk.Scrollbar(
    lift_coast_segments_container,
    orient="horizontal",
    command=lift_coast_segments_tree.xview,
)
lift_coast_segments_tree.configure(
    yscrollcommand=lift_coast_segments_vsb.set,
    xscrollcommand=lift_coast_segments_hsb.set,
)

lift_coast_segments_tree.grid(row=0, column=0, sticky="nsew")
lift_coast_segments_vsb.grid(row=0, column=1, sticky="ns")
lift_coast_segments_hsb.grid(row=1, column=0, sticky="ew")

lift_coast_segments_container.rowconfigure(0, weight=1)
lift_coast_segments_container.columnconfigure(0, weight=1)

# --- Contenuto tab Race Timeline --- #
race_timeline_info = tk.StringVar(
    value=(
        "Genera una sequenza di eventi unificata combinando sorpassi, pit, Race Control, meteo, "
        "analisi gap e team radio (per il pilota selezionato)."
    )
)
ttk.Label(
    race_timeline_tab_frame,
    textvariable=race_timeline_info,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
).pack(fill="x", padx=5, pady=(5, 2))

timeline_buttons_frame = ttk.Frame(race_timeline_tab_frame, style="Card.TFrame")
timeline_buttons_frame.pack(fill="x", padx=5, pady=(0, 4))

ttk.Button(
    timeline_buttons_frame,
    text="Genera Timeline Gara",
    command=on_generate_race_timeline_click,
).pack(side="left")
ttk.Button(
    timeline_buttons_frame,
    text="Esporta Timeline",
    command=on_export_race_timeline_click,
).pack(side="left", padx=(6, 0))

race_timeline_table = ttk.Frame(race_timeline_tab_frame, style="Card.TFrame")
race_timeline_table.pack(fill="both", expand=True, padx=5, pady=5)

race_timeline_columns = ("timestamp", "lap", "type", "description", "drivers")
race_timeline_tree = ttk.Treeview(
    race_timeline_table,
    columns=race_timeline_columns,
    show="headings",
    height=14,
)

race_timeline_tree.heading("timestamp", text="Data / Ora")
race_timeline_tree.heading("lap", text="Giro")
race_timeline_tree.heading("type", text="Tipo")
race_timeline_tree.heading("description", text="Descrizione")
race_timeline_tree.heading("drivers", text="Pilota/i")

race_timeline_tree.column("timestamp", width=170, anchor="center")
race_timeline_tree.column("lap", width=70, anchor="center")
race_timeline_tree.column("type", width=110, anchor="center")
race_timeline_tree.column("description", width=780, anchor="w")
race_timeline_tree.column("drivers", width=170, anchor="w")

race_timeline_vsb = ttk.Scrollbar(
    race_timeline_table, orient="vertical", command=race_timeline_tree.yview
)
race_timeline_hsb = ttk.Scrollbar(
    race_timeline_table, orient="horizontal", command=race_timeline_tree.xview
)
race_timeline_tree.configure(
    yscrollcommand=race_timeline_vsb.set, xscrollcommand=race_timeline_hsb.set
)

race_timeline_tree.grid(row=0, column=0, sticky="nsew")
race_timeline_vsb.grid(row=0, column=1, sticky="ns")
race_timeline_hsb.grid(row=1, column=0, sticky="ew")

race_timeline_table.rowconfigure(0, weight=1)
race_timeline_table.columnconfigure(0, weight=1)

race_timeline_tree.bind("<<TreeviewSelect>>", on_race_timeline_select)

race_timeline_detail_var = tk.StringVar(
    value="Timeline pronta: premi 'Genera Timeline Gara' per creare una sequenza di eventi."
)
ttk.Label(
    race_timeline_tab_frame,
    textvariable=race_timeline_detail_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
).pack(fill="x", padx=5, pady=(0, 5))

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

# --- Contenuto tab Degrado Gomme (Practice) --- #
tyre_wear_info_var = tk.StringVar(
    value="Analisi degrado gomme: seleziona una sessione Practice e un pilota."
)
ttk.Label(
    tyre_wear_tab_frame,
    textvariable=tyre_wear_info_var,
    anchor="w",
    wraplength=1250,
    style="Info.TLabel",
).pack(fill="x", padx=5, pady=(4, 4))

tyre_wear_list_frame = ttk.LabelFrame(
    tyre_wear_tab_frame, text="Giri disponibili (selezione multipla)"
)
tyre_wear_list_frame.pack(fill="both", expand=True, padx=5, pady=(0, 4))

tyre_wear_columns = ("lap_number", "lap_time", "note")
tyre_wear_laps_tree = ttk.Treeview(
    tyre_wear_list_frame,
    columns=tyre_wear_columns,
    show="headings",
    height=9,
    selectmode="extended",
)
tyre_wear_laps_tree.heading("lap_number", text="Giro")
tyre_wear_laps_tree.heading("lap_time", text="Lap time")
tyre_wear_laps_tree.heading("note", text="Note")

tyre_wear_laps_tree.column("lap_number", width=70, anchor="center")
tyre_wear_laps_tree.column("lap_time", width=120, anchor="center")
tyre_wear_laps_tree.column("note", width=280, anchor="w")

tyre_wear_vsb = ttk.Scrollbar(
    tyre_wear_list_frame,
    orient="vertical",
    command=tyre_wear_laps_tree.yview,
)
tyre_wear_laps_tree.configure(yscrollcommand=tyre_wear_vsb.set)

tyre_wear_laps_tree.grid(row=0, column=0, sticky="nsew")
tyre_wear_vsb.grid(row=0, column=1, sticky="ns")

tyre_wear_list_frame.rowconfigure(0, weight=1)
tyre_wear_list_frame.columnconfigure(0, weight=1)

tyre_wear_controls = ttk.Frame(tyre_wear_tab_frame, style="Card.TFrame")
tyre_wear_controls.pack(fill="x", padx=5, pady=(0, 4))

ttk.Button(
    tyre_wear_controls,
    text="Calcola degrado gomme",
    command=on_compute_tyre_wear_click,
).pack(side="left")

tyre_wear_results_frame = ttk.LabelFrame(tyre_wear_tab_frame, text="Risultati sintetici")
tyre_wear_results_frame.pack(fill="x", padx=5, pady=(0, 4))

tyre_wear_results_vars = {
    "slope": tk.StringVar(value="Degrado medio: --"),
    "intercept": tk.StringVar(value="Tempo stimato all'inizio: --"),
    "r2": tk.StringVar(value="Qualità del fit (R²): --"),
    "diagnosis": tk.StringVar(value="Diagnosi: --"),
    "degradation_level": tk.StringVar(value="Livello degrado: --"),
}

ttk.Label(tyre_wear_results_frame, textvariable=tyre_wear_results_vars["slope"], anchor="w").grid(
    row=0, column=0, sticky="w", padx=5, pady=2
)
ttk.Label(
    tyre_wear_results_frame, textvariable=tyre_wear_results_vars["intercept"], anchor="w"
).grid(row=0, column=1, sticky="w", padx=5, pady=2)
ttk.Label(tyre_wear_results_frame, textvariable=tyre_wear_results_vars["r2"], anchor="w").grid(
    row=1, column=0, sticky="w", padx=5, pady=2
)
ttk.Label(
    tyre_wear_results_frame, textvariable=tyre_wear_results_vars["diagnosis"], anchor="w"
).grid(row=1, column=1, sticky="w", padx=5, pady=2)
tyre_wear_level_label = tk.Label(
    tyre_wear_results_frame,
    textvariable=tyre_wear_results_vars["degradation_level"],
    anchor="w",
)
tyre_wear_level_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

tyre_wear_results_frame.columnconfigure(0, weight=1)
tyre_wear_results_frame.columnconfigure(1, weight=1)

ttk.Label(
    tyre_wear_tab_frame,
    text="Grafico smoothing + regressione",
    font=("", 9, "bold"),
).pack(anchor="w", padx=5)

tyre_wear_plot_frame = ttk.Frame(tyre_wear_tab_frame, style="Card.TFrame")
tyre_wear_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 6))

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

weather_advanced_section = ttk.LabelFrame(
    weather_tab_frame, text="Analisi meteo-strategia avanzata"
)
weather_advanced_section.pack(fill="both", expand=True, padx=5, pady=(0, 5))

weather_phase_section = ttk.LabelFrame(
    weather_advanced_section, text="Impatto pioggia sul passo gara"
)
weather_phase_section.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)

weather_phase_info_var = tk.StringVar(
    value="Analizza l'impatto della pioggia sul passo gara del pilota selezionato."
)
ttk.Label(
    weather_phase_section,
    textvariable=weather_phase_info_var,
    anchor="w",
    wraplength=580,
).pack(fill="x", padx=5, pady=(4, 2))

weather_phase_buttons = ttk.Frame(weather_phase_section)
weather_phase_buttons.pack(fill="x", padx=5, pady=(0, 4))

ttk.Button(
    weather_phase_buttons,
    text="Analisi impatto pioggia",
    command=on_compute_rain_impact_click,
).pack(side="left")

weather_phase_table_frame = ttk.Frame(weather_phase_section)
weather_phase_table_frame.pack(fill="both", expand=True, padx=5, pady=(0, 4))

phase_columns = ("phase", "laps", "avg", "best", "std", "note")
weather_phase_tree = ttk.Treeview(
    weather_phase_table_frame,
    columns=phase_columns,
    show="headings",
    height=6,
)
weather_phase_tree.heading("phase", text="Phase")
weather_phase_tree.heading("laps", text="Laps")
weather_phase_tree.heading("avg", text="Avg lap time")
weather_phase_tree.heading("best", text="Best lap")
weather_phase_tree.heading("std", text="σ lap")
weather_phase_tree.heading("note", text="Note")

weather_phase_tree.column("phase", width=100, anchor="center")
weather_phase_tree.column("laps", width=60, anchor="center")
weather_phase_tree.column("avg", width=120, anchor="center")
weather_phase_tree.column("best", width=120, anchor="center")
weather_phase_tree.column("std", width=90, anchor="center")
weather_phase_tree.column("note", width=200, anchor="w")

weather_phase_vsb = ttk.Scrollbar(
    weather_phase_table_frame, orient="vertical", command=weather_phase_tree.yview
)
weather_phase_tree.configure(yscrollcommand=weather_phase_vsb.set)

weather_phase_tree.grid(row=0, column=0, sticky="nsew")
weather_phase_vsb.grid(row=0, column=1, sticky="ns")

weather_phase_table_frame.rowconfigure(0, weight=1)
weather_phase_table_frame.columnconfigure(0, weight=1)

weather_phase_plot_frame = ttk.Frame(weather_phase_section)
weather_phase_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

compound_weather_section = ttk.LabelFrame(
    weather_advanced_section, text="Scelta compound vs condizioni pista"
)
compound_weather_section.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)

compound_weather_info_var = tk.StringVar(
    value="Analizza come i compound performano rispetto alla track temperature."
)
ttk.Label(
    compound_weather_section,
    textvariable=compound_weather_info_var,
    anchor="w",
    wraplength=580,
).pack(fill="x", padx=5, pady=(4, 2))

compound_weather_buttons = ttk.Frame(compound_weather_section)
compound_weather_buttons.pack(fill="x", padx=5, pady=(0, 4))

ttk.Button(
    compound_weather_buttons,
    text="Analisi compound vs meteo",
    command=on_compute_compound_weather_click,
).pack(side="left")

compound_weather_table_frame = ttk.Frame(compound_weather_section)
compound_weather_table_frame.pack(fill="both", expand=True, padx=5, pady=(0, 4))

compound_columns = ("compound", "track_temp", "avg_lap", "laps")
compound_weather_tree = ttk.Treeview(
    compound_weather_table_frame,
    columns=compound_columns,
    show="headings",
    height=6,
)
compound_weather_tree.heading("compound", text="Compound")
compound_weather_tree.heading("track_temp", text="Track temp media")
compound_weather_tree.heading("avg_lap", text="Lap medio")
compound_weather_tree.heading("laps", text="Giri")

compound_weather_tree.column("compound", width=120, anchor="center")
compound_weather_tree.column("track_temp", width=140, anchor="center")
compound_weather_tree.column("avg_lap", width=130, anchor="center")
compound_weather_tree.column("laps", width=70, anchor="center")

compound_weather_vsb = ttk.Scrollbar(
    compound_weather_table_frame,
    orient="vertical",
    command=compound_weather_tree.yview,
)
compound_weather_tree.configure(yscrollcommand=compound_weather_vsb.set)

compound_weather_tree.grid(row=0, column=0, sticky="nsew")
compound_weather_vsb.grid(row=0, column=1, sticky="ns")

compound_weather_table_frame.rowconfigure(0, weight=1)
compound_weather_table_frame.columnconfigure(0, weight=1)

compound_weather_plot_frame = ttk.Frame(compound_weather_section)
compound_weather_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

weather_advanced_section.columnconfigure(0, weight=1)
weather_advanced_section.columnconfigure(1, weight=1)
weather_advanced_section.rowconfigure(0, weight=1)

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

undercut_label = ttk.Label(
    pit_strategy_tab_frame,
    text="Undercut/overcut tra piloti ravvicinati:",
    font=("", 9, "bold"),
)
undercut_label.pack(anchor="w", padx=5, pady=(4, 0))

pit_undercut_frame = ttk.Frame(pit_strategy_tab_frame)
pit_undercut_frame.pack(fill="x", expand=False, padx=5, pady=(0, 4))

pit_undercut_columns = (
    "driver_a",
    "driver_b",
    "event_type",
    "pit_a",
    "pit_b",
    "laps",
    "delta",
)

pit_undercut_tree = ttk.Treeview(
    pit_undercut_frame,
    columns=pit_undercut_columns,
    show="headings",
    height=5,
)

pit_undercut_tree.heading("driver_a", text="Driver A")
pit_undercut_tree.heading("driver_b", text="Driver B")
pit_undercut_tree.heading("event_type", text="Tipo")
pit_undercut_tree.heading("pit_a", text="Giro pit A")
pit_undercut_tree.heading("pit_b", text="Giro pit B")
pit_undercut_tree.heading("laps", text="Giri analizzati")
pit_undercut_tree.heading("delta", text="Delta guadagnato (s)")

pit_undercut_tree.column("driver_a", width=160, anchor="w")
pit_undercut_tree.column("driver_b", width=160, anchor="w")
pit_undercut_tree.column("event_type", width=90, anchor="center")
pit_undercut_tree.column("pit_a", width=80, anchor="center")
pit_undercut_tree.column("pit_b", width=80, anchor="center")
pit_undercut_tree.column("laps", width=130, anchor="center")
pit_undercut_tree.column("delta", width=140, anchor="center")

pit_undercut_vsb = ttk.Scrollbar(
    pit_undercut_frame,
    orient="vertical",
    command=pit_undercut_tree.yview,
)
pit_undercut_tree.configure(yscrollcommand=pit_undercut_vsb.set)

pit_undercut_tree.grid(row=0, column=0, sticky="nsew")
pit_undercut_vsb.grid(row=0, column=1, sticky="ns")

pit_undercut_frame.rowconfigure(0, weight=1)
pit_undercut_frame.columnconfigure(0, weight=1)

ttk.Label(
    pit_strategy_tab_frame,
    text="Pit window & posizione virtuale dopo pit:",
    font=("", 9, "bold"),
).pack(anchor="w", padx=5, pady=(4, 0))

pit_window_frame = ttk.Frame(pit_strategy_tab_frame)
pit_window_frame.pack(fill="x", expand=False, padx=5, pady=(0, 4))

pit_window_columns = (
    "driver_name",
    "lap",
    "virtual_position",
    "gap",
    "pit_loss",
    "comment",
)

pit_window_tree = ttk.Treeview(
    pit_window_frame,
    columns=pit_window_columns,
    show="headings",
    height=6,
)

pit_window_tree.heading("driver_name", text="Pilota")
pit_window_tree.heading("lap", text="Giro")
pit_window_tree.heading("virtual_position", text="Posizione virtuale")
pit_window_tree.heading("gap", text="Gap rilevante (s)")
pit_window_tree.heading("pit_loss", text="Pit loss stimato (s)")
pit_window_tree.heading("comment", text="Commento")

pit_window_tree.column("driver_name", width=170, anchor="w")
pit_window_tree.column("lap", width=60, anchor="center")
pit_window_tree.column("virtual_position", width=120, anchor="center")
pit_window_tree.column("gap", width=120, anchor="center")
pit_window_tree.column("pit_loss", width=130, anchor="center")
pit_window_tree.column("comment", width=400, anchor="w")

pit_window_vsb = ttk.Scrollbar(
    pit_window_frame,
    orient="vertical",
    command=pit_window_tree.yview,
)
pit_window_tree.configure(yscrollcommand=pit_window_vsb.set)

pit_window_tree.grid(row=0, column=0, sticky="nsew")
pit_window_vsb.grid(row=0, column=1, sticky="ns")

pit_window_frame.rowconfigure(0, weight=1)
pit_window_frame.columnconfigure(0, weight=1)

ttk.Label(
    pit_strategy_tab_frame,
    text="Distribuzione pit stop per giro:",
    font=("", 9, "bold")
).pack(anchor="w", padx=5, pady=(0, 2))

pit_strategy_plot_frame = ttk.Frame(pit_strategy_tab_frame, style="Card.TFrame")
pit_strategy_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

ttk.Label(
    pit_strategy_tab_frame,
    text="Posizione virtuale dopo pit (grafico):",
    font=("", 9, "bold"),
).pack(anchor="w", padx=5, pady=(0, 2))

pit_window_plot_frame = ttk.Frame(pit_strategy_tab_frame, style="Card.TFrame")
pit_window_plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

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
