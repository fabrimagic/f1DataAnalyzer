import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request
import urllib.error
import json

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Endpoint API --- #
API_SESSIONS_URL = "https://api.openf1.org/v1/sessions?year="
API_RESULTS_URL = "https://api.openf1.org/v1/session_result?session_key="
API_DRIVERS_URL = "https://api.openf1.org/v1/drivers?driver_number={driver_number}&session_key={session_key}"
API_INTERVALS_URL = "https://api.openf1.org/v1/intervals?session_key={session_key}&driver_number={driver_number}"
API_STINTS_URL = "https://api.openf1.org/v1/stints?session_key={session_key}&driver_number={driver_number}"
API_LAPS_URL = "https://api.openf1.org/v1/laps?session_key={session_key}&driver_number={driver_number}"
API_PIT_URL = "https://api.openf1.org/v1/pit?session_key={session_key}"
# Meteo: uso session_key come richiesto
API_WEATHER_URL = "https://api.openf1.org/v1/weather?session_key={session_key}"

# Cache per i nomi dei piloti: chiave = (session_key, driver_number)
DRIVER_CACHE = {}

# Canvas matplotlib per i grafici (uno per tab)
gap_plot_canvas = None
stints_plot_canvas = None

# Dati e handler per il click sul grafico distacchi
gap_fig = None
gap_ax = None
gap_click_cid = None
gap_click_data = {"laps": [], "gap_leader": [], "gap_prev": []}
gap_point_info_var = None  # inizializzato dopo la creazione della GUI

# Meteo
weather_canvas = None
weather_fig = None
weather_plot_frame = None
weather_info_var = None


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
    except (ValueError, TypeError):
        return ""

    cache_key = (session_key, dn_key)
    if cache_key in DRIVER_CACHE:
        return DRIVER_CACHE[cache_key]

    url = API_DRIVERS_URL.format(driver_number=dn_key, session_key=session_key)

    try:
        data = http_get_json(url)
    except RuntimeError:
        DRIVER_CACHE[cache_key] = ""
        return ""

    if not isinstance(data, list) or not data:
        DRIVER_CACHE[cache_key] = ""
        return ""

    first = data[0]
    full_name = first.get("full_name", "")
    if not isinstance(full_name, str):
        full_name = ""

    DRIVER_CACHE[cache_key] = full_name
    return full_name


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

    # Disconnette eventuale handler click dal grafico distacchi
    if gap_fig is not None and gap_click_cid is not None:
        try:
            gap_fig.canvas.mpl_disconnect(gap_click_cid)
        except Exception:
            pass

    gap_fig = None
    gap_ax = None
    gap_click_cid = None
    gap_click_data = {
        "laps": [],
        "gap_leader": [],
        "gap_prev": [],
        "raw_gap_leader": [],
        "raw_gap_prev": [],
    }

    if gap_plot_canvas is not None:
        gap_plot_canvas.get_tk_widget().destroy()
        gap_plot_canvas = None

    if stints_plot_canvas is not None:
        stints_plot_canvas.get_tk_widget().destroy()
        stints_plot_canvas = None

    if gap_point_info_var is not None:
        gap_point_info_var.set(
            "Clicca un punto sul grafico distacchi per vedere gap_to_leader e interval (raw)."
        )


def clear_weather_plot():
    """Rimuove il grafico meteo, se presente."""
    global weather_canvas, weather_fig, weather_info_var
    if weather_canvas is not None:
        weather_canvas.get_tk_widget().destroy()
        weather_canvas = None
    weather_fig = None
    if weather_info_var is not None:
        weather_info_var.set(
            "Meteo sessione: in attesa di una sessione selezionata."
        )


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
    raw_gap_leader = gap_click_data.get("raw_gap_leader", [])
    raw_gap_prev = gap_click_data.get("raw_gap_prev", [])

    # Trova il giro più vicino sull'asse X
    nearest_idx = min(range(len(laps)), key=lambda i: abs(laps[i] - x))
    if abs(laps[nearest_idx] - x) > 0.5:
        # Click troppo lontano da un giro intero -> ignora
        return

    lap_num = laps[nearest_idx]
    gl = gaps_leader[nearest_idx]
    gp = gaps_prev[nearest_idx]
    gl_raw = raw_gap_leader[nearest_idx] if len(raw_gap_leader) > nearest_idx else gl
    gp_raw = raw_gap_prev[nearest_idx] if len(raw_gap_prev) > nearest_idx else gp

    gl_str = f"{gl:.3f}" if isinstance(gl, (int, float)) else "N/A"
    gp_str = f"{gp:.3f}" if isinstance(gp, (int, float)) else "N/A"

    gl_raw_str = str(gl_raw) if gl_raw is not None else "N/A"
    gp_raw_str = str(gp_raw) if gp_raw is not None else "N/A"

    if gap_point_info_var is not None:
        gap_point_info_var.set(
            f"Giro {lap_num}: "
            f"gap_to_leader = {gl_str} s (raw: {gl_raw_str}), "
            f"interval = {gp_str} s (raw: {gp_raw_str})"
        )


# --------------------- Meteo sessione --------------------- #

def update_weather_for_session(session_key: int):
    """
    Scarica e visualizza il meteo per la sessione selezionata,
    usando esclusivamente la session_key.
    """
    global weather_canvas, weather_fig, weather_plot_frame, weather_info_var

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

    # Calcola presenza pioggia
    rain_present = any((isinstance(r, (int, float)) and r > 0) for r in rainfall_vals)
    if weather_info_var is not None:
        weather_info_var.set(
            f"Pioggia rilevata: {'Sì' if rain_present else 'No'} "
            "(in base ai campioni meteo della sessione)."
        )

    # Figura con due sottografi:
    # 1) track temp + air temp
    # 2) humidity + wind speed
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


# --------------------- Callback GUI --------------------- #

def on_fetch_sessions_click():
    """Carica l'elenco sessioni per l'anno inserito."""
    global DRIVER_CACHE
    DRIVER_CACHE = {}
    clear_driver_plots()
    clear_weather_plot()

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

    # Svuota tabelle
    for tree in (sessions_tree, results_tree, laps_tree, pits_tree):
        for item in tree.get_children():
            tree.delete(item)

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
    global DRIVER_CACHE
    DRIVER_CACHE = {}
    clear_driver_plots()
    clear_weather_plot()

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

    # Svuota tabelle risultati, giri e pit stop
    for tree in (results_tree, laps_tree, pits_tree):
        for item in tree.get_children():
            tree.delete(item)

    if not results:
        status_var.set(f"Nessun risultato trovato per la sessione {session_key}.")
        # Meteo comunque mostrato, se disponibile
        update_weather_for_session(session_key)
        return

    # Ordina per posizione se possibile
    try:
        results_sorted = sorted(
            results,
            key=lambda r: (r.get("position", 9999)
                           if isinstance(r.get("position"), int) else 9999)
        )
    except Exception:
        results_sorted = results

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

        results_tree.insert(
            "",
            tk.END,
            values=(
                position,
                driver_number,
                full_name,
                laps,
                points,
                status_str,
                gap,
                duration_str,
            ),
        )

    # Pit stop per Race/Sprint
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

        if pit_data:
            # Ordina per data (campo 'date', formato ISO 8601)
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

                lap_number = p.get("lap_number", "")
                pit_duration = p.get("pit_duration", "")
                if isinstance(pit_duration, (int, float)):
                    pit_duration_str = f"{pit_duration:.3f}"
                else:
                    pit_duration_str = ""

                pits_tree.insert(
                    "",
                    tk.END,
                    values=(full_name, lap_number, pit_duration_str),
                )

            status_var.set(
                f"Mostrati {len(results_sorted)} risultati (Race/Sprint) e "
                f"{len(pit_sorted)} pit stop. "
                "Seleziona un pilota per vedere grafici e tabella giri."
            )
        else:
            status_var.set(
                f"Mostrati {len(results_sorted)} risultati (Race/Sprint), "
                "ma nessun pit stop disponibile. "
                "Seleziona comunque un pilota per grafici e giri."
            )
    else:
        status_var.set(
            f"Mostrati {len(results_sorted)} risultati (tipo: {session_type}). "
            "Grafici, giri e pit stop sono disponibili solo per Race/Sprint."
        )

    # Meteo sempre, per ogni sessione selezionata, usando session_key
    update_weather_for_session(session_key)


def on_show_driver_plots_click():
    """
    Se Race/Sprint e un pilota è selezionato:
      - genera grafico distacchi (tab 1)
      - genera grafico stint gomme (tab 2)
      - popola tabella giri con tempi/settori/speeds/out laps
    """
    global gap_plot_canvas, stints_plot_canvas
    global gap_fig, gap_ax, gap_click_cid, gap_click_data

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
    # Svuota tabella giri (i pit stop + meteo restano perché sono dell'intera sessione)
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

        raw_gap_leader_vals = []
        raw_gap_prev_vals = []

        for idx, entry in enumerate(intervals, start=1):
            laps_idx.append(idx)
            gap_leader = entry.get("gap_to_leader", None)
            raw_gap_leader_vals.append(gap_leader)
            gaps_leader.append(gap_leader if isinstance(gap_leader, (int, float)) else None)
            interval_val = entry.get("interval", None)
            raw_gap_prev_vals.append(interval_val)
            gaps_prev.append(interval_val if isinstance(interval_val, (int, float)) else None)

        fig_gap = Figure(figsize=(6, 3.2))
        ax_gap = fig_gap.add_subplot(111)

        ax_gap.plot(laps_idx, gaps_leader, marker="o", label="Gap dal leader (s)")
        ax_gap.plot(laps_idx, gaps_prev, marker="o", linestyle="--", label="Gap dal pilota davanti (s)")

        ax_gap.set_xlabel("Giro (indice dei dati intervals)")
        ax_gap.set_ylabel("Distacco (secondi)")
        ax_gap.set_title(f"Distacchi giro per giro - {title_driver}")
        ax_gap.grid(True)
        ax_gap.legend()

        global gap_plot_canvas, gap_fig, gap_ax, gap_click_cid, gap_click_data
        gap_plot_canvas = FigureCanvasTkAgg(fig_gap, master=gap_tab_frame)
        gap_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        gap_plot_canvas.draw()

        # Memorizza dati & handler per il click
        gap_fig = fig_gap
        gap_ax = ax_gap
        gap_click_data = {
            "laps": laps_idx,
            "gap_leader": gaps_leader,
            "gap_prev": gaps_prev,
            "raw_gap_leader": raw_gap_leader_vals,
            "raw_gap_prev": raw_gap_prev_vals,
        }
        gap_click_cid = fig_gap.canvas.mpl_connect("button_press_event", on_gap_plot_click)
    else:
        messagebox.showinfo(
            "Info",
            "Nessun dato di intervallo (gap/interval) disponibile per questo pilota in questa sessione."
        )

    # --- Grafico stint gomme --- #
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

    if stints:
        compound_colors = {
            "SOFT": "red",
            "MEDIUM": "yellow",
            "HARD": "white",
            "INTERMEDIATE": "green",
            "WET": "blue",
        }

        try:
            stints_sorted = sorted(
                stints,
                key=lambda s: (
                    s.get("stint_number", 9999)
                    if isinstance(s.get("stint_number"), int)
                    else s.get("lap_start", 9999)
                )
            )
        except Exception:
            stints_sorted = stints

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
        ax_stints.set_title(f"Stint gomme - {title_driver}")
        ax_stints.grid(True, axis="x", linestyle="--", linewidth=0.5)

        if patches_used:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=col, edgecolor="black", label=comp)
                for comp, col in patches_used.items()
            ]
            ax_stints.legend(handles=legend_handles, title="Compound")

        global stints_plot_canvas
        stints_plot_canvas = FigureCanvasTkAgg(fig_stints, master=stints_tab_frame)
        stints_plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        stints_plot_canvas.draw()
    else:
        messagebox.showinfo(
            "Info",
            "Nessun dato stint gomme disponibile per questo pilota in questa sessione."
        )

    # --- Tabella giri --- #
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

    # Porta davanti la tab dei distacchi di default
    plots_notebook.select(gap_tab_frame)

    status_var.set(
        f"Grafici e tabella giri aggiornati per pilota {driver_number} ({title_driver}). "
        "La tabella pit stop mostra tutti i pit della sessione; la tab Meteo mostra l'evoluzione meteo (per sessione)."
    )


# --------------------- Costruzione GUI --------------------- #

root = tk.Tk()
root.title("Calendario, Risultati, Distacchi, Stint, Giri, Pit e Meteo F1 - OpenF1 API")
# Dimensione pensata per schermo 1920x1080
root.geometry("1400x900")
root.minsize(1200, 800)

# --- Frame input anno --- #
input_frame = ttk.Frame(root, padding=10)
input_frame.pack(fill="x")

ttk.Label(input_frame, text="Anno (es. 2023):").pack(side="left")
year_entry = ttk.Entry(input_frame, width=10)
year_entry.pack(side="left", padx=5)
year_entry.insert(0, "2023")

fetch_sessions_button = ttk.Button(
    input_frame,
    text="Recupera calendario",
    command=on_fetch_sessions_click
)
fetch_sessions_button.pack(side="left", padx=5)

# --- Frame principale --- #
main_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
main_frame.pack(fill="both", expand=True)

# Sessioni
ttk.Label(main_frame, text="Sessioni disponibili:", font=("", 10, "bold")).pack(anchor="w")

sessions_frame = ttk.Frame(main_frame)
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
    height=8
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
sessions_tree.column("datetime", width=160, anchor="center")
sessions_tree.column("session_name", width=170)
sessions_tree.column("session_type", width=80, anchor="center")
sessions_tree.column("circuit", width=150)
sessions_tree.column("location", width=150)
sessions_tree.column("country", width=130)
sessions_tree.column("session_key", width=0, stretch=False)
sessions_tree.column("meeting_key", width=0, stretch=False)

sessions_vsb = ttk.Scrollbar(
    sessions_frame,
    orient="vertical",
    command=sessions_tree.yview
)
sessions_tree.configure(yscrollcommand=sessions_vsb.set)

sessions_tree.grid(row=0, column=0, sticky="nsew")
sessions_vsb.grid(row=0, column=1, sticky="ns")

sessions_frame.rowconfigure(0, weight=1)
sessions_frame.columnconfigure(0, weight=1)

results_button_frame = ttk.Frame(main_frame)
results_button_frame.pack(fill="x", pady=(5, 0))

fetch_results_button = ttk.Button(
    results_button_frame,
    text="Mostra risultati della sessione selezionata",
    command=on_fetch_results_click
)
fetch_results_button.pack(side="left")

sessions_tree.bind("<Double-1>", on_fetch_results_click)

ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=8)

# Area centrale: risultati + (giri + pit) affiancati
center_frame = ttk.Frame(main_frame)
center_frame.pack(fill="both", expand=True)

# Risultati
left_frame = ttk.Frame(center_frame)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

ttk.Label(left_frame, text="Risultati sessione:", font=("", 10, "bold")).pack(anchor="w")

results_frame = ttk.Frame(left_frame)
results_frame.pack(fill="both", expand=True)

results_columns = (
    "position",
    "driver_number",
    "driver_name",
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
    height=10
)

results_tree.heading("position", text="Pos.")
results_tree.heading("driver_number", text="N°")
results_tree.heading("driver_name", text="Pilota")
results_tree.heading("laps", text="Giri")
results_tree.heading("points", text="Punti")
results_tree.heading("status", text="Stato")
results_tree.heading("gap", text="Gap leader (s)")
results_tree.heading("duration", text="Durata totale")

results_tree.column("position", width=45, anchor="center")
results_tree.column("driver_number", width=45, anchor="center")
results_tree.column("driver_name", width=180, anchor="w")
results_tree.column("laps", width=60, anchor="center")
results_tree.column("points", width=60, anchor="center")
results_tree.column("status", width=80, anchor="center")
results_tree.column("gap", width=110, anchor="center")
results_tree.column("duration", width=130, anchor="center")

results_vsb = ttk.Scrollbar(
    results_frame,
    orient="vertical",
    command=results_tree.yview
)
results_tree.configure(yscrollcommand=results_vsb.set)

results_tree.grid(row=0, column=0, sticky="nsew")
results_vsb.grid(row=0, column=1, sticky="ns")

results_frame.rowconfigure(0, weight=1)
results_frame.columnconfigure(0, weight=1)

# Colonna destra: giri + pit stop
right_frame = ttk.Frame(center_frame)
right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

# Tabella giri pilota selezionato
ttk.Label(
    right_frame,
    text="Dettaglio giri del pilota selezionato (Race/Sprint):",
    font=("", 10, "bold")
).pack(anchor="w")

laps_frame = ttk.Frame(right_frame)
laps_frame.pack(fill="both", expand=True)

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
    laps_frame,
    columns=laps_columns,
    show="headings",
    height=8
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

laps_tree.column("lap_number", width=50, anchor="center")
laps_tree.column("lap_time", width=90, anchor="center")
laps_tree.column("s1_time", width=90, anchor="center")
laps_tree.column("s2_time", width=90, anchor="center")
laps_tree.column("s3_time", width=90, anchor="center")
laps_tree.column("s1_speed", width=80, anchor="center")
laps_tree.column("s2_speed", width=80, anchor="center")
laps_tree.column("st_speed", width=80, anchor="center")
laps_tree.column("out_lap", width=60, anchor="center")

# Evidenzia out lap
laps_tree.tag_configure("outlap", background="#ffe4b5")  # arancio chiaro

laps_vsb = ttk.Scrollbar(
    laps_frame,
    orient="vertical",
    command=laps_tree.yview
)
laps_tree.configure(yscrollcommand=laps_vsb.set)

laps_tree.grid(row=0, column=0, sticky="nsew")
laps_vsb.grid(row=0, column=1, sticky="ns")

laps_frame.rowconfigure(0, weight=1)
laps_frame.columnconfigure(0, weight=1)

# Tabella pit stop della sessione
ttk.Label(
    right_frame,
    text="Pit stop della sessione (Race/Sprint):",
    font=("", 10, "bold")
).pack(anchor="w", pady=(5, 0))

pits_frame = ttk.Frame(right_frame)
pits_frame.pack(fill="both", expand=True)

pits_columns = (
    "pit_driver_name",
    "pit_lap_number",
    "pit_duration",
)

pits_tree = ttk.Treeview(
    pits_frame,
    columns=pits_columns,
    show="headings",
    height=6
)

pits_tree.heading("pit_driver_name", text="Pilota")
pits_tree.heading("pit_lap_number", text="Giro")
pits_tree.heading("pit_duration", text="Pit Time (s)")

pits_tree.column("pit_driver_name", width=160, anchor="w")
pits_tree.column("pit_lap_number", width=60, anchor="center")
pits_tree.column("pit_duration", width=80, anchor="center")

pits_vsb = ttk.Scrollbar(
    pits_frame,
    orient="vertical",
    command=pits_tree.yview
)
pits_tree.configure(yscrollcommand=pits_vsb.set)

pits_tree.grid(row=0, column=0, sticky="nsew")
pits_vsb.grid(row=0, column=1, sticky="ns")

pits_frame.rowconfigure(0, weight=1)
pits_frame.columnconfigure(0, weight=1)

# Pulsante grafici + giri
plot_button_frame = ttk.Frame(main_frame)
plot_button_frame.pack(fill="x", pady=(5, 0))

plot_button = ttk.Button(
    plot_button_frame,
    text="Mostra grafici e tabella giri per il pilota selezionato (Race/Sprint)",
    command=on_show_driver_plots_click
)
plot_button.pack(side="left")

ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=8)

# Notebook con tre tab: distacchi / stint gomme / meteo
plots_notebook = ttk.Notebook(main_frame)
plots_notebook.pack(fill="both", expand=True)

gap_tab_frame = ttk.Frame(plots_notebook)
stints_tab_frame = ttk.Frame(plots_notebook)
weather_tab_frame = ttk.Frame(plots_notebook)

plots_notebook.add(gap_tab_frame, text="Grafico distacchi")
plots_notebook.add(stints_tab_frame, text="Grafico stint gomme")
plots_notebook.add(weather_tab_frame, text="Meteo sessione")

# Contenuto tab Meteo: label testo + frame per grafico
weather_info_var = tk.StringVar(
    value="Meteo sessione: in attesa di una sessione selezionata."
)
weather_info_label = ttk.Label(weather_tab_frame, textvariable=weather_info_var, anchor="w")
weather_info_label.pack(fill="x", pady=(4, 2))

weather_plot_frame = ttk.Frame(weather_tab_frame)
weather_plot_frame.pack(fill="both", expand=True)

# Label per mostrare i valori del punto cliccato nel grafico distacchi
gap_point_info_var = tk.StringVar(
    value=(
        "Clicca un punto sul grafico distacchi per vedere gap_to_leader e interval (raw)."
    )
)
gap_point_info_label = ttk.Label(
    main_frame, textvariable=gap_point_info_var, anchor="w"
)
gap_point_info_label.pack(fill="x", pady=(4, 0))

# Barra di stato
status_var = tk.StringVar(
    value="Inserisci un anno, premi 'Recupera calendario', poi seleziona una sessione."
)
status_label = ttk.Label(root, textvariable=status_var, anchor="w", padding=5)
status_label.pack(fill="x")

# Avvio GUI
root.mainloop()
