
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import random
import math
import csv
import io
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# SHOTZS WEB VERSION
# Ported from: SHOTZS_v5_DigitalTwin(1).py
# Smart HVAC Occupancy-Aware Thermal Zoning System
# Software-only prototype
# ============================================================

st.set_page_config(
    page_title="SHOTZS — Smart HVAC Thermal Zoning",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- CONSTANTS --------------------
BG = "#F4F7FB"
CARD = "#FFFFFF"
NAVY = "#17324D"
TEXT = "#25364A"
MUTED = "#6B7C8F"
BORDER = "#D8E1EA"
BLUE = "#2F80ED"
GREEN = "#27AE60"
ORANGE = "#F2994A"
PURPLE = "#8E6CDB"
RED = "#EB5757"

ZONE_COLORS = ["#EEF6FF", "#F2F8F3", "#FFF6EA", "#F5F0FF"]
ZONE_ACCENTS = [BLUE, GREEN, ORANGE, PURPLE]

DEMAND_BG = {
    "UNOCCUPIED": "#F1F3F5",
    "MINIMUM": "#EEF6FF",
    "LOW": "#F2F8F3",
    "MEDIUM": "#FFF6EA",
    "HIGH": "#FDEEEE",
}

DEMAND_FG = {
    "UNOCCUPIED": MUTED,
    "MINIMUM": BLUE,
    "LOW": GREEN,
    "MEDIUM": ORANGE,
    "HIGH": RED,
}

COMFORT_TEMPERATURE = 24.0
COMFORT_HUMIDITY = 50.0
MAX_OCCUPANCY = 40
AC_CAPACITY_KW = 5.275
BASE_POWER_KW = 0.80
VARIABLE_POWER_KW = 2.70

PREDICTION_HORIZON_MIN = 5
PREDICTION_BLEND = 0.40
MAX_HISTORY = 30

DEFAULTS = [
    (31.0, 10, 84.0),
    (32.0, 27, 35.0),
    (24.0, 0, 88.0),
    (28.0, 15, 83.0),
]

# -------------------- PAGE CSS --------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
    }}
    .block-container {{
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}
    .shotzs-header {{
        background: {NAVY};
        border-radius: 12px;
        padding: 22px 28px;
        color: white;
        margin-bottom: 18px;
    }}
    .shotzs-header .brand {{
        font-size: 30px;
        font-weight: 800;
        letter-spacing: .5px;
    }}
    .shotzs-header .title {{
        font-size: 16px;
        font-weight: 700;
        margin-top: 2px;
    }}
    .shotzs-header .subtitle {{
        font-size: 12px;
        color: #BFD0E0;
        margin-top: 5px;
    }}
    .section-title {{
        color: {NAVY};
        font-size: 17px;
        font-weight: 800;
        margin: 20px 0 2px 0;
    }}
    .section-subtitle {{
        color: {MUTED};
        font-size: 12px;
        margin-bottom: 10px;
    }}
    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
    }}
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 105px;
    }}
    .metric-label {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 800;
    }}
    .metric-value {{
        color: {NAVY};
        font-size: 22px;
        font-weight: 800;
        margin-top: 5px;
    }}
    .zone-map {{
        background: #FBFCFE;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px;
    }}
    .hvac {{
        background: {NAVY};
        color: white;
        border-radius: 8px;
        text-align: center;
        padding: 10px;
        font-weight: 800;
        margin: 0 auto 18px auto;
        max-width: 260px;
    }}
    .hvac small {{
        display: block;
        color: #C9D8E7;
        font-weight: 500;
        margin-top: 3px;
    }}
    .zone-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        position: relative;
    }}
    .zone-box {{
        border: 2px dashed;
        border-radius: 9px;
        padding: 15px;
        min-height: 105px;
    }}
    .zone-name {{
        color: {NAVY};
        font-weight: 800;
        font-size: 13px;
    }}
    .zone-result {{
        font-weight: 800;
        margin-top: 8px;
        font-size: 13px;
    }}
    .airflow {{
        font-family: monospace;
        letter-spacing: 1px;
        margin-top: 5px;
    }}
    .map-note {{
        color: {MUTED};
        font-size: 10px;
        margin-top: 10px;
    }}
    .reason {{
        border-radius: 8px;
        padding: 11px 13px;
        border: 1px solid {BORDER};
        margin-bottom: 7px;
        font-size: 12px;
        color: {TEXT};
    }}
    .reason strong {{
        margin-right: 10px;
    }}
    .recommend {{
        background: {CARD};
        border-left: 5px solid {BLUE};
        border-top: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        border-radius: 8px;
        padding: 14px 16px;
        color: {NAVY};
    }}
    .footer {{
        background: {NAVY};
        color: #CBD5E1;
        border-radius: 8px;
        padding: 12px 15px;
        margin-top: 20px;
        font-size: 11px;
    }}
    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- CORE ENGINE --------------------
def clamp(value, low, high):
    return max(low, min(high, value))


def raw_thermal_score(temperature, occupancy, humidity):
    if occupancy <= 0:
        return 0.0

    temp_score = clamp((temperature - COMFORT_TEMPERATURE) / 6 * 100, 0, 100)
    occupancy_score = clamp(occupancy / MAX_OCCUPANCY * 100, 0, 100)
    humidity_score = clamp((humidity - COMFORT_HUMIDITY) / 30 * 100, 0, 100)

    return (
        0.50 * temp_score
        + 0.35 * occupancy_score
        + 0.15 * humidity_score
    )


@st.cache_resource
def train_predictive_model():
    random.seed(42)
    features = []
    targets = []

    for _ in range(3500):
        temperature = random.uniform(20, 35)
        occupancy = random.uniform(0, MAX_OCCUPANCY)
        humidity = random.uniform(30, 90)

        temp_change = random.uniform(-0.6, 0.6)
        occupancy_change = random.uniform(-6, 6)
        humidity_change = random.uniform(-4, 4)

        future_temperature = clamp(
            temperature + temp_change * PREDICTION_HORIZON_MIN, 15, 45
        )
        future_occupancy = clamp(
            occupancy + occupancy_change * 0.5, 0, MAX_OCCUPANCY
        )
        future_humidity = clamp(
            humidity + humidity_change * 0.5, 0, 100
        )

        future_score = raw_thermal_score(
            future_temperature, future_occupancy, future_humidity
        )
        future_score = clamp(
            future_score + random.gauss(0, 2.0), 0, 100
        )

        features.append([
            temperature, occupancy, humidity,
            temp_change, occupancy_change, humidity_change
        ])
        targets.append(future_score)

    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return model, r2


predictive_model, predictive_model_r2 = train_predictive_model()


def demand_from_score(score):
    if score <= 20:
        return "MINIMUM", 20
    if score <= 40:
        return "LOW", 40
    if score <= 70:
        return "MEDIUM", 70
    return "HIGH", 100


def calculate_zone(temperature, occupancy, humidity):
    temperature = clamp(float(temperature), 15, 45)
    occupancy = clamp(float(occupancy), 0, MAX_OCCUPANCY)
    humidity = clamp(float(humidity), 0, 100)

    if occupancy == 0:
        return "UNOCCUPIED", 0, 0.0

    score = raw_thermal_score(temperature, occupancy, humidity)
    demand, damper = demand_from_score(score)
    return demand, damper, score


def predict_future_score(
    temperature, occupancy, humidity,
    temp_change, occupancy_change, humidity_change
):
    if predictive_model is None or occupancy <= 0:
        return 0.0

    features = [[
        float(temperature), float(occupancy), float(humidity),
        float(temp_change), float(occupancy_change), float(humidity_change)
    ]]
    prediction = float(predictive_model.predict(features)[0])
    return clamp(prediction, 0, 100)


def zone_action(demand):
    actions = {
        "UNOCCUPIED": "Zone unoccupied → close damper",
        "MINIMUM": "Low demand → maintain minimum airflow",
        "LOW": "Low demand → provide controlled airflow",
        "MEDIUM": "Moderate demand → increase targeted airflow",
        "HIGH": "High demand → maximum targeted airflow",
    }
    return actions.get(demand, "Run analysis")


def decision_reason(temperature, occupancy, humidity, demand):
    reasons = []

    if occupancy == 0:
        return "Zone unoccupied → airflow minimized."

    if temperature >= 30:
        reasons.append("high temperature")
    elif temperature >= 26:
        reasons.append("elevated temperature")

    if occupancy >= 25:
        reasons.append("high occupancy")
    elif occupancy >= 10:
        reasons.append("moderate occupancy")

    if humidity >= 75:
        reasons.append("high humidity")
    elif humidity >= 60:
        reasons.append("elevated humidity")

    if not reasons:
        return "Conditions are close to the comfort target."

    return " + ".join(reasons).capitalize() + f" → {demand.lower()} cooling."


def analyze_zones(zone_inputs, previous_inputs):
    results = []

    for i, (temperature, occupancy, humidity) in enumerate(zone_inputs):
        _, _, current_score = calculate_zone(
            temperature, occupancy, humidity
        )

        previous = previous_inputs[i]
        if previous is None:
            dt = do = dh = 0.0
        else:
            dt = temperature - previous[0]
            do = occupancy - previous[1]
            dh = humidity - previous[2]

        predicted_score = predict_future_score(
            temperature, occupancy, humidity, dt, do, dh
        )

        control_score = (
            (1 - PREDICTION_BLEND) * current_score
            + PREDICTION_BLEND * predicted_score
        )

        if occupancy <= 0:
            demand, damper = "UNOCCUPIED", 0
            control_score = 0.0
        else:
            demand, damper = demand_from_score(control_score)

        results.append({
            "zone": f"ZONE {i+1}",
            "temperature": float(temperature),
            "occupancy": int(round(occupancy)),
            "humidity": float(humidity),
            "demand": demand,
            "damper": damper,
            "thermal_score": float(control_score),
            "predicted_score": float(predicted_score),
            "action": zone_action(demand),
            "reason": decision_reason(
                float(temperature), int(round(occupancy)),
                float(humidity), demand
            ),
        })

    avg_damper = sum(x["damper"] for x in results) / 4
    active = sum(1 for x in results if x["occupancy"] > 0)
    high = sum(1 for x in results if x["demand"] == "HIGH")
    medium = sum(1 for x in results if x["demand"] == "MEDIUM")
    avg_predicted = sum(x["predicted_score"] for x in results) / 4

    cooling_kw = AC_CAPACITY_KW * avg_damper / 100

    conventional_kw = BASE_POWER_KW + VARIABLE_POWER_KW
    shotzs_kw = (
        BASE_POWER_KW
        + VARIABLE_POWER_KW * avg_damper / 100
    )
    saving = max(
        0,
        min(
            100,
            ((conventional_kw - shotzs_kw) / conventional_kw) * 100
        )
    )

    if avg_damper == 0:
        hvac = "STANDBY"
    elif avg_damper <= 30:
        hvac = "LOW LOAD"
    elif avg_damper <= 60:
        hvac = "MEDIUM LOAD"
    elif avg_damper <= 80:
        hvac = "HIGH LOAD"
    else:
        hvac = "PEAK LOAD"

    if high:
        recommendation = (
            f"Priority cooling required in {high} zone(s). "
            "SHOTZS is using current conditions plus predicted near-future "
            "demand to direct airflow proactively."
        )
    elif medium:
        recommendation = (
            f"Controlled cooling is active. {medium} zone(s) require moderate "
            "airflow; the predictive layer is checking the next 5 minutes."
        )
    elif active:
        recommendation = (
            "Occupied zones have relatively low thermal demand. Maintain "
            "targeted minimum airflow while monitoring predicted demand."
        )
    else:
        recommendation = (
            "All zones are unoccupied. Minimize zone airflow and HVAC operation."
        )

    system = {
        "hvac": hvac,
        "avg_damper": avg_damper,
        "cooling_kw": cooling_kw,
        "active_zones": active,
        "conventional_kw": conventional_kw,
        "shotzs_kw": shotzs_kw,
        "saving": saving,
        "avg_predicted": avg_predicted,
        "recommendation": recommendation,
    }

    return results, system


# -------------------- SESSION STATE --------------------
if "zone_inputs" not in st.session_state:
    st.session_state.zone_inputs = [list(x) for x in DEFAULTS]
if "previous_inputs" not in st.session_state:
    st.session_state.previous_inputs = [None] * 4
if "results" not in st.session_state:
    st.session_state.results = None
if "system" not in st.session_state:
    st.session_state.system = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = True
if "twin" not in st.session_state:
    st.session_state.twin = {
        "running": False,
        "minute": 0,
        "temps": [27.0, 25.5, 26.5, 24.8],
        "hum": [58.0, 52.0, 60.0, 48.0],
        "occ": [8, 20, 14, 4],
        "energy_shotzs": 0.0,
        "energy_conventional": 0.0,
        "history_t": [[], [], [], []],
        "history_avg": [],
        "last_inputs": [None, None, None, None],
        "last_controls": [(0, 0)] * 4,
        "outdoor": 31.0,
    }

# -------------------- HEADER --------------------
st.markdown(
    """
    <div class="shotzs-header">
        <div class="brand">SHOTZS</div>
        <div class="title">Smart HVAC Occupancy-Aware Thermal Zoning System</div>
        <div class="subtitle">Software prototype • Single room • Four thermal zones</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- THERMAL ZONE MAP --------------------
st.markdown('<div class="section-title">❯ THERMAL ZONE MAP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Live room view — one physical room with four virtual thermal zones</div>',
    unsafe_allow_html=True
)

map_results = st.session_state.results

if map_results:
    # IMPORTANT: Use Streamlit's HTML component for this visualization.
    # Do NOT use st.markdown for the room HTML; Streamlit Cloud may escape
    # SVG/embedded markup and display the source code instead.
    zone_cards = []
    for i, r in enumerate(map_results):
        demand = float(r["thermal_score"])
        pred = float(r["predicted_score"])
        damper = float(r["damper"])
        demand_name = r["demand"]
        accent = ZONE_ACCENTS[i]
        bg = DEMAND_BG[demand_name]
        fg = DEMAND_FG[demand_name]

        zone_cards.append(f"""
        <div class="zone zone{i+1}" style="background:{bg};border-color:{accent};">
            <div class="zone-title" style="color:{NAVY};">ZONE {i+1}</div>
            <div class="zone-data">{r["temperature"]:.1f}°C &nbsp;|&nbsp; {r["occupancy"]} people &nbsp;|&nbsp; {r["humidity"]:.0f}% RH</div>
            <div class="zone-demand" style="color:{fg};">
                Thermal demand: {demand:.0f}% &nbsp;|&nbsp; Predicted: {pred:.0f}%
            </div>
            <div class="zone-airflow" style="color:{accent};">
                AIRFLOW ALLOCATION &nbsp;{damper:.0f}%
            </div>
        </div>
        """)

    # CSS arrows are used instead of SVG so the browser receives ordinary HTML.
    # Their length/thickness/opacity are controlled by the calculated damper.
    arrow_angles = [-153, -27, 153, 27]
    arrows = []
    for i, r in enumerate(map_results):
        damper = float(r["damper"])
        accent = ZONE_ACCENTS[i]
        width = max(2.5, 2.5 + damper / 14)
        opacity = max(0.18, damper / 100)
        length = min(39, 25 + damper * 0.14)
        arrows.append(f"""
        <div class="air-arrow a{i+1}"
             style="--accent:{accent};--width:{width:.1f}px;--opacity:{opacity:.2f};
                    --length:{length:.1f}%;--angle:{arrow_angles[i]}deg;">
            <span></span>
        </div>
        """)

    room_html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing:border-box; }}
        html, body {{ margin:0; padding:0; background:transparent; }}
        body {{ font-family:Arial, sans-serif; }}
        .room {{
            position:relative;
            width:100%;
            height:570px;
            background:#FBFCFE;
            border:3px solid {NAVY};
            border-radius:18px;
            overflow:hidden;
        }}
        .twin-title {{
            text-align:center;
            padding-top:15px;
            color:{NAVY};
            font-size:20px;
            font-weight:700;
        }}
        .twin-sub {{
            text-align:center;
            margin-top:5px;
            color:{MUTED};
            font-size:11px;
        }}
        .ac {{
            position:absolute;
            left:50%;
            top:68px;
            transform:translateX(-50%);
            width:190px;
            height:62px;
            background:{NAVY};
            color:white;
            border-radius:13px;
            text-align:center;
            padding-top:11px;
            z-index:5;
            box-shadow:0 4px 10px rgba(0,0,0,.12);
        }}
        .ac-main {{ font-size:16px; font-weight:700; }}
        .ac-sub {{ font-size:10px; color:#C9D8E7; margin-top:5px; }}
        .zone {{
            position:absolute;
            width:44%;
            height:27%;
            border:3px dashed;
            border-radius:12px;
            padding:18px 20px;
            z-index:3;
        }}
        .zone1 {{ left:3%; top:34%; }}
        .zone2 {{ right:3%; top:34%; }}
        .zone3 {{ left:3%; bottom:5%; }}
        .zone4 {{ right:3%; bottom:5%; }}
        .zone-title {{ font-size:16px; font-weight:700; }}
        .zone-data {{ margin-top:8px; font-size:13px; color:{TEXT}; }}
        .zone-demand {{ margin-top:9px; font-size:12px; font-weight:700; }}
        .zone-airflow {{
            position:absolute;
            right:18px;
            bottom:15px;
            font-size:13px;
            font-weight:700;
        }}
        .air-arrow {{
            position:absolute;
            left:50%;
            top:27%;
            height:var(--width);
            width:var(--length);
            background:var(--accent);
            opacity:var(--opacity);
            transform-origin:0 50%;
            transform:rotate(var(--angle));
            border-radius:999px;
            z-index:2;
            animation:pulse 1.2s ease-in-out infinite;
        }}
        .air-arrow span {{
            position:absolute;
            right:-1px;
            top:50%;
            transform:translateY(-50%);
            width:0;
            height:0;
            border-top:8px solid transparent;
            border-bottom:8px solid transparent;
            border-left:14px solid var(--accent);
        }}
        @keyframes pulse {{
            0%,100% {{ filter:brightness(1); }}
            50% {{ filter:brightness(1.25); }}
        }}
        .room-note {{
            position:absolute;
            left:0;
            right:0;
            bottom:7px;
            text-align:center;
            color:{MUTED};
            font-size:11px;
            z-index:4;
        }}
        @media (max-width:700px) {{
            .room {{ height:520px; }}
            .zone {{ height:28%; padding:11px 12px; }}
            .zone-title {{ font-size:13px; }}
            .zone-data, .zone-demand {{ font-size:10px; }}
            .zone-airflow {{ font-size:10px; right:10px; bottom:10px; }}
            .ac {{ width:145px; height:55px; top:65px; }}
            .ac-main {{ font-size:13px; }}
            .ac-sub {{ font-size:8px; }}
        }}
    </style>
    </head>
    <body>
      <div class="room">
        <div class="twin-title">VIRTUAL ROOM DIGITAL TWIN</div>
        <div class="twin-sub">No physical partitions • Cooling dynamically rebalanced by SHOTZS</div>
        <div class="ac">
          <div class="ac-main">CENTRAL AC</div>
          <div class="ac-sub">CENTRAL AIRFLOW CONTROL</div>
        </div>
        {''.join(arrows)}
        {''.join(zone_cards)}
        <div class="room-note">
          Virtual zones represent thermal conditions inside one open room — no physical walls.
        </div>
      </div>
    </body>
    </html>
    """

    components.html(room_html, height=590, scrolling=False)

else:
    st.markdown(
        f'<div class="zone-map"><div class="hvac">CENTRAL HVAC<small>Supply Air</small></div>'
        f'<div class="zone-grid">' +
        "".join(
            f'<div class="zone-box" style="background:{ZONE_COLORS[i]};border-color:{ZONE_ACCENTS[i]}">'
            f'<div class="zone-name">ZONE {i+1}</div>'
            f'<div style="color:{MUTED};margin-top:18px;font-size:12px">Run analysis to activate the room twin</div>'
            f'</div>' for i in range(4)
        ) +
        '</div><div class="map-note">The four zones are virtual thermal zones inside one physical room.</div></div>',
        unsafe_allow_html=True
    )

# -------------------- SENSOR INPUTS --------------------
st.markdown('<div class="section-title">❯ ZONE SENSOR INPUTS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Adjust simulated sensor readings for each thermal zone</div>',
    unsafe_allow_html=True
)

input_cols = st.columns(4)
new_inputs = []

for i in range(4):
    with input_cols[i]:
        st.markdown(
            f'<div style="font-weight:800;color:{ZONE_ACCENTS[i]};font-size:14px;margin-bottom:6px">ZONE {i+1}</div>',
            unsafe_allow_html=True
        )
        temp = st.slider(
            "Temperature (°C)",
            min_value=15.0, max_value=45.0,
            value=float(st.session_state.zone_inputs[i][0]),
            step=0.1, key=f"temp_{i}"
        )
        occ = st.slider(
            "Occupancy (persons)",
            min_value=0, max_value=40,
            value=int(st.session_state.zone_inputs[i][1]),
            step=1, key=f"occ_{i}"
        )
        hum = st.slider(
            "Humidity (% RH)",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state.zone_inputs[i][2]),
            step=0.5, key=f"hum_{i}"
        )
        new_inputs.append([temp, occ, hum])

st.session_state.zone_inputs = new_inputs

# -------------------- ACTIONS --------------------
a1, a2, a3, a4, a5 = st.columns(5)
with a1:
    if st.button("● LIVE MODE: ON" if st.session_state.live_mode else "○ LIVE MODE: OFF", use_container_width=True):
        st.session_state.live_mode = not st.session_state.live_mode
        st.rerun()

with a2:
    run_clicked = st.button("▶ RUN SHOTZS ANALYSIS", type="primary", use_container_width=True)

with a3:
    reset_clicked = st.button("↻ RESET OUTPUTS", use_container_width=True)

with a4:
    twin_jump = st.button("◈ DIGITAL TWIN", use_container_width=True)

with a5:
    export_clicked = st.button("⇩ EXPORT RESULTS", use_container_width=True)

if reset_clicked:
    st.session_state.results = None
    st.session_state.system = None
    st.session_state.history = []
    st.session_state.previous_inputs = [None] * 4
    st.session_state.last_analysis = None
    st.rerun()

if run_clicked or (st.session_state.live_mode and st.session_state.results is None):
    results, system = analyze_zones(
        st.session_state.zone_inputs,
        st.session_state.previous_inputs
    )
    st.session_state.results = results
    st.session_state.system = system
    st.session_state.previous_inputs = [
        (x[0], x[1], x[2]) for x in st.session_state.zone_inputs
    ]
    st.session_state.history.append(system["avg_damper"])
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    st.session_state.last_analysis = datetime.now()

# -------------------- OUTPUT TABLE --------------------
st.markdown('<div class="section-title">❯ ZONE CONTROL OUTPUT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Controller decision for each zone</div>',
    unsafe_allow_html=True
)

if st.session_state.results:
    table_rows = []
    for r in st.session_state.results:
        table_rows.append({
            "ZONE": r["zone"],
            "COOLING DEMAND": r["demand"],
            "DAMPER": f'{r["damper"]}%',
            "THERMAL SCORE": f'{r["thermal_score"]:.0f}%',
            "5-MIN PREDICTED": f'{r["predicted_score"]:.0f}%',
            "RECOMMENDED ACTION": r["action"],
        })
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "THERMAL SCORE": st.column_config.TextColumn(width="small"),
            "5-MIN PREDICTED": st.column_config.TextColumn(width="small"),
            "RECOMMENDED ACTION": st.column_config.TextColumn(width="large"),
        },
    )
else:
    st.info("Adjust the sensor values and run SHOTZS analysis.")

# -------------------- DECISION LOG --------------------
st.markdown('<div class="section-title">❯ INTELLIGENT DECISION LOG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Why SHOTZS selected the current cooling response</div>',
    unsafe_allow_html=True
)

if st.session_state.results:
    for i, r in enumerate(st.session_state.results):
        st.markdown(
            f'<div class="reason" style="background:{DEMAND_BG[r["demand"]]};">'
            f'<strong style="color:{ZONE_ACCENTS[i]}">ZONE {i+1}</strong>'
            f'{r["reason"]} Predicted in {PREDICTION_HORIZON_MIN} min: '
            f'{r["predicted_score"]:.0f}% thermal demand.</div>',
            unsafe_allow_html=True
        )
else:
    st.info("Waiting for analysis...")

# -------------------- PERFORMANCE --------------------
st.markdown('<div class="section-title">❯ SYSTEM PERFORMANCE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Central HVAC response and estimated resource utilization</div>',
    unsafe_allow_html=True
)

if st.session_state.system:
    s = st.session_state.system
    metrics = st.columns(4)
    metric_data = [
        ("CENTRAL HVAC", s["hvac"]),
        ("AVG. DAMPER", f'{s["avg_damper"]:.1f}%'),
        ("COOLING DEMAND", f'{s["cooling_kw"]:.2f} kW'),
        ("ACTIVE ZONES", f'{s["active_zones"]} / 4'),
    ]
    for col, (label, value) in zip(metrics, metric_data):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True
            )
else:
    st.write("")
    st.info("Run analysis to display system performance.")

# -------------------- ENERGY --------------------
st.markdown('<div class="section-title">❯ ENERGY ESTIMATION</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Prototype control-efficiency comparison from the source model</div>',
    unsafe_allow_html=True
)

if st.session_state.system:
    s = st.session_state.system
    e1, e2, e3 = st.columns(3)
    for col, label, value in [
        (e1, "CONVENTIONAL HVAC", f'{s["conventional_kw"]:.2f} kW'),
        (e2, "SHOTZS POWER", f'{s["shotzs_kw"]:.2f} kW'),
        (e3, "ESTIMATED SAVING", f'{s["saving"]:.1f}%'),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True
            )

# -------------------- RECOMMENDATION --------------------
if st.session_state.system:
    s = st.session_state.system
    timestamp = (
        st.session_state.last_analysis.strftime("%d-%m-%Y  %I:%M:%S %p")
        if st.session_state.last_analysis else "--"
    )
    st.markdown(
        f'<div class="recommend"><b style="color:{BLUE}">💡 SYSTEM RECOMMENDATION</b>'
        f'<div style="margin-top:5px;font-weight:700">{s["recommendation"]}</div>'
        f'<div style="color:{MUTED};font-size:10px;margin-top:5px">'
        f'Last analysis: {timestamp} &nbsp;|&nbsp; ML R²: {predictive_model_r2:.2f}</div></div>',
        unsafe_allow_html=True
    )

# -------------------- HISTORY --------------------
st.markdown('<div class="section-title">❯ LIVE SYSTEM HISTORY</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Recent controller decisions and estimated average damper position</div>',
    unsafe_allow_html=True
)

if st.session_state.history:
    hist = pd.DataFrame({
        "Analysis": range(1, len(st.session_state.history) + 1),
        "Average Damper (%)": st.session_state.history
    }).set_index("Analysis")
    st.line_chart(hist, y="Average Damper (%)", height=240)
else:
    st.info("Run SHOTZS analysis to begin recording system history.")

# -------------------- DIGITAL TWIN --------------------
st.markdown('<div class="section-title">❯ SHOTZS DIGITAL TWIN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Virtual single room • Four thermal zones • No physical partitions</div>',
    unsafe_allow_html=True
)

with st.expander("Open Virtual Room Digital Twin", expanded=True):
    twin = st.session_state.twin

    def reset_twin_state():
        st.session_state.twin = {
            "running": False,
            "minute": 0,
            "temps": [27.0, 25.5, 26.5, 24.8],
            "hum": [58.0, 52.0, 60.0, 48.0],
            "occ": [8, 20, 14, 4],
            "energy_shotzs": 0.0,
            "energy_conventional": 0.0,
            "history_t": [[], [], [], []],
            "history_avg": [],
            "last_inputs": [None, None, None, None],
            "last_controls": [(0, 0)] * 4,
            "outdoor": 31.0,
        }

    def twin_conditions(minute):
        outdoor = 31.0 + 3.0 * math.sin((minute - 12) / 15.0)
        occ = [
            6 + 18 * max(0, math.sin((minute - 4) / 12.0)),
            12 + 22 * max(0, math.sin((minute + 2) / 14.0)),
            3 + 12 * max(0, math.sin((minute - 14) / 16.0)),
            2 + 7 * max(0, math.sin((minute + 6) / 11.0)),
        ]
        if 18 <= minute <= 26:
            occ[0] += 12
        if 36 <= minute <= 46:
            occ[1] += 10
        occ = [int(max(0, min(40, round(v)))) for v in occ]

        rh = [
            48 + 0.60 * occ[0] + 4 * math.sin(minute / 11),
            46 + 0.65 * occ[1] + 4 * math.sin((minute + 3) / 12),
            48 + 0.55 * occ[2] + 4 * math.sin((minute + 6) / 13),
            45 + 0.60 * occ[3] + 3 * math.sin((minute + 2) / 10),
        ]
        rh = [round(max(35, min(85, x)), 1) for x in rh]
        return outdoor, occ, rh

    def twin_step():
        if twin["minute"] >= 60:
            return

        twin["minute"] += 1
        outdoor, occ, rh = twin_conditions(twin["minute"])
        twin["outdoor"] = outdoor
        twin["occ"] = occ
        twin["hum"] = rh

        controls = []
        scores = []
        predicted_scores = []

        for i in range(4):
            prev = twin["last_inputs"][i]
            current = (twin["temps"][i], twin["occ"][i], twin["hum"][i])
            twin["last_inputs"][i] = current

            if prev is None:
                dt = do = dh = 0.0
            else:
                dt = current[0] - prev[0]
                do = current[1] - prev[1]
                dh = current[2] - prev[2]

            current_score = raw_thermal_score(*current)
            predicted = predict_future_score(
                current[0], current[1], current[2], dt, do, dh
            )
            control = 0.60 * current_score + 0.40 * predicted
            if current[1] <= 0:
                control = 0.0
            control = clamp(control, 0, 100)

            controls.append((control, predicted))
            scores.append(control)
            predicted_scores.append(predicted)

        total_score = sum(scores)
        allocation = [s / total_score for s in scores] if total_score > 0 else [0.25] * 4
        avg_score = sum(scores) / 4

        total_cooling = (
            AC_CAPACITY_KW * clamp(avg_score / 100, 0.10, 1.0)
            if avg_score > 0 else 0.0
        )

        new_temps = []
        for i in range(4):
            envelope_gain = 0.045 * (outdoor - twin["temps"][i])
            people_gain = 0.012 * twin["occ"][i]
            cooling = total_cooling * allocation[i]
            cooling_effect = 0.16 * cooling
            delta = envelope_gain + people_gain - cooling_effect
            new_temps.append(clamp(twin["temps"][i] + delta, 18, 40))

        twin["temps"] = [round(x, 2) for x in new_temps]

        shotzs_power = BASE_POWER_KW + VARIABLE_POWER_KW * clamp(avg_score / 100, 0, 1)
        avg_temp = sum(twin["temps"]) / 4
        conventional_fraction = clamp((avg_temp - 23.5) / 5.0, 0.20, 1.0)
        conventional_power = BASE_POWER_KW + VARIABLE_POWER_KW * conventional_fraction

        twin["energy_shotzs"] += shotzs_power / 60.0
        twin["energy_conventional"] += conventional_power / 60.0

        for i in range(4):
            twin["history_t"][i].append(twin["temps"][i])
        twin["history_avg"].append(avg_score)
        twin["last_controls"] = controls

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ ADVANCE 1 MINUTE", use_container_width=True):
            twin_step()
            st.rerun()
    with c2:
        if st.button("▶ RUN FULL 60-MINUTE SIMULATION", type="primary", use_container_width=True):
            while twin["minute"] < 60:
                twin_step()
            st.rerun()
    with c3:
        if st.button("↻ RESET DIGITAL TWIN", use_container_width=True):
            reset_twin_state()
            st.rerun()

    difference = 0.0
    if twin["energy_conventional"] > 0:
        difference = (
            1 - twin["energy_shotzs"] / twin["energy_conventional"]
        ) * 100

    st.markdown(
        f"**Status:** {'SIMULATION COMPLETE' if twin['minute'] >= 60 else 'READY / STEPPED'}  "
        f"| **Simulated time:** {twin['minute']:02d}:00 / 60:00  "
        f"| **Outdoor:** {twin['outdoor']:.1f} °C"
    )
    st.markdown(
        f"**SHOTZS energy:** {twin['energy_shotzs']:.3f} kWh  "
        f"| **Conventional:** {twin['energy_conventional']:.3f} kWh  "
        f"| **Estimated difference:** {difference:.1f}%"
    )

    # -------------------- DYNAMIC ROOM DIGITAL TWIN --------------------
    # Robust HTML/CSS renderer: avoids SVG escaping issues on Streamlit Cloud.
    zones_html = []
    for i in range(4):
        control, pred = twin["last_controls"][i]
        temp, occ, hum = twin["temps"][i], twin["occ"][i], twin["hum"][i]
        if occ <= 0: label, fg = "UNOCCUPIED", MUTED
        elif control <= 20: label, fg = "MINIMUM", BLUE
        elif control <= 40: label, fg = "LOW", GREEN
        elif control <= 70: label, fg = "MEDIUM", ORANGE
        else: label, fg = "HIGH", RED
        bg, accent = DEMAND_BG[label], ZONE_ACCENTS[i]
        zones_html.append(f"""<div class=\"zone z{i+1}\" style=\"background:{bg};border-color:{accent};\">
          <div class=\"zt\">ZONE {i+1}</div>
          <div class=\"zs\">{temp:.1f}°C &nbsp;|&nbsp; {occ:.0f} people &nbsp;|&nbsp; {hum:.0f}% RH</div>
          <div class=\"zd\" style=\"color:{fg}\">Thermal demand: {control:.0f}% &nbsp;|&nbsp; Predicted: {pred:.0f}%</div>
          <div class=\"za\" style=\"color:{accent}\">AIRFLOW ALLOCATION &nbsp;{control:.0f}%</div>
        </div>""")
    arrow_specs = [(30,35,-28),(58,35,28),(30,70,28),(58,70,-28)]
    arrows=[]
    for i,(left,top,angle) in enumerate(arrow_specs):
        control=twin["last_controls"][i][0]; accent=ZONE_ACCENTS[i]
        width=max(3.0,3.0+control/14.0); opacity=max(0.18,control/100.0); length=95+2.0*control
        arrows.append(f"<div class=\"arrow\" style=\"left:{left}%;top:{top}%;width:{length:.0f}px;height:{width:.1f}px;background:{accent};opacity:{opacity:.2f};transform:rotate({angle}deg)\"><i style=\"border-left-color:{accent}\"></i></div>")
    twin_html = f"""<!doctype html><html><head><style>
      *{{box-sizing:border-box}} body{{margin:0;background:transparent;font-family:Arial,sans-serif}}
      .room{{position:relative;width:100%;height:555px;background:#fff;border:3px solid {NAVY};border-radius:18px;overflow:hidden}}
      .title{{text-align:center;padding-top:13px;color:{NAVY};font-size:22px;font-weight:800}}
      .sub{{text-align:center;color:{MUTED};font-size:11px;margin-top:4px}}
      .ac{{position:absolute;z-index:10;left:50%;top:60px;transform:translateX(-50%);width:190px;padding:10px 5px;text-align:center;background:{NAVY};color:white;border-radius:12px}}
      .ac b{{display:block;font-size:17px}} .ac small{{display:block;color:#C9D8E7;font-size:10px;margin-top:3px}}
      .zone{{position:absolute;width:44%;height:28%;border:3px dashed;border-radius:14px;padding:16px 20px;z-index:4}}
      .z1{{left:3.5%;top:34%}} .z2{{right:3.5%;top:34%}} .z3{{left:3.5%;top:64%}} .z4{{right:3.5%;top:64%}}
      .zt{{font-size:17px;font-weight:800;color:{NAVY}}}.zs{{font-size:12px;font-weight:700;color:#25364A;margin-top:7px}}
      .zd{{font-size:12px;font-weight:800;margin-top:9px}}.za{{position:absolute;right:18px;bottom:14px;font-size:12px;font-weight:800}}
      .arrow{{position:absolute;z-index:3;transform-origin:left center;border-radius:99px;animation:pulse 1.2s ease-in-out infinite}}
      .arrow i{{position:absolute;right:-2px;top:50%;transform:translateY(-50%);width:0;height:0;border-top:9px solid transparent;border-bottom:9px solid transparent;border-left:16px solid}}
      @keyframes pulse{{0%,100%{{filter:brightness(.9)}}50%{{filter:brightness(1.3)}}}}
      .foot{{position:absolute;left:0;bottom:8px;width:100%;text-align:center;color:{MUTED};font-size:10px}}
    </style></head><body><div class="room"><div class="title">SHOTZS DIGITAL TWIN</div>
      <div class="sub">Virtual single room • Four thermal zones • No physical partitions • Cooling dynamically rebalanced</div>
      <div class="ac"><b>CENTRAL AC</b><small>CENTRAL AIRFLOW CONTROL</small></div>
      {''.join(arrows)}{''.join(zones_html)}
      <div class="foot">Outdoor: {twin["outdoor"]:.1f}°C &nbsp;•&nbsp; Simulated time: {twin["minute"]:02d}:00 / 60:00 &nbsp;•&nbsp; Airflow continuously rebalanced</div>
    </div></body></html>"""
    components.html(twin_html, height=575, scrolling=False)

    # Compact live values below the visual.
    twin_rows = []
    for i in range(4):
        control, pred = twin["last_controls"][i]
        twin_rows.append({
            "ZONE": f"ZONE {i+1}",
            "TEMP": f'{twin["temps"][i]:.1f}°C',
            "OCC": twin["occ"][i],
            "RH": f'{twin["hum"][i]:.0f}%',
            "DEMAND": f'{control:.0f}%',
            "PREDICTED": f'{pred:.0f}%',
        })
    st.dataframe(pd.DataFrame(twin_rows), use_container_width=True, hide_index=True)

    if any(twin["history_t"][0]):
        chart_data = pd.DataFrame({
            "Zone 1": twin["history_t"][0],
            "Zone 2": twin["history_t"][1],
            "Zone 3": twin["history_t"][2],
            "Zone 4": twin["history_t"][3],
        })
        st.line_chart(chart_data, height=220)

    hottest = max(range(4), key=lambda i: twin["temps"][i])
    avg_pred = (
        sum(x[1] for x in twin["last_controls"]) / 4
        if twin["last_controls"] else 0
    )
    st.info(
        f'Outdoor temperature: {twin["outdoor"]:.1f}°C • '
        f'Hottest zone: Zone {hottest+1} ({twin["temps"][hottest]:.1f}°C) • '
        f'Average predicted demand: {avg_pred:.0f}% • '
        'The visual airflow paths update with the simulated control decision.'
    )

# -------------------- EXPORT --------------------
if export_clicked:
    if not st.session_state.results:
        st.warning("Please run SHOTZS analysis before exporting results.")
    else:
        rows = [
            ["SHOTZS SYSTEM REPORT"],
            ["Generated", datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")],
            [],
            [
                "Zone", "Temperature °C", "Occupancy", "Humidity %",
                "Demand", "Damper %", "Thermal Score", "Decision Reason"
            ],
        ]

        for r in st.session_state.results:
            rows.append([
                r["zone"],
                f'{r["temperature"]:.1f}',
                r["occupancy"],
                f'{r["humidity"]:.1f}',
                r["demand"],
                r["damper"],
                f'{r["thermal_score"]:.1f}',
                r["reason"],
            ])

        s = st.session_state.system
        rows.extend([
            [],
            ["System Metric", "Value"],
            ["Central HVAC", s["hvac"]],
            ["Average Damper", f'{s["avg_damper"]:.1f}%'],
            ["Cooling Demand", f'{s["cooling_kw"]:.2f} kW'],
            ["Active Zones", f'{s["active_zones"]} / 4'],
            ["Conventional HVAC", f'{s["conventional_kw"]:.2f} kW'],
            ["SHOTZS Power", f'{s["shotzs_kw"]:.2f} kW'],
            ["Estimated Saving", f'{s["saving"]:.1f}%'],
        ])

        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        st.download_button(
            "Download SHOTZS CSV report",
            data=buf.getvalue().encode("utf-8-sig"),
            file_name="SHOTZS_system_report.csv",
            mime="text/csv",
        )

# -------------------- FOOTER --------------------
st.markdown(
    '<div class="footer"><b>SHOTZS</b> &nbsp; Smart HVAC Occupancy-Aware Thermal Zoning System'
    '<span style="float:right">Estimated software model • Web version</span></div>',
    unsafe_allow_html=True
)
