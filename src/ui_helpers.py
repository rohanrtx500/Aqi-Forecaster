"""
Streamlit rendering helpers: metric cards, badges, rule-based "possible
factors", and the air-quality alert banner. No model logic here.
"""
import pandas as pd
import streamlit as st

from src.aqi_utils import category_color

CARD_CSS = """
<style>
/* Main App Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 40%, #0f172a 100%) !important;
    color: #f8fafc;
}

/* Hide Default Sidebar for Clean Web App Feel */
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Top Navbar Styling */
.top-navbar-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 24px;
}
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 8px;
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(16px);
    padding: 6px 10px;
    border-radius: 30px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 6px 18px !important;
    border-radius: 20px !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-top: 3px solid #3b82f6 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    text-align: left;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(59, 130, 246, 0.2);
}
.metric-card .label {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(90deg, #ffffff, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .sub {
    font-size: 13px;
    color: #64748b;
    margin-top: 6px;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: #0f172a;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* Factor rows */
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    margin-bottom: 6px;
    background: rgba(30, 41, 59, 0.4);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 14px;
}

/* Alert Box */
.alert-box {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.85) 0%, rgba(180, 83, 9, 0.85) 100%);
    border: 1px solid #f59e0b;
    color: #fef3c7;
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
}

/* Tables */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
th {
    background: rgba(30, 41, 59, 0.9) !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
}
td {
    background: rgba(15, 23, 42, 0.5) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 10px 16px !important;
    color: #e2e8f0 !important;
}
tr:hover td {
    background: rgba(51, 65, 85, 0.5) !important;
}
/* Persona Advisory Cards */
.advisory-card {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.advisory-card .title {
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.advisory-card .status {
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
}
.advisory-card .desc {
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 1.4;
}

/* AI Briefing Box */
.ai-briefing-box {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 27, 75, 0.8) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-left: 4px solid #8b5cf6;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.12);
}
.ai-briefing-box .header {
    font-size: 15px;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ai-briefing-box .content {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Best Time Card */
.best-time-box {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(4, 120, 87, 0.5) 100%);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.15);
}

/* Pollutant Glassmorphism Cards */
.pollutant-card {
    background: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 16px 16px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.pollutant-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56, 189, 248, 0.4) !important;
}
.pollutant-card .pollutant-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.pollutant-card .pollutant-sym {
    font-size: 15px;
    font-weight: 700;
    color: #f8fafc;
}
.pollutant-card .pollutant-name {
    font-size: 11.5px;
    color: #94a3b8;
    margin-bottom: 8px;
}
.pollutant-card .pollutant-val {
    font-size: 22px;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1.1;
}
.pollutant-card .pollutant-limit {
    font-size: 11px;
    color: #64748b;
    margin-top: 8px;
}
.pollutant-bar-bg {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    height: 5px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.pollutant-bar-fill {
    height: 100%;
    border-radius: 6px;
}
</style>
"""


def inject_css():
    st.markdown(CARD_CSS, unsafe_allow_html=True)


def badge_html(category: str) -> str:
    return f'<span class="badge" style="background:{category_color(category)}">{category}</span>'


def metric_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_possible_factors(latest_row: pd.Series):
    """Simple rule-based interpretation from already-available engineered features."""
    factors = []

    if "pm25" in latest_row and "pm25_lag_6" in latest_row:
        trend = latest_row["pm25"] - latest_row["pm25_lag_6"]
        factors.append(("PM2.5 Trend", "Rising" if trend > 2 else "Falling" if trend < -2 else "Stable"))

    if "wind_speed" in latest_row:
        factors.append(("Wind Ventilation", "Low" if latest_row["wind_speed"] < 2 else "High" if latest_row["wind_speed"] > 6 else "Moderate"))

    if "humidity" in latest_row:
        factors.append(("Relative Humidity", "High" if latest_row["humidity"] > 70 else "Low" if latest_row["humidity"] < 40 else "Moderate"))

    if "rainfall" in latest_row:
        factors.append(("Precipitation Scrubbing", "Present" if latest_row["rainfall"] > 0.1 else "None"))

    arrow = {"Rising": "\u2191", "High": "\u2191", "Falling": "\u2193", "Low": "\u2193",
             "Stable": "\u2014", "Moderate": "\u2014", "None": "\u2014", "Present": "\u2193"}

    for name, val in factors:
        st.markdown(
            f'<div class="factor-row"><span>{name}</span><span>{arrow.get(val,"")} {val}</span></div>',
            unsafe_allow_html=True,
        )


def render_best_outdoor_window(forecast_rows: list):
    """Find the best 2-hour window with lowest average PM2.5 in the next 24 hours."""
    if not forecast_rows or len(forecast_rows) < 2:
        return

    best_start_idx = 0
    min_avg = float("inf")
    for i in range(len(forecast_rows) - 1):
        avg2 = (forecast_rows[i]["pm25"] + forecast_rows[i + 1]["pm25"]) / 2.0
        if avg2 < min_avg:
            min_avg = avg2
            best_start_idx = i

    t1 = forecast_rows[best_start_idx]["time"]
    t2 = forecast_rows[best_start_idx + 1]["time"]
    cat = forecast_rows[best_start_idx]["category"]

    st.markdown(
        f'<div class="best-time-box">'
        f'<strong style="font-size:16px; color:#6ee7b7;">🌟 Best Window for Outdoor Activity: {t1} – {t2}</strong><br>'
        f'<span style="font-size:13.5px; color:#d1fae5;">Predicted average concentration is lowest at <strong>{min_avg:.1f} µg/m³</strong> ({cat}). '
        f'Ideal time for jogging, walking, cycling, or outdoor ventilation.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_health_advisories(current_category: str):
    """Render targeted health advisory cards for 4 key personas."""
    is_poor = current_category in ("Poor", "Very Poor", "Severe", "Unhealthy", "Very Unhealthy", "Hazardous")
    is_mod = current_category in ("Moderate", "Unhealthy for Sensitive Groups", "Satisfactory")

    # Personas configuration
    if is_poor:
        adv = {
            "athletes": ("⚠️ High Caution", "#f97316", "Avoid high-intensity outdoor cardio. Shift workouts indoors or wear an N95 respirator if outside."),
            "respiratory": ("🚨 High Risk", "#ef4444", "Keep emergency inhalers accessible. Keep windows closed and run HEPA air purifiers indoors."),
            "elderly": ("⚠️ Limit Exposure", "#f97316", "Avoid morning walks and prolonged outdoor stays. Keep indoor spaces well-filtered."),
            "general": ("🛡️ Moderate Action", "#eab308", "Consider wearing a mask during peak traffic commutes. Avoid burning trash or lighting incense."),
        }
    elif is_mod:
        adv = {
            "athletes": ("✅ Low-Moderate", "#22c55e", "Safe for general outdoor training. Sensitive individuals should pace intensity during morning hours."),
            "respiratory": ("⚠️ Watch Symptoms", "#eab308", "Monitor for coughing or throat irritation. Consider light indoor activity during peak traffic."),
            "elderly": ("✅ Acceptable", "#22c55e", "Safe for normal outdoor activities. Avoid prolonged stays near busy highways."),
            "general": ("✅ Normal", "#22c55e", "Good for daily commutes and activities. Standard ventilation is safe."),
        }
    else:
        adv = {
            "athletes": ("🌟 Ideal Conditions", "#10b981", "Perfect for outdoor runs, cycling, and vigorous athletics. Enjoy clean ambient air!"),
            "respiratory": ("🌟 Clean Air", "#10b981", "Air quality is pristine. Open windows for fresh air circulation and enjoy outdoor walks."),
            "elderly": ("🌟 Safe & Fresh", "#10b981", "Great time for park visits and outdoor relaxation. No respiratory precautions needed."),
            "general": ("🌟 Clean Day", "#10b981", "Optimal air quality across the city. Ideal time for natural ventilation."),
        }

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "🏃 Athletes & Joggers", adv["athletes"]),
        (c2, "🫁 Asthma & Patients", adv["respiratory"]),
        (c3, "👶 Children & Elderly", adv["elderly"]),
        (c4, "🚗 Commuters & Public", adv["general"]),
    ]

    for col, title, (status_text, status_color, desc) in cards:
        with col:
            st.markdown(
                f'<div class="advisory-card">'
                f'<div class="title">{title}</div>'
                f'<div class="status" style="color:{status_color};">{status_text}</div>'
                f'<div class="desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_ai_daily_briefing(city: str, current: dict, forecast_rows: list, latest_row: pd.Series):
    """Generate an analytical briefing of atmospheric trends and forecasted particulate dispersion."""
    if not forecast_rows:
        return

    peak = max(forecast_rows, key=lambda r: r["pm25"])
    low = min(forecast_rows, key=lambda r: r["pm25"])
    cur_pm = current["pm25"]
    
    wind = float(latest_row.get("wind_speed", 3.0))
    humidity = float(latest_row.get("humidity", 50.0))

    # Atmospheric condition analysis
    dispersion = "moderate particulate dispersion"
    if wind < 2.0:
        dispersion = "calm winds causing particulate trapping and ground-level accumulation"
    elif wind > 5.0:
        dispersion = "strong boundary-layer winds actively ventilating and dispersing pollutants"

    hum_note = ""
    if humidity > 75:
        hum_note = "High relative humidity may facilitate hygroscopic growth of fine aerosols."

    briefing_text = (
        f"**{city} Air Intelligence Briefing:** Air quality currently stands at **{cur_pm} µg/m³** ({current['category']}). "
        f"Over the next 24 hours, $\\text{{PM}}_{{2.5}}$ levels are projected to peak at **{peak['pm25']} µg/m³** around **{peak['time']}**, "
        f"before relaxing toward a 24-hour minimum of **{low['pm25']} µg/m³** at **{low['time']}**. "
        f"Atmospheric modeling detects {dispersion}. {hum_note}"
    )

    st.markdown(
        f'<div class="ai-briefing-box">'
        f'<div class="header">🤖 AI Atmospheric & Environmental Intelligence Briefing</div>'
        f'<div class="content">{briefing_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_alert(city: str, forecast_rows: list):
    """Show an alert only if forecast crosses into Poor/Very Poor/Severe/Unhealthy territory."""
    bad = [r for r in forecast_rows if r["category"] in
           ("Unhealthy", "Very Unhealthy", "Hazardous", "Poor", "Very Poor", "Severe")]
    if not bad:
        return
    worst = max(bad, key=lambda r: r["pm25"])
    duration = len(bad)
    st.markdown(
        f'<div class="alert-box"><strong>⚠️ Air Quality Advisory Alert</strong><br>'
        f'{city} is projected to experience <strong>{worst["category"]}</strong> air quality around <strong>{worst["time"]}</strong>.<br>'
        f'Particulate concentrations are expected to remain elevated for approximately <strong>{duration} hour(s)</strong>.</div>',
        unsafe_allow_html=True,
    )


def render_pollutants_breakdown(pm25: float, standard: str = "CPCB", live_pollutants: dict = None):
    """Render a dedicated 6-column glassmorphism section for all criteria air pollutants."""
    pm25 = max(1.0, float(pm25))

    # Standard NAAQS Indian and EPA limits
    limit_pm25 = 60.0 if standard == "CPCB" else 35.0
    limit_pm10 = 100.0 if standard == "CPCB" else 150.0
    limit_no2 = 80.0 if standard == "CPCB" else 100.0
    limit_so2 = 80.0 if standard == "CPCB" else 75.0
    limit_co = 2.0  # mg/m3
    limit_o3 = 100.0 if standard == "CPCB" else 70.0

    if live_pollutants:
        pm25 = live_pollutants.get("pm25", pm25)
        pm10 = live_pollutants.get("pm10", round(pm25 * 1.76, 1))
        no2 = live_pollutants.get("no2", round(16.0 + pm25 * 0.26, 1))
        so2 = live_pollutants.get("so2", round(7.0 + pm25 * 0.11, 1))
        co = live_pollutants.get("co", round(0.35 + pm25 * 0.007, 2))
        o3 = live_pollutants.get("o3", round(20.0 + pm25 * 0.15, 1))
    else:
        pm10 = round(pm25 * 1.76, 1)
        no2 = round(16.0 + pm25 * 0.26, 1)
        so2 = round(7.0 + pm25 * 0.11, 1)
        co = round(0.35 + pm25 * 0.007, 2)
        o3 = round(20.0 + pm25 * 0.15, 1)

    pollutants = [
        ("🌫️ PM2.5", "Fine Inhalable Particles", f"{pm25} µg/m³", pm25, limit_pm25, f"Limit: {limit_pm25:.0f} µg/m³"),
        ("💨 PM10", "Coarse Dust Particulates", f"{pm10} µg/m³", pm10, limit_pm10, f"Limit: {limit_pm10:.0f} µg/m³"),
        ("🚗 NO₂", "Nitrogen Dioxide", f"{no2} µg/m³", no2, limit_no2, f"Limit: {limit_no2:.0f} µg/m³"),
        ("🏭 SO₂", "Sulfur Dioxide", f"{so2} µg/m³", so2, limit_so2, f"Limit: {limit_so2:.0f} µg/m³"),
        ("🔥 CO", "Carbon Monoxide", f"{co} mg/m³", co, limit_co, f"Limit: {limit_co:.1f} mg/m³"),
        ("☀️ O₃", "Ground-Level Ozone", f"{o3} µg/m³", o3, limit_o3, f"Limit: {limit_o3:.0f} µg/m³"),
    ]

    cols = st.columns(6)
    for col, (sym, name, val_str, val, limit, limit_str) in zip(cols, pollutants):
        pct = min(100.0, (val / limit) * 100.0)
        if pct <= 60:
            badge_text, badge_bg, bar_color = "Safe", "#10b981", "#10b981"
        elif pct <= 100:
            badge_text, badge_bg, bar_color = "Moderate", "#f59e0b", "#f59e0b"
        else:
            badge_text, badge_bg, bar_color = "High", "#ef4444", "#ef4444"

        with col:
            st.markdown(
                f'<div class="pollutant-card">'
                f'<div class="pollutant-top">'
                f'<span class="pollutant-sym">{sym}</span>'
                f'<span class="badge" style="background:{badge_bg}; color:#ffffff; font-size:11px; padding:2px 8px;">{badge_text}</span>'
                f'</div>'
                f'<div class="pollutant-name">{name}</div>'
                f'<div class="pollutant-val">{val_str}</div>'
                f'<div class="pollutant-bar-bg"><div class="pollutant-bar-fill" style="width:{pct:.0f}%; background:{bar_color};"></div></div>'
                f'<div class="pollutant-limit">{limit_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
