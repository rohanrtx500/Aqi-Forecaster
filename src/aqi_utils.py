"""AQI category -> color/label helpers with dual-standard support (Indian CPCB & US EPA)."""
from src.utils import pm25_to_aqi as pm25_to_us_aqi

# ---- Indian CPCB (National AQI) Breakpoints (24h avg PM2.5 in ug/m3) ----
CPCB_BREAKPOINTS = [
    (0.0, 30.0, 0, 50, "Good"),
    (30.1, 60.0, 51, 100, "Satisfactory"),
    (60.1, 90.0, 101, 200, "Moderate"),
    (90.1, 120.0, 201, 300, "Poor"),
    (120.1, 250.0, 301, 400, "Very Poor"),
    (250.1, 500.0, 401, 500, "Severe"),
]

CATEGORY_COLORS = {
    # US EPA Categories
    "Good": "#2ecc71",
    "Satisfactory": "#27ae60",
    "Moderate": "#f1c40f",
    "Unhealthy for Sensitive Groups": "#f39c12",
    "Unhealthy": "#e67e22",
    "Poor": "#e67e22",
    "Very Unhealthy": "#e74c3c",
    "Very Poor": "#e74c3c",
    "Hazardous": "#8e44ad",
    "Severe": "#8e44ad",
    "Unknown": "#7f8c8d",
}


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, "#7f8c8d")


def pm25_to_cpcb_aqi(pm25: float):
    """Convert PM2.5 (ug/m3) to official Indian CPCB National AQI (NAQI)."""
    if pm25 is None or pm25 != pm25:
        return None, "Unknown"
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi, category in CPCB_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
            return round(aqi), category
    return 500, "Severe"


def pm25_to_aqi_labeled(pm25: float, standard: str = "CPCB"):
    """
    Returns (aqi, category, color) in one call.
    standard: 'CPCB' (Indian National AQI) or 'US_EPA' (US EPA AQI).
    """
    if standard == "US_EPA":
        aqi, category = pm25_to_us_aqi(pm25)
    else:
        aqi, category = pm25_to_cpcb_aqi(pm25)
    return aqi, category, category_color(category)

