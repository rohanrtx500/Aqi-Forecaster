"""
Centralized city configuration. Kolkata reuses the existing flat data/models
folders (already produced by Phases 1-5) so nothing already working breaks.
Other cities use data/<slug>/ and models/<slug>/ — until someone runs the
same pipeline for them, they simply show as unavailable (never faked).
"""
import os

from src.utils import DATA_DIR as _LEGACY_DATA_DIR, MODELS_DIR as _LEGACY_MODELS_DIR

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_ROOT = os.path.join(_PROJECT_ROOT, "data")
MODELS_ROOT = os.path.join(_PROJECT_ROOT, "models")

CITIES = {
    "Kolkata":            {"slug": "kolkata",            "lat": 22.5726, "lon": 88.3639, "state": "West Bengal",       "timezone": "Asia/Kolkata"},
    "Delhi":              {"slug": "delhi",              "lat": 28.7041, "lon": 77.1025, "state": "NCR / Delhi",        "timezone": "Asia/Kolkata"},
    "Mumbai":             {"slug": "mumbai",             "lat": 19.0760, "lon": 72.8777, "state": "Maharashtra",       "timezone": "Asia/Kolkata"},
    "Bengaluru":          {"slug": "bengaluru",          "lat": 12.9716, "lon": 77.5946, "state": "Karnataka",         "timezone": "Asia/Kolkata"},
    "Chennai":            {"slug": "chennai",            "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu",        "timezone": "Asia/Kolkata"},
    "Hyderabad":          {"slug": "hyderabad",          "lat": 17.3850, "lon": 78.4867, "state": "Telangana",         "timezone": "Asia/Kolkata"},
    "Pune":               {"slug": "pune",               "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra",       "timezone": "Asia/Kolkata"},
    "Ahmedabad":          {"slug": "ahmedabad",          "lat": 23.0225, "lon": 72.5714, "state": "Gujarat",           "timezone": "Asia/Kolkata"},
    "Jaipur":             {"slug": "jaipur",             "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan",         "timezone": "Asia/Kolkata"},
    "Lucknow":            {"slug": "lucknow",            "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh",     "timezone": "Asia/Kolkata"},
    "Patna":              {"slug": "patna",              "lat": 25.5941, "lon": 85.1376, "state": "Bihar",             "timezone": "Asia/Kolkata"},
    "Chandigarh":         {"slug": "chandigarh",         "lat": 30.7333, "lon": 76.7794, "state": "Punjab & Haryana",  "timezone": "Asia/Kolkata"},
    "Guwahati":           {"slug": "guwahati",           "lat": 26.1445, "lon": 91.7362, "state": "Assam",             "timezone": "Asia/Kolkata"},
    "Bhopal":             {"slug": "bhopal",             "lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh",   "timezone": "Asia/Kolkata"},
    "Kochi":              {"slug": "kochi",              "lat": 9.9312,  "lon": 76.2673, "state": "Kerala",            "timezone": "Asia/Kolkata"},
    "Visakhapatnam":      {"slug": "visakhapatnam",      "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh",    "timezone": "Asia/Kolkata"},
    "Varanasi":           {"slug": "varanasi",           "lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh",     "timezone": "Asia/Kolkata"},
    "Surat":              {"slug": "surat",              "lat": 21.1702, "lon": 72.8311, "state": "Gujarat",           "timezone": "Asia/Kolkata"},
    "Indore":             {"slug": "indore",             "lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh",   "timezone": "Asia/Kolkata"},
    "Nagpur":             {"slug": "nagpur",             "lat": 21.1458, "lon": 79.0882, "state": "Maharashtra",       "timezone": "Asia/Kolkata"},
    "Agra":               {"slug": "agra",               "lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh",     "timezone": "Asia/Kolkata"},
    "Amritsar":           {"slug": "amritsar",           "lat": 31.6340, "lon": 74.8723, "state": "Punjab",            "timezone": "Asia/Kolkata"},
    "Ranchi":             {"slug": "ranchi",             "lat": 23.3441, "lon": 85.3096, "state": "Jharkhand",         "timezone": "Asia/Kolkata"},
    "Bhubaneswar":        {"slug": "bhubaneswar",        "lat": 20.2961, "lon": 85.8245, "state": "Odisha",            "timezone": "Asia/Kolkata"},
    "Coimbatore":         {"slug": "coimbatore",         "lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu",        "timezone": "Asia/Kolkata"},
    "Thiruvananthapuram": {"slug": "thiruvananthapuram", "lat": 8.5241,  "lon": 76.9366, "state": "Kerala",            "timezone": "Asia/Kolkata"},
    "Dehradun":           {"slug": "dehradun",           "lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand",       "timezone": "Asia/Kolkata"},
    "Shimla":             {"slug": "shimla",             "lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh",  "timezone": "Asia/Kolkata"},
    "Srinagar":           {"slug": "srinagar",           "lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir",   "timezone": "Asia/Kolkata"},
    "Mysuru":             {"slug": "mysuru",             "lat": 12.2958, "lon": 76.6394, "state": "Karnataka",         "timezone": "Asia/Kolkata"},
}

REQUIRED_MODEL_FILES = ["lstm_24h_model.pth", "lstm_24h_scaler.pkl", "lstm_24h_features.pkl"]


def city_data_dir(city: str) -> str:
    if city == "Kolkata":
        return _LEGACY_DATA_DIR  # existing flat data/ folder
    return os.path.join(DATA_ROOT, CITIES[city]["slug"])


def city_models_dir(city: str) -> str:
    if city == "Kolkata":
        return _LEGACY_MODELS_DIR  # existing flat models/ folder
    return os.path.join(MODELS_ROOT, CITIES[city]["slug"])


def processed_csv_path(city: str) -> str:
    return os.path.join(city_data_dir(city), "processed_aqi.csv")


def model_available(city: str) -> bool:
    d = city_models_dir(city)
    return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_MODEL_FILES)


def data_available(city: str) -> bool:
    return os.path.exists(processed_csv_path(city))
