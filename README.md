# 🌍 AirSense AI — Environmental Intelligence & Multi-City AQI Forecasting System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AirSense AI** is a production-grade environmental intelligence platform that combines **real-time atmospheric satellite streaming**, **ground-level particulate monitoring (OpenAQ v3)**, and **PyTorch Deep Recurrent Neural Networks (Multi-Step 2-Layer LSTM)** to provide 24-hour predictive air quality forecasts, public health advisories, and outdoor activity planning across **30 major Indian cities**.

---

## 🌟 Key Features

* **📡 Live Real-Time Pollutant Streaming**: Instantaneous 0-minute lag ingestion for all 6 criteria air pollutants ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, $\text{O}_3$) and surface meteorological parameters.
* **🧠 Multi-Step Deep Learning Forecasts**: Custom 2-Layer PyTorch LSTM neural networks predicting 24-hour future trajectories with **90% confidence uncertainty ribbons**.
* **🇮🇳 All-India Coverage (30 Major Cities)**: Complete real-time monitoring across North, South, West, East, and Central India.
* **⚖️ Dual AQI Standard Support**: Dynamic switching between official **Indian CPCB (NAQI)** and **US EPA** regulatory standards.
* **🏃 Best Outdoor Activity Recommender**: Rolling 2-hour minimum search algorithm calculating the cleanest window of the day for outdoor exercise and commuting.
* **🛡️ Targeted Public Health Advisories**: Actionable medical guidelines tailored for 4 distinct personas (Athletes, Respiratory Patients, Children/Elderly, Commuters).
* **📊 Multi-City Analytics & Side-by-Side Duel**: Compare up to 4 cities simultaneously with overlaid 24-hour forecast trajectories, national leaderboards (cleanest vs most polluted), and regional comparisons.
* **📥 One-Click CSV Export**: Instant report generation for researchers, municipal planners, and citizens.

---

## 🏗️ System Architecture

```
                       ┌─────────────────────────────────────────────────────────┐
                       │                   AirSense AI Dashboard                 │
                       │           (Streamlit + Custom Glassmorphism UI)         │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                       ┌────────────────────────────┴────────────────────────────┐
                       │                   FastAPI Backend REST Layer            │
                       │                    (Uvicorn ASGI Server)                │
                       └──────────────┬───────────────────────────┬──────────────┘
                                      │                           │
          ┌───────────────────────────┴───────────┐   ┌───────────┴──────────────────────────┐
          │      Deep Learning Forecast Engine    │   │      Live Atmospheric Streaming      │
          │  • PyTorch (2-Layer Multi-Step LSTM)  │   │  • Open-Meteo Real-Time API          │
          │  • Scikit-Learn (MinMax Scalers)      │   │  • OpenAQ v3 Ground Sensors          │
          │  • 48-Hour Sequential Feature Vectors │   │  • 6 Criteria Pollutants Stream      │
          └───────────────────────────────────────┘   └──────────────────────────────────────┘
```

---

## 🏙️ Monitored Cities Across India (30 Cities)

* **North**: Delhi, Chandigarh, Amritsar, Lucknow, Varanasi, Agra, Dehradun, Shimla, Srinagar
* **West**: Mumbai, Pune, Nagpur, Ahmedabad, Surat, Jaipur
* **South**: Bengaluru, Mysuru, Chennai, Coimbatore, Hyderabad, Visakhapatnam, Kochi, Thiruvananthapuram
* **East & Northeast**: Kolkata, Patna, Ranchi, Bhubaneswar, Guwahati
* **Central**: Bhopal, Indore

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/rohanrtx500/airsense-ai.git
cd airsense-ai
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
OPENAQ_API_KEY=your_openaq_api_key_here
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

---

## ⚡ Running the Application

### Start the FastAPI Backend Server
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at `http://127.0.0.1:8000/docs`.

### Start the Streamlit Dashboard
```bash
python -m streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser to access the live dashboard.

---

## 🧠 Machine Learning & Deep Learning Details

* **Model Architecture**: 2-Layer LSTM with 64 hidden units, dropout ($p=0.2$), and a linear projection head mapping directly to a 24-step horizon.
* **Feature Engineering**:
  * Auto-regressive lag terms: $t-1, t-2, t-3, t-6, t-12, t-24, t-48$ hours.
  * Rolling window statistics: $6\text{h}, 12\text{h}, 24\text{h}$ rolling means and rolling standard deviations.
  * Cyclical time encodings: $\sin/\cos$ transformations of hour of day and day of week.
* **Optimization**: Adam optimizer ($\text{lr} = 0.001$), MSE Loss, and early stopping with checkpoint restoration.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
