import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests

# Constants
API_KEY = "F4CB01BEKYJ5SHZF"
BASE_URL = "https://www.alphavantage.co/query"

# Funció per generar dades fictícies
def generate_fake_data(start_date, end_date, start_rate, volatility=0.01):
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Només dies laborables
    rates = [start_rate]
    for _ in range(len(date_range) - 1):
        change = np.random.normal(0, volatility)
        rates.append(rates[-1] * (1 + change))
    return pd.DataFrame({'Close': rates}, index=date_range)

# Funció per obtenir dades històriques
@st.cache
def fetch_historical_data(from_currency, to_currency):
    params = {
        'function': 'FX_DAILY',
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'apikey': API_KEY,
        'outputsize': 'full'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if "Time Series FX (Daily)" in data:
            rates = data["Time Series FX (Daily)"]
            df = pd.DataFrame.from_dict(rates, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.rename(columns={"4. close": "Close"}, inplace=True)
            return df[["Close"]]
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    return generate_fake_data(start_date, end_date, start_rate=1.10)

# Inicialitza l'estat si no està definit
if "rules" not in st.session_state:
    st.session_state.rules = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Eina"

# Sidebar
st.sidebar.header("Mode de l'Aplicació")
if st.sidebar.button("Eina"):
    st.session_state.active_mode = "Eina"
if st.sidebar.button("Validador"):
    st.session_state.active_mode = "Validador"

from_currency = st.sidebar.selectbox("Moneda base", ["EUR", "USD", "GBP", "JPY"])
to_currency = st.sidebar.selectbox("Moneda destí", ["USD", "EUR", "GBP", "JPY"])
volume = st.sidebar.number_input("Volum necessari", min_value=1.0, step=1.0)

if st.sidebar.button("Actualitzar dades"):
    fetch_historical_data.clear()
    st.success("Memòria cau esborrada. Dades actualitzades.")

# Funció per configurar el nombre de regles
def configure_rules():
    st.subheader("Selecciona el nombre de regles")
    cols = st.columns(6)
    num_rules = len(st.session_state.rules)

    for i, col in enumerate(cols):
        if col.button(str(i)):
            if i < num_rules:
                st.session_state.rules = st.session_state.rules[:i]
            else:
                st.session_state.rules.extend(
                    [{"type": "Simple", "target_rate": 1.0, "coverage_percentage": 50, "trend_condition": "Cap"} for _ in range(i - num_rules)]
                )
            break  # Per evitar múltiples crides consecutives
    # Genera targetes de regles
    generate_rule_cards()

# Funció per generar targetes de regles
def generate_rule_cards():
    for i, rule in enumerate(st.session_state.rules):
        with st.expander(f"Regla {i + 1}"):
            rule["type"] = st.radio(
                f"Tipus de regla {i + 1}:",
                ["Simple", "Condicionada"],
                key=f"rule_type_{i}",
                index=["Simple", "Condicionada"].index(rule.get("type", "Simple"))
            )
            rule["target_rate"] = st.number_input(
                f"Tipus de canvi objectiu per a la Regla {i + 1}:",
                min_value=0.0,
                step=0.01,
                key=f"target_rate_{i}",
                value=rule.get("target_rate", 1.0)
            )
            rule["coverage_percentage"] = st.slider(
                f"Percentatge de cobertura per a la Regla {i + 1}:",
                min_value=0,
                max_value=100,
                value=rule.get("coverage_percentage", 50),
                key=f"coverage_percentage_{i}"
            )
            if rule["type"] == "Condicionada":
                rule["trend_condition"] = st.radio(
                    f"Condició de tendència per a la Regla {i + 1}:",
                    ["Cap", "Tendència alcista", "Tendència baixista"],
                    key=f"trend_condition_{i}",
                    index=["Cap", "Tendència alcista", "Tendència baixista"].index(rule.get("trend_condition", "Cap"))
                )
            else:
                rule["trend_condition"] = "Cap"

# Mode Eina
if st.session_state.active_mode == "Eina":
    st.title("Eina de Decisió Financera Forex - Mode Eina")
    if from_currency == to_currency:
        st.warning("La moneda base i destí no poden ser iguals.")
    else:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            data = data[-730:]
            st.line_chart(data)
        else:
            st.error("No s'han pogut obtenir dades històriques.")
        configure_rules()

# Mode Validador
elif st.session_state.active_mode == "Validador":
    st.title("Validador de Regles de Decisió")
    if from_currency != to_currency:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            data = data[-730:]
            st.line_chart(data["Close"])
            configure_rules()
        else:
            st.error("No s'han pogut obtenir dades històriques.")
    else:
        st.warning("La moneda base i destí no poden ser iguals.")
