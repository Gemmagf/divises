import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import requests

# Constants
API_KEY = "F4CB01BEKYJ5SHZF"  # Substitueix per la teva clau API real
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
@st.cache_data(show_spinner=True)
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
        else:
            st.error(f"Resposta inesperada de l'API: {data}")
    else:
        st.error(f"Error en la petició: {response.status_code} - {response.text}")

    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    return generate_fake_data(start_date, end_date, start_rate=1.10)

# Inicialització de l'estat
if "rules" not in st.session_state:
    st.session_state.rules = []

if "update_data" not in st.session_state:
    st.session_state.update_data = False

if "active_mode" not in st.session_state:
    st.session_state.active_mode = None  # Per recordar l'opció seleccionada

# Funció per generar targetes de regles
def generate_rule_cards():
    for i, rule in enumerate(st.session_state.rules):
        with st.expander(f"Regla {i + 1}"):
            rule["type"] = st.radio(
                f"Tipus de regla {i + 1}:",
                ["Simple", "Condicionada"],
                key=f"rule_type_{i}",
                index=["Simple", "Condicionada"].index(rule["type"])
            )
            rule["target_rate"] = st.number_input(
                f"Tipus de canvi objectiu per a la Regla {i + 1}:",
                min_value=0.0,
                step=0.01,
                key=f"target_rate_{i}",
                value=rule["target_rate"]
            )
            rule["coverage_percentage"] = st.slider(
                f"Percentatge de cobertura per a la Regla {i + 1}:",
                min_value=0,
                max_value=100,
                value=rule["coverage_percentage"],
                key=f"coverage_percentage_{i}"
            )
            if rule["type"] == "Condicionada":
                rule["trend_condition"] = st.radio(
                    f"Condició de tendència per a la Regla {i + 1}:",
                    ["Cap", "Tendència alcista", "Tendència baixista"],
                    key=f"trend_condition_{i}",
                    index=["Cap", "Tendència alcista", "Tendència baixista"].index(rule["trend_condition"])
                )
            else:
                rule["trend_condition"] = "Cap"

# Funció per configurar el nombre de regles
def configure_rules():
    st.subheader("Selecciona el nombre de regles")
    cols = st.columns(6)
    num_rules = len(st.session_state.rules)
    for i, col in enumerate(cols):
        if col.button(str(i), use_container_width=True):
            if i < num_rules:
                st.session_state.rules = st.session_state.rules[:i]
            else:
                st.session_state.rules.extend(
                    [{"type": "Simple", "target_rate": 1.0, "coverage_percentage": 50, "trend_condition": "Cap"} for _ in range(i - num_rules)]
                )
            st.experimental_rerun()
    return len(st.session_state.rules)


# Model SARIMA
def fit_sarima_model(data):
    model = SARIMAX(data, order=(0, 1, 0), seasonal_order=(0, 0, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def predict_sarima(fitted_model, steps=30):
    forecast = fitted_model.get_forecast(steps=steps)
    return forecast.predicted_mean

# Aplicació principal
st.title("Eina de Decisió Financera Forex")

# Sidebar
st.sidebar.header("Mode de l'Aplicació")
col1, col2 = st.sidebar.columns(2)

if col1.button('Eina'):
    st.session_state.active_mode = "eina"

if col2.button('Validador'):
    st.session_state.active_mode = "validador"
from_currency = st.sidebar.selectbox("Moneda base", ["EUR", "USD", "GBP", "JPY"])
to_currency = st.sidebar.selectbox("Moneda destí", ["USD", "EUR", "GBP", "JPY"])
volume = st.sidebar.number_input("Volum necessari", min_value=1.0, step=1.0)

# Mode Eina
if st.session_state.active_mode == "eina":
    if from_currency == to_currency:
        st.warning("La moneda base i destí no poden ser iguals.")
    else:
        st.subheader("Evolució històrica del tipus de canvi")
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            data = data[-730:]
            st.line_chart(data)
        else:
            st.error("No s'han pogut obtenir dades històriques.")

        st.subheader("Defineix el preu escandall")
        suggested_price = data["Close"].iloc[-1] if data is not None else None
        preu_escandall = st.number_input("Preu escandall (€/$)", value=suggested_price or 1.10, step=0.01)

        num_rules = configure_rules()
        if num_rules > 0:
            generate_rule_cards()

# Mode Validador
elif st.session_state.active_mode == "validador":
    st.title("Validador de Regles de Decisió")
    if from_currency != to_currency:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            data = data[-730:]

            num_rules = configure_rules()
            if num_rules > 0:
                generate_rule_cards()

                st.subheader("Comparació de Resultats")
                naive_returns = data["Close"].pct_change().cumsum()
                tool_returns = data["Close"].pct_change().cumsum()  # Simulació placeholder
                st.line_chart({"Naive Decisions": naive_returns, "Eina Decisions": tool_returns})
            else:
                st.info("Defineix almenys una regla per simular l'evolució.")
        else:
            st.error("No s'han pogut obtenir dades històriques.")
    else:
        st.warning("La moneda base i destí no poden ser iguals.")
else:
    st.write("Selecciona una opció")