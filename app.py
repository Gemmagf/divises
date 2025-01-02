import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def prepare_features(data, look_back=30):
    """Prepara les dades per a entrenar Random Forest amb una finestra mòbil."""
    features, targets = [], []
    for i in range(len(data) - look_back):
        features.append(data[i:i + look_back])
        targets.append(data[i + look_back])
    return np.array(features), np.array(targets)

# Funció per ajustar Random Forest amb una finestra mòbil
def fit_random_forest(data, look_back=30):
    data_values = data['Close'].values
    X, y = prepare_features(data_values, look_back)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ajustem Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")
    return model, look_back

# Funció per fer prediccions amb Random Forest
def predict_random_forest(model, steps, data, look_back):
    last_data = data['Close'].values[-look_back:]
    predictions = []
    
    for _ in range(steps):
        input_features = last_data.reshape(1, -1)
        next_pred = model.predict(input_features)[0]
        predictions.append(next_pred)
        last_data = np.append(last_data[1:], next_pred)  # Shift i afegeix la predicció
        
    return predictions
    
    
    
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

def simulate_purchases(data, rules, initial_balance=10000):
    """
    Simulate purchases based on rules and calculate remaining balance.
    
    Parameters:
        data (pd.DataFrame): The financial data (e.g., last 30 days of prices).
        rules (list): List of rules dictating when and how to purchase.
        initial_balance (float): Starting balance for simulation.
    
    Returns:
        pd.DataFrame: Details of purchases.
        float: Remaining balance.
    """
    purchases = []
    balance = initial_balance
    
    for index, row in data.iterrows():
        for rule in rules:
            # Check the rule conditions
            if rule['type'] == 'Simple' and row['Close'] <= rule['target_rate']:
                amount_to_purchase = (balance * rule['coverage_percentage'] / 100)
                balance -= amount_to_purchase
                purchases.append({
                    'Date': index,
                    'Price': row['Close'],
                    'Amount': amount_to_purchase
                })
            elif rule['type'] == 'Condicionada' and row['Close'] <= rule['target_rate']:
                if rule['trend_condition'] == 'Tendència alcista' and row['Close'] > row['Close'].mean():
                    amount_to_purchase = (balance * rule['coverage_percentage'] / 100)
                    balance -= amount_to_purchase
                    purchases.append({
                        'Date': index,
                        'Price': row['Close'],
                        'Amount': amount_to_purchase
                    })
                elif rule['trend_condition'] == 'Tendència baixista' and row['Close'] < row['Close'].mean():
                    amount_to_purchase = (balance * rule['coverage_percentage'] / 100)
                    balance -= amount_to_purchase
                    purchases.append({
                        'Date': index,
                        'Price': row['Close'],
                        'Amount': amount_to_purchase
                    })
    
    return pd.DataFrame(purchases), balance

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
            st.experimental_rerun()
    return len(st.session_state.rules)

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
        generate_rule_cards()
        
    if st.button("Simula la transacció"):
        model, look_back = fit_random_forest(data)
        prediction = predict_random_forest(model, steps=30, data=data, look_back=look_back)
        st.line_chart(prediction, use_container_width=True)
        purchases, remaining_balance = simulate_purchases(data[-30:], st.session_state.rules)
        st.write(f"Balanç restant: {remaining_balance:.2f}")
        st.write("Compres realitzades:")
        st.dataframe(purchases)

# Mode Validador
elif st.session_state.active_mode == "Validador":
    st.title("Validador de Regles de Decisió")
    if from_currency != to_currency:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            data = data[-730:]
            st.line_chart(data["Close"], height=200, use_container_width=True)
            configure_rules()
            generate_rule_cards()
        else:
            st.error("No s'han pogut obtenir dades històriques.")
    else:
        st.warning("La moneda base i destí no poden ser iguals.")
