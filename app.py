import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
        'outputsize': 'compact'
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
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    return generate_fake_data(start_date, end_date, start_rate=1.10)

# Inicialitza l'estat si no està definit
if "rules" not in st.session_state:
    st.session_state.rules = []

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Eina"

if "initial_balance" not in st.session_state:
    st.session_state.initial_balance = 10000.0

if "target_date" not in st.session_state:
    st.session_state.target_date = 30

# Sidebar
st.sidebar.header("Mode de l'Aplicació")
if st.sidebar.button("Eina"):
    st.session_state.active_mode = "Eina"
if st.sidebar.button("Validador"):
    st.session_state.active_mode = "Validador"

from_currency = st.sidebar.selectbox("Moneda base", ["EUR", "USD", "GBP", "JPY"])
to_currency = st.sidebar.selectbox("Moneda destí", ["USD", "EUR", "GBP", "JPY"])
volume = st.sidebar.number_input("Volum necessari", min_value=1.0, step=1.0)
target_date = st.sidebar.slider("Dies fins a l'objectiu", min_value=1, max_value=365, value=30)
st.session_state.target_date = target_date

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

# Funció per simular resultats basats en regles
def simulate_purchases(data, rules, initial_balance):
    purchases = []
    balance = initial_balance

    for index, row in data.iterrows():
        for rule in rules:
            if rule["type"] == "Simple":
                # Regla simple: compra immediata si es compleix la condició
                if row["Close"] <= rule["target_rate"]:
                    amount_to_purchase = balance * rule["coverage_percentage"] / 100
                    balance -= amount_to_purchase
                    purchases.append({"Date": index, "Price": row["Close"], "Amount": amount_to_purchase, "Rule": "Simple"})
            elif rule["type"] == "Condicionada":
                # Regla condicionada: compra només si es compleix la condició de tendència
                if row["Close"] <= rule["target_rate"]:
                    if rule["trend_condition"] == "Tendència alcista" and row["Close"] > data["Close"].rolling(5).mean().loc[index]:
                        amount_to_purchase = balance * rule["coverage_percentage"] / 100
                        balance -= amount_to_purchase
                        purchases.append({"Date": index, "Price": row["Close"], "Amount": amount_to_purchase, "Rule": "Condicionada (Alcista)"})
                    elif rule["trend_condition"] == "Tendència baixista" and row["Close"] < data["Close"].rolling(5).mean().loc[index]:
                        amount_to_purchase = balance * rule["coverage_percentage"] / 100
                        balance -= amount_to_purchase
                        purchases.append({"Date": index, "Price": row["Close"], "Amount": amount_to_purchase, "Rule": "Condicionada (Baixista)"})
                    elif rule["trend_condition"] == "Cap":
                        amount_to_purchase = balance * rule["coverage_percentage"] / 100
                        balance -= amount_to_purchase
                        purchases.append({"Date": index, "Price": row["Close"], "Amount": amount_to_purchase, "Rule": "Condicionada (Cap)"})

    total_spent = sum(purchase['Amount'] / purchase['Price'] for purchase in purchases)
    final_value = total_spent * data.iloc[-1]['Close']
    profit_loss = final_value - initial_balance

    return pd.DataFrame(purchases), balance, profit_loss



# Funció per representar gràficament la predicció i les regles aplicades
def plot_simulation(data, predictions, purchases):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Tipus de canvi real', color='blue')

    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=len(predictions))
    plt.plot(future_dates, predictions, label='Predicció tipus de canvi', color='red')

    for _, purchase in purchases.iterrows():
        if "Condicionada" in purchase['Rule']:
            color = 'orange'
        else:
            color = 'green'
        plt.axvline(purchase['Date'], color=color, linestyle='--', label=f"Compra ({purchase['Rule']})")

    plt.xlabel('Data')
    plt.ylabel('Tipus de canvi')
    plt.title("Simulació de l'evolució del Forex amb regles aplicades")
    plt.legend()
    st.pyplot(plt)


# Funció per predir el tipus de canvi futur utilitzant Random Forest amb lags
def predict_forex(data, periods=30, lag=5):
    data = data.reset_index()
    data['Index'] = np.arange(len(data))

    # Generar característiques amb lags
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)

    data = data[data.notnull().all(axis=1)]

    X = data[[f'lag_{i}' for i in range(1, lag + 1)]]
    y = data['Close']

    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
    model.fit(X, y)

    # Predicció iterativa
    last_features = X.iloc[-1].values
    predictions = []
    for _ in range(periods):
        next_pred = model.predict([last_features])[0]
        predictions.append(next_pred)
        last_features = np.roll(last_features, -1)
        last_features[-1] = next_pred

    return predictions

# Mode Eina
if st.session_state.active_mode == "Eina":
    st.title("Eina de Decisió Financera Forex - Mode Eina")
    if from_currency == to_currency:
        st.warning("La moneda base i destí no poden ser iguals.")
    else:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            historical_data = data[-365:]
            st.line_chart(historical_data, use_container_width=True)

            configure_rules()

            if st.button("Simula la transacció"):
                predictions = predict_forex(historical_data, periods=10, lag=5)
                purchases, remaining_balance, profit_loss = simulate_purchases(historical_data, st.session_state.rules, st.session_state.initial_balance)

                st.write(f"Balanç restant: {remaining_balance:.2f}")
                st.write(f"Benefici/Pèrdua: {profit_loss:.2f}")

                if profit_loss > 0:
                    st.success("L'operació serà positiva.")
                else:
                    st.error("L'operació no serà positiva.")

                st.write("Compres realitzades:")
                st.dataframe(purchases)

                plot_simulation(historical_data, predictions, purchases)

# Mode Validador
# Mode Validador
elif st.session_state.active_mode == "Validador":
    st.title("Validador de Regles de Decisió")
    
    if from_currency != to_currency:
        data = fetch_historical_data(from_currency, to_currency)
        if data is not None:
            # Selecció del període històric
            st.subheader("Selecciona el període històric")
            start_date = st.date_input("Data d'inici", datetime.now() - timedelta(days=365 * 2))
            end_date = st.date_input("Data de final", datetime.now())
            
            if start_date >= end_date:
                st.error("La data d'inici ha de ser anterior a la data de final.")
            else:
                historical_data = data.loc[start_date:end_date]
                
                if historical_data.empty:
                    st.warning("No hi ha dades per al període seleccionat.")
                else:
                    st.line_chart(historical_data, use_container_width=True)
                    configure_rules()

                    if st.button("Validar Operacions"):
                        purchases, remaining_balance, profit_loss = simulate_purchases(historical_data, st.session_state.rules, st.session_state.initial_balance)

                        st.write(f"Balanç restant: {remaining_balance:.2f}")
                        st.write(f"Benefici/Pèrdua: {profit_loss:.2f}")

                        if profit_loss > 0:
                            st.success("L'operació hauria estat positiva.")
                        else:
                            st.error("L'operació hauria estat negativa.")

                        st.write("Compres realitzades:")
                        st.dataframe(purchases)

                        # Representació gràfica
                        plot_simulation(historical_data, [], purchases)
    else:
        st.warning("La moneda base i destí no poden ser iguals.")
