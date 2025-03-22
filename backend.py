import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

class StockForecastModel:
    BASE_URL = "https://api.marketstack.com/v1/eod"
    API_KEY = "d468e90321e08b9f42001be629860645"

    def __init__(self):
        self.symbol = None
        self.df = None
        self.timeframe = "Monthly"
        self.start_date = None
        self.end_date = None

    def fetch_stock_data(self):
        if not self.symbol:
            st.error("Please select a stock symbol.")
            return None

        url = f"{self.BASE_URL}?access_key={self.API_KEY}&symbols={self.symbol}&date_from={self.start_date}&date_to={self.end_date}&limit=5000"

        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"API request failed with status code: {response.status_code}")
            return None

        data = response.json()
        if "data" not in data or not data["data"]:
            st.error("No data received from API.")
            return None

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"date": "timestamp", "close": "4. close"})
        df = df.set_index("timestamp")["4. close"].sort_index()

        self.df = df
        return self.df

    def process_data(self):
        if self.df is None:
            st.error("No data available. Fetch stock data first.")
            return None

        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
        if self.timeframe in freq_map:
            df_processed = self.df.resample(freq_map[self.timeframe]).last()
        else:
            st.error("Invalid timeframe selected.")
            return None

        df_processed = df_processed.dropna().reset_index()
        df_processed["Period_Ordinal"] = df_processed.index + 1
        df_processed["timestamp"] = df_processed["timestamp"].dt.tz_localize(None)
        return df_processed

    def apply_random_forest(self, df_processed):
        if len(df_processed) < 6:
            st.error("Not enough data for model training.")
            return None, None

        X = df_processed[["Period_Ordinal"]]
        y = df_processed["4. close"]
        
        X_train, y_train = X[:-1], y[:-1]
        X_test = pd.DataFrame({"Period_Ordinal": [X.iloc[-1, 0] + 1]})

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)[0]

        mse = mean_squared_error(y_train[-5:], model.predict(X_train[-5:]))
        return prediction, mse

    def apply_moving_average(self, df_processed, window=3):
        if len(df_processed) < window:
            st.error("Not enough data for Moving Average.")
            return None, None

        df_processed["Moving_Avg"] = df_processed["4. close"].rolling(window=window).mean()
        predicted_price = df_processed["Moving_Avg"].iloc[-1]
        mse = mean_squared_error(df_processed["4. close"].iloc[-window:], df_processed["Moving_Avg"].iloc[-window:])
        return predicted_price, mse

    def plot_predictions(self, df_processed, model_name):
        start_date_naive = pd.to_datetime(self.start_date).tz_localize(None)
        end_date_naive = pd.to_datetime(self.end_date).tz_localize(None)

        filtered_df = df_processed[(df_processed["timestamp"] >= start_date_naive) &
                                   (df_processed["timestamp"] <= end_date_naive)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df["timestamp"], y=filtered_df["4. close"], mode='lines', name='Actual Price', line=dict(color='blue')))
        
        if model_name == "Moving Average":
            fig.add_trace(go.Scatter(x=filtered_df["timestamp"], y=filtered_df["Moving_Avg"], mode='lines', name='Predicted (MA)', line=dict(color='red')))
        
        fig.update_layout(title=f"{model_name} Predictions vs Actual", xaxis_title="Date", yaxis_title="Stock Price")
        st.plotly_chart(fig)

    def run(self):
        st.title("📈 Stock Price Forecasting")
        stock_symbols = {"AAPL": "AAPL", "MSFT": "MSFT", "TSLA": "TSLA", "AMZN": "AMZN", "GOOGL": "GOOGL"}
        self.symbol = st.sidebar.selectbox("Select a Stock Symbol:", list(stock_symbols.keys()))
        self.timeframe = st.sidebar.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
        
        self.start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
        self.end_date = st.sidebar.date_input("End Date", datetime.today())

        if st.button("Fetch & Predict"):
            self.fetch_stock_data()
            if self.df is None:
                return
            
            df_processed = self.process_data()
            predicted_rf, mse_rf = self.apply_random_forest(df_processed)
            predicted_ma, mse_ma = self.apply_moving_average(df_processed)

            if mse_rf is not None and mse_ma is not None:
                best_model = "Random Forest" if mse_rf < mse_ma else "Moving Average"
                best_price = predicted_rf if mse_rf < mse_ma else predicted_ma
                st.write(f"Best Model: **{best_model}**")
                st.write(f"Predicted Stock Price: **${best_price:.2f}**")
                self.plot_predictions(df_processed, best_model)

if __name__ == "__main__":
    app = StockForecastModel()
    app.run()
