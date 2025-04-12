import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from google import genai
import openai

openai.api_key = "sk-proj-mkAEprCtbWcd2Ui2_eMMaLh5VSBHomM6olcpdtMlWiWxrixaYlT9KqFDOQpoufJGPrARTh1eyuT3BlbkFJTeB1XjaIcZ0ut1b6T0pHz7jLNWzGfxMe8z_1owjLy04r_0rDO9YxkuF7JBOppteHP66kIqQHA"

class StockForecastModel:
    BASE_URL = "https://api.marketstack.com/v1/eod"
    API_KEY = "d7fb2fb3a988d314e2019bd56e62965a"


    

    def __init__(self):
        self.symbol = None
        self.df = None
        self.timeframe = "Monthly"
        self.start_date = None
        self.end_date = None
        self.genai_client = genai.Client(api_key="AIzaSyANcyTEGxPrVnOQ1_KTD6xfNNmyYmJEyZ4")


    def generate_recommendation(self, symbol, start, end, model, price, mse, frequency):
        prompt = (
            f"Act as a professional stock trader and Based on the stock prediction results for {symbol} from {start} to {end} "
            f"using {frequency} frequency data:\n\n"
            f"• Best model: {model}\n"
            f"• Predicted price: ${price:.2f}\n"
            f"• Model MSE: {mse:.4f}\n\n"
            f"Generate a short recommendation (2-3 lines) for investors considering accuracy and model used. "
            f"Be realistic and professional."
        )
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"⚠️ Could not generate recommendation due to error: {e}"


    def generate_openai_recommendation(self, symbol, start, end, model, price, mse, frequency):

        client = openai.OpenAI(api_key="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")

        prompt = (
            f"Stock: {symbol}\n"
            f"Date Range: {start} to {end}\n"
            f"Frequency: {frequency}\n"
            f"Best Model: {model}\n"
            f"Predicted Price: ${price:.2f}\n"
            f"MSE: {mse:.4f}\n\n"
            "Give a short recommendation to an investor in 1-2 sentences based on this prediction. "
            "Keep it practical and realistic."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial advisor."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(OpenAI failed to generate recommendation: {str(e)})"



    def fetch_stock_data(self):
        if not self.symbol:
            st.error("Please select a stock symbol.")
            return None

        url = f"{self.BASE_URL}?access_key={self.API_KEY}&symbols={self.symbol}&date_from={self.start_date}&date_to={self.end_date}&limit=5000"

        try:
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
            # st.success(f"Successfully fetched data for {self.symbol} from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            return self.df
        except requests.exceptions.RequestException as e:
            st.error(f"Error during API request: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

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
        if df_processed is None or len(df_processed) < 6:
            return None, None

        X = df_processed[["Period_Ordinal"]]
        y = df_processed["4. close"]

        X_train, y_train = X[:-1], y[:-1]
        X_test = pd.DataFrame({"Period_Ordinal": [X.iloc[-1, 0] + 1]})
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        try:
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)[0]
            y_pred = model.predict(X_train[-5:])
            mse = mean_squared_error(y_train[-5:], y_pred)
            return prediction, mse
        except Exception as e:
            st.error(f"Error applying Random Forest: {e}")
            return None, None
            


    def apply_moving_average(self, df_processed, window=3):
            if df_processed is None or len(df_processed) < window:
                return None, None

            df_processed["Moving_Avg"] = df_processed["4. close"].rolling(window=window).mean()
            if df_processed["Moving_Avg"].isnull().any():
                return None, None
            predicted_price = df_processed["Moving_Avg"].iloc[-1]
            y_true = df_processed["4. close"].iloc[-window:]
            y_pred = df_processed["Moving_Avg"].iloc[-window:]
            try:
                mse = mean_squared_error(y_true, y_pred)
                return predicted_price, mse
            except ValueError:
                return None, None
            except Exception as e:
                st.error(f"Error applying Moving Average: {e}")
                return None, None
        

    def apply_double_exponential_smoothing(self, df_processed):
        if df_processed is None or len(df_processed) < 3:
            return None, None
        try:
            model = ExponentialSmoothing(df_processed["4. close"], trend="add", seasonal=None).fit()
            predicted_price = model.forecast(1).iloc[0]
            mse = mean_squared_error(df_processed["4. close"], model.fittedvalues)
            return predicted_price, mse
        except ValueError:
            return None, None
        except Exception as e:
            st.error(f"Error applying Double Exponential Smoothing: {e}")
            return None, None


    def apply_arima(self, df_processed):
            if df_processed is None:
                st.warning("Input DataFrame is None.")
                return None, None

            if len(df_processed) < 10:
                st.warning(f"Not enough data points ({len(df_processed)} < 10). Cannot apply ARIMA.")
                return None, None

            df_processed = df_processed.dropna(subset=["4. close"])
            if df_processed.empty:
                st.warning("No valid data points after removing NaN values.")
                return None, None

            if len(df_processed) < 3:
                st.warning(f"Insufficient valid data points ({len(df_processed)} < 3) after cleaning.")
                return None, None

            try:
                model = ARIMA(df_processed["4. close"], order=(1, 1, 1)).fit()
                predicted_price = model.forecast(steps=1)[0]

                fitted_values = model.fittedvalues.dropna()
                if len(fitted_values) < 2:
                    st.warning("Insufficient fitted values to calculate MSE.")
                    return predicted_price, None

                mse = mean_squared_error(df_processed["4. close"][:len(fitted_values)], fitted_values)
                return predicted_price, mse

            except ValueError as ve:
                #st.error(f"ARIMA ValueError: {ve}")
                return None, None
            except Exception as e:
                #st.error(f"ARIMA Error: {e}")
                return None, None



    def plot_predictions(self, df_processed, model_name):
    # ✅ Initialize fig at the start
        fig = go.Figure()
    
    # ✅ Ensure the function doesn't fail due to missing data
        if df_processed is None or df_processed.empty:
            st.warning("No data to plot.")
            st.plotly_chart(fig)  # Show empty figure
            return

        start_date_naive = pd.to_datetime(self.start_date).tz_localize(None)
        end_date_naive = pd.to_datetime(self.end_date).tz_localize(None)

        filtered_df = df_processed[(df_processed["timestamp"] >= start_date_naive) &
                                (df_processed["timestamp"] <= end_date_naive)].copy()

        if filtered_df.empty:
            st.warning("No data within the selected date range.")
            st.plotly_chart(fig)  # Show empty figure
            return

        fig.add_trace(go.Scatter(x=filtered_df["timestamp"], y=filtered_df["4. close"],
                                mode='lines', name='Actual Price', line=dict(color='blue')))

        filtered_df["Prediction"] = None

        try:
            if model_name == "Moving Average":
                filtered_df["Prediction"] = filtered_df["4. close"].rolling(window=3).mean()
            elif model_name == "Double Exponential Smoothing":
                model = ExponentialSmoothing(df_processed["4. close"], trend="add", seasonal=None).fit()
                filtered_df["Prediction"] = model.fittedvalues
            elif model_name == "ARIMA":
                model = ARIMA(df_processed["4. close"], order=(1,1,1)).fit()
                fitted_values = model.fittedvalues.dropna()
                filtered_df = filtered_df.iloc[-len(fitted_values):]
                filtered_df["Prediction"] = fitted_values
            elif model_name == "Random Forest":
                if "Period_Ordinal" in df_processed.columns:
                    X = df_processed[["Period_Ordinal"]]
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X[:-1], df_processed["4. close"][:-1])
                    filtered_df["Prediction"] = rf_model.predict(X[:len(filtered_df)])
                else:
                    st.error("Random Forest model requires 'Period_Ordinal' column.")
                    st.plotly_chart(fig)  # Show empty figure
                    return

            filtered_df["Prediction"].fillna(method='bfill', inplace=True)
            filtered_df = filtered_df.dropna(subset=['Prediction'])

            if not filtered_df.empty and 'Prediction' in filtered_df.columns:
                fig.add_trace(go.Scatter(x=filtered_df["timestamp"], y=filtered_df["Prediction"],
                                        mode='lines', name=f'Predicted ({model_name})', line=dict(color='red')))

        except Exception as e:
            st.error(f"Error fitting {model_name}: {e}")
            st.plotly_chart(fig)  # Show empty figure
            return



        
        fig.update_layout(
            title=dict(
                text=f"{model_name} Predictions vs Actual",
                font=dict(size=22, color='black')
            ),
            xaxis=dict(
                title=dict(text="Date", font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)'
            ),
            yaxis=dict(
                title=dict(text="Stock Price", font=dict(color="black")),
                tickfont=dict(color="black"),
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)'
            ),
            plot_bgcolor='rgba(255, 255, 255, 0.4)',  # Semi-transparent background
            paper_bgcolor='rgba(255, 255, 255, 0.0)',
            font=dict(color='black'),
            legend=dict(
            orientation="h",  # Horizontal layout
            yanchor="bottom",
            y=-0.3,            # Push it below the chart
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )



        st.plotly_chart(fig)


    def run(self):
        
        try:
            df = pd.read_csv("stocks.csv")  # Replace with the actual filename
            df = df[df["Test Issue"] == "N"]  # Exclude test listings
            df = df[df["ETF"] == "N"]         # Optional: Exclude ETFs if not needed
            stock_symbols = dict(zip(df["Symbol"], df["Security Name"]))
        except Exception as e:
            st.error("Failed to load stock symbols from CSV. Using default list.")
            stock_symbols = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation",
                "TSLA": "Tesla, Inc.",
                "AMZN": "Amazon.com, Inc.",
                "GOOGL": "Alphabet Inc.",
                "META": "Meta Platforms, Inc.",
                "NFLX": "Netflix, Inc.",
                "NVDA": "NVIDIA Corporation",
                "AMD": "Advanced Micro Devices, Inc.",
                "INTC": "Intel Corporation",
                "IBM": "International Business Machines Corporation",
                "ORCL": "Oracle Corporation",
                "CSCO": "Cisco Systems, Inc.",
                "QCOM": "Qualcomm Incorporated",
                "ADBE": "Adobe Inc.",
                "PYPL": "PayPal Holdings, Inc.",
                "CRM": "Salesforce, Inc.",
                "UBER": "Uber Technologies, Inc.",
                "LYFT": "Lyft, Inc.",
                "SQ": "Block, Inc. (Square)",
                "BABA": "Alibaba Group Holding Limited",
                "TCEHY": "Tencent Holdings Limited",
                "SHOP": "Shopify Inc.",
                "V": "Visa Inc.",
                "MA": "Mastercard Incorporated",
                "JPM": "JPMorgan Chase & Co.",
                "GS": "Goldman Sachs Group, Inc.",
                "WMT": "Walmart Inc.",
                "COST": "Costco Wholesale Corporation",
                "PEP": "PepsiCo, Inc.",
                "KO": "The Coca-Cola Company",
                "MCD": "McDonald's Corporation",
                "NKE": "Nike, Inc.",
                "DIS": "The Walt Disney Company"
            }


        self.symbol = st.sidebar.selectbox("Select a Stock Symbol:", list(stock_symbols.keys()))
        self.timeframe = st.sidebar.selectbox("Select Timeframe:", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])

        self.start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
        self.end_date = st.sidebar.date_input("Current Date", datetime.today())

        if st.button("Fetch & Predict"):
            self.fetch_stock_data()
            if self.df is None:
                return

            df_processed = self.process_data()
            if df_processed is None:
                return

            predicted_rf, mse_rf = self.apply_random_forest(df_processed)
            predicted_ma, mse_ma = self.apply_moving_average(df_processed)
            predicted_des, mse_des = self.apply_double_exponential_smoothing(df_processed)
            predicted_arima, mse_arima = self.apply_arima(df_processed)

            models = {
                "Random Forest": (predicted_rf, mse_rf),
                "Moving Average": (predicted_ma, mse_ma),
                "Double Exponential Smoothing": (predicted_des, mse_des),
                "ARIMA": (predicted_arima, mse_arima),
            }

            valid_models = {k: v for k, v in models.items() if v[1] is not None}

            if valid_models:
                best_model = min(valid_models, key=lambda k: valid_models[k][1])
                best_price = valid_models[best_model][0]
                best_mse = valid_models[best_model][1] # Get the best MSE value.
                # Styled output block
                st.markdown("""
                    <div style="
                        background: rgba(255, 255, 255, 0.2);
                        padding: 20px;
                        border-radius: 15px;
                        margin-top: 20px;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
                        backdrop-filter: blur(12px);
                        -webkit-backdrop-filter: blur(12px);
                    ">
                        <h4 style="margin-bottom: 10px;">✅ <u>Prediction Summary</u></h4>
                        <p><b>📈 Successfully fetched data for:</b> <span style="color: #007bff;">{symbol}</span></p>
                        <p><b>📅 Date Range:</b> {start} to {end}</p>
                        <p><b>🤖 Best Model:</b> <span style="color: #2b9348;">{model}</span></p>
                        <p><b>💰 Predicted Price:</b> <span style="color: #d00000;">${price:.2f}</span></p>
                        <p><b>📉 Model MSE:</b> {mse:.4f}</p>
                    </div>
                """.format(
                    symbol=self.symbol,
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d'),
                    model=best_model,
                    price=best_price,
                    mse=best_mse
                ), unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)


                # Generate OpenAI recommendation in same format
                openai_recommendation = self.generate_openai_recommendation(
                    self.symbol, 
                    self.start_date.strftime('%Y-%m-%d'),
                    self.end_date.strftime('%Y-%m-%d'),
                    best_model,
                    best_price,
                    best_mse,
                    self.timeframe
                )

                st.markdown("""
                    <div style="
                        margin-top: 20px;
                        padding: 15px;
                        border-left: 5px solid #4CAF50;
                        background-color: rgba(255,255,255,0.3);
                        border-radius: 10px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    ">
                    <h4>🧠 OpenAI's Recommendation</h4>
                    <p style="font-style: italic;">{}</p>
                    </div>
                """.format(openai_recommendation), unsafe_allow_html=True)




                recommendation = self.generate_recommendation(
                    self.symbol, 
                    self.start_date.strftime('%Y-%m-%d'),
                    self.end_date.strftime('%Y-%m-%d'),
                    best_model,
                    best_price,
                    best_mse,
                    self.timeframe
                )

                st.markdown("""
                    <div style="
                        margin-top: 30px;
                        padding: 15px;
                        border-left: 5px solid #2196F3;
                        background-color: rgba(255,255,255,0.3);
                        border-radius: 10px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    ">
                    <h4>🔍 Gemini AI's Recommendation</h4>
                    <p style="font-style: italic;">{}</p>
                    </div>
                """.format(recommendation), unsafe_allow_html=True)

                st.markdown("<br><br>", unsafe_allow_html=True)

                self.plot_predictions(df_processed, best_model)
            else:
                st.warning("Could not calculate MSE for any model. Please check the data and timeframe.")

if __name__ == "__main__":
    app = StockForecastModel()
    app.run()
    
    