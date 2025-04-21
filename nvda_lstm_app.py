import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns  # For better plot aesthetics

st.set_page_config(page_title="📈 Stock Market Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

st.markdown("""
### 🌟 Model Objective
The goal of this app is to predict future closing prices of any stock, for swing zone trading using historical data with the help of Deep Learning using an LSTM (Long Short-Term Memory) neural network model. It provides:
- Stock data insights
- Closing Price predictions for the next 7 and 30 days using LSTM
- Future Trend analysis of any stock using Moving Averages (MA100 VS MA200)
""")

ticker = st.text_input("🔍 Enter any stock ticker symbol (e.g., NVDA, AAPL):", placeholder="Enter Stock Ticker")

df = None
selected_day_pred = None
selected_date = None
days_ahead = None
avg_lstm_full = None

if ticker:
    end_date = datetime.datetime.now()
    start_date = datetime.datetime(2015, 1, 1)  # Set start date to 2015-01-01
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ No data found for this ticker.")
    else:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df['Volume'] = df['Volume'] / 1e6
        df['Turnover'] = (df['Close'] * df['Volume']) / 1e6

        # Format the index (Date column) to dd-mm-yy
        df.index = pd.to_datetime(df.index)  # Ensure the index is a datetime object

# Add a 'Date' column formatted as dd-mm-yy for display (without changing the index)
        df['Date'] = df.index.strftime('%d-%m-%Y')

# Use the original datetime index to extract the 'Year'
        df['Year'] = df.index.year

# Set 'Date' as the display column, not the index
        df.set_index('Date', inplace=True)

        df_display = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_display['Turnover'] = (df_display['Close'] * df_display['Volume'] * 1e6) / 1e9

        df_display['Open'] = df_display['Open'].map(lambda x: f"${x:.2f}")
        df_display['High'] = df_display['High'].map(lambda x: f"${x:.2f}")
        df_display['Low'] = df_display['Low'].map(lambda x: f"${x:.2f}")
        df_display['Close'] = df_display['Close'].map(lambda x: f"${x:.2f}")
        df_display['Volume'] = df_display['Volume'].map(lambda x: f"${x:.2f}M")
        df_display['Turnover'] = df_display['Turnover'].map(lambda x: f"${x:.2f}B")

        st.subheader("📊 Stock Data (Year 2015-2025)")
        st.dataframe(df_display.iloc[::-1]) # Display in descending order

        st.markdown("""
#### 📘 Data Explanation:
- **Data Source:** All stock data is fetched from [Yahoo Finance](https://finance.yahoo.com/) using the `yfinance` Python library.
- **Open:** Price at which the stock opened trading on a particular day.
- **Close:** Final price of the stock when the market closed.
- **High/Low:** Maximum and minimum prices during the day.
- **Volume:** Number of shares traded during the day (in millions).
- **Turnover:** Total traded value = Close × Volume (in billions).
""")

        if len(df) >= 200:
            df['MA100'] = df['Close'].rolling(100).mean()
            df['MA200'] = df['Close'].rolling(200).mean()

            ma_signal = "📈 Bullish (Golden Cross)" if df['MA100'].iloc[-1] > df['MA200'].iloc[-1] else "🔴 Bearish (Death Cross)"
            st.subheader("📉 Moving Averages (MA100 vs MA200) Analysis:")
            st.write(f"Latest MA100: {df['MA100'].iloc[-1]:.2f}, MA200: {df['MA200'].iloc[-1]:.2f}")
            st.markdown(ma_signal)
        else:
            st.warning("⚠️ Not enough data to compute MA100 and MA200 (need at least 200 records).")

        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        look_back = 100
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=64, verbose=0)

        predictions = model.predict(X, verbose=0)
        predictions_rescaled = scaler.inverse_transform(predictions)
        original_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

        # Ensure the index is a datetime object (DatetimeIndex)
        df.index = pd.to_datetime(df.index, format='%d-%m-%Y')

# Now we can safely access the year
        df['Year'] = df.index.year

        avg_close_per_year = df.groupby('Year')['Close'].mean()
        avg_actual = float(avg_close_per_year.mean())
        avg_lstm_full = np.mean(predictions_rescaled)
        model_accuracy = 100 - abs(avg_actual - avg_lstm_full) / avg_actual * 100

        inputs = scaled_data[-look_back:]
        inputs = inputs.reshape(1, look_back, 1)
        preds = []
        for _ in range(7):
            pred = model.predict(inputs, verbose=0)
            preds.append(pred[0, 0])
            inputs = np.concatenate([inputs[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

        predicted_7_days = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        avg_next_7 = np.mean(predicted_7_days)

        st.subheader("📈 7-Day Forecast")
        st.markdown(f"**Predicted average price for next 7 days**: `${avg_next_7:.2f}`")

        st.subheader("🗕️ Predict Stock's Price for any Particular Date Within Next 30 Days")
        today = datetime.date.today()
        future_date = st.date_input("Select a date within the next 30 days:", min_value=today + datetime.timedelta(1), max_value=today + datetime.timedelta(30))

        if future_date:
            delta = (future_date - today).days
            inputs = scaled_data[-look_back:].reshape(1, look_back, 1)
            for _ in range(delta):
                pred = model.predict(inputs, verbose=0)
                inputs = np.concatenate([inputs[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

            predicted_day = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
            selected_day_pred = predicted_day
            selected_date = future_date
            st.markdown(f"**Predicted average price for {future_date}**: `${predicted_day:.2f}`")

        st.subheader("📉 Price Trend Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-len(original_rescaled):], original_rescaled, label="Actual Price", color='cyan')
        ax.plot(df.index[-len(predictions_rescaled):], predictions_rescaled, label="Model Prediction", color='orange')
        ax.set_title(f"{ticker.upper()} Closing Price vs LSTM Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 📊 Moving Averages (MA100 vs MA200) Summary:")
        signal = "📈 Golden Cross (Potential Bullish Trend)" if df['MA100'].iloc[-1] > df['MA200'].iloc[-1] else "📉 Death Cross (Potential Bearish Trend)"
        st.write(signal)

        st.markdown("### 🔮 Predictions")
        st.write(f"**Average price of Stock for next 7 days:** ${avg_next_7:.2f}")
        if selected_day_pred:
            st.write(f"**Predicted price on {selected_date}:** ${selected_day_pred:.2f}")

        st.subheader("✅ Estimated Model Accuracy (10-Year Historical Match)")
        st.write("**Method Used:** Average of each year's closing average (2015–2024)")
        st.write(f"**Model predicted 10-year avg closing price:** `${avg_lstm_full:.2f}`")
        st.write(f"**Actual 10-year avg closing price:** `${avg_actual:.2f}`")
        st.success(f"📊 **Accuracy**: `{model_accuracy:.2f}%`")

#         st.subheader("📋 Statistical Summary (10-Year Data)")
#         numerical_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']]
#         desc_stats = {
#             'Mean': numerical_data.mean(),
#             'Median': numerical_data.median(),
#             'Mode': numerical_data.mode().iloc[0],
#             'Std Dev': numerical_data.std(),
#             'Range': numerical_data.max() - numerical_data.min(),
#             'IQR': numerical_data.quantile(0.75) - numerical_data.quantile(0.25)
#         }
#         stats_df = pd.DataFrame(desc_stats)
#         st.dataframe(stats_df)

#         st.subheader("🔹 Importance of Statistical Metrics")
#         st.markdown("""
# - **Mean:** Average stock value over time.
# - **Median:** Middle value; less affected by outliers.
# - **Mode:** Most frequent value in dataset.
# - **Standard Deviation:** Indicates volatility or price fluctuation.
# - **Range:** Spread between the highest and lowest values.
# - **IQR (Interquartile Range):** Spread of the middle 50% of data, highlighting core variability.
# """)

#         # Skewness Plot
#         st.subheader("📈 Skewness Analysis")
#         skew_vals = numerical_data['Close'].skew()
#         fig_skew, ax_skew = plt.subplots()
#         sns.histplot(numerical_data['Close'], kde=True, ax=ax_skew)
#         ax_skew.axvline(numerical_data['Close'].mean().item(), color='k', linestyle='dashed', linewidth=1, label=f'Mean: {numerical_data["Close"].mean().item():.2f}')
#         ax_skew.axvline(numerical_data['Close'].median().item(), color='r', linestyle='dashed', linewidth=1, label=f'Median: {numerical_data["Close"].median().item():.2f}')
#         ax_skew.set_title(f"Skewness of {ticker.upper()} Closing Price (Skew: {float(skew_vals):.2f})")
#         ax_skew.set_xlabel("Closing Price")
#         ax_skew.set_ylabel("Frequency")
#         ax_skew.legend()
#         st.pyplot(fig_skew)
#         if skew_vals.item() > 0:
#             st.info("Positive Skew: The distribution has a long tail extending to the right, suggesting more frequent smaller gains and occasional large gains (potentially bullish).")
#         elif skew_vals.item() < 0:
#             st.info("Negative Skew: The distribution has a long tail extending to the left, suggesting more frequent smaller losses and occasional large losses (potentially bearish).")
#         else:
#             st.info("Approximately Symmetric Distribution.")

#         # Kurtosis Plot
#         st.subheader("📊 Kurtosis Analysis")
#         kurt_vals = numerical_data['Close'].kurtosis()
#         fig_kurt, ax_kurt = plt.subplots()
#         sns.kdeplot(numerical_data['Close'], ax=ax_kurt, fill=True)
#         peak = numerical_data['Close'].mode().iloc[0]
#         ax_kurt.axvline(peak.item(), color='g', linestyle='dashed', linewidth=1, label=f'Mode: {peak.item():.2f}')
#         ax_kurt.set_title(f"Kurtosis of {ticker.upper()} Closing Price (Kurtosis: {kurt_vals.item():.2f})")
#         ax_kurt.set_xlabel("Closing Price")
#         ax_kurt.set_ylabel("Density")
#         ax_kurt.legend()
#         st.pyplot(fig_kurt)
#         if kurt_vals.item() > 0:
#             st.info("Leptokurtic (High Kurtosis): The distribution has heavier tails and a sharper peak than a normal distribution, indicating a higher         probability of extreme price movements.")
#         elif kurt_vals.item() < 0:
#             st.info("Platykurtic (Low Kurtosis): The distribution has thinner tails and a flatter peak than a normal distribution, indicating a lower         probability of extreme price movements.")            
#         else:
#             st.info("Mesokurtic (Kurtosis close to 0): Similar tail and peak characteristics to a normal distribution.")

#         st.subheader("🧠 Final Conclusion")
#         labels = ['Bullish', 'Bearish', 'Neutral']
#         if df['MA100'].iloc[-1] > df['MA200'].iloc[-1]:
#             sizes = [60, 20, 20]
#         elif df['MA100'].iloc[-1] < df['MA200'].iloc[-1]:
#             sizes = [20, 60, 20]
#         else:
#             sizes = [33.3, 33.3, 33.3]

#         fig3, ax3 = plt.subplots()
#         ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#F44336', '#FFC107'])
#         ax3.axis('equal')
#         st.pyplot(fig3)