# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Commodity Price Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Commodity Price Index Dashboard (2012–2023)")
st.caption("Interactive analysis and forecasting of monthly price indices")

# === DATA LOADING & CLEANING ===
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("ADS Assignemnt dataset.csv")
    
    # Extract date range from column names
    start_date = pd.to_datetime("2012-04")
    n_months = len(df.columns) - 3
    dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
    
    # Clean price data
    price_df = df.iloc[:, 3:].copy()
    price_df.columns = dates.strftime('%Y-%m')
    price_df = price_df.replace(0.0, np.nan)
    price_df = price_df.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    
    metadata = df.iloc[:, :3].copy()
    metadata.columns = ['COMM_NAME', 'COMM_CODE', 'COMM_WT']
    metadata['COMM_WT'] = metadata['COMM_WT'].fillna(metadata['COMM_WT'].median())
    
    return metadata, price_df, dates

metadata, price_df, dates = load_and_clean_data()

# === SIDEBAR FILTERS ===
st.sidebar.header("🔍 Filters")
selected_commodities = st.sidebar.multiselect(
    "Select Commodities",
    options=metadata['COMM_NAME'].tolist(),
    default=metadata.nlargest(5, 'COMM_WT')['COMM_NAME'].tolist()[:3]
)

forecast_item = st.sidebar.selectbox(
    "Forecast Item",
    options=metadata['COMM_NAME'].tolist(),
    index=0
)

forecast_horizon = st.sidebar.slider("Forecast Horizon (months)", min_value=6, max_value=24, value=12)

# === MAIN DASHBOARD ===

# Section 1: Overview Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Commodities", len(metadata))
col2.metric("Time Periods", len(dates))
col3.metric("Avg. Weight", f"{metadata['COMM_WT'].mean():.4f}")
col4.metric("Max Price Index", f"{price_df.max().max():.1f}")

# Section 2: Historical Trends
st.subheader("📈 Historical Price Trends")
if selected_commodities:
    fig = go.Figure()
    for name in selected_commodities:
        idx = metadata[metadata['COMM_NAME'] == name].index[0]
        series = price_df.iloc[idx]
        fig.add_trace(go.Scatter(
            x=dates, y=series,
            mode='lines',
            name=name,
            line=dict(width=2)
        ))
    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Price Index",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one commodity to view trends.")

# Section 3: Top Commodities by Weight
st.subheader("⚖️ Top 10 Commodities by Weight")
top10 = metadata.nlargest(10, 'COMM_WT').copy()
top10['Weight %'] = (top10['COMM_WT'] / top10['COMM_WT'].sum()) * 100

fig2 = px.bar(
    top10,
    x='Weight %',
    y='COMM_NAME',
    orientation='h',
    text_auto='.2f',
    title="Top 10 Commodities by Weight Contribution",
    labels={'Weight %': 'Weight (%)', 'COMM_NAME': 'Commodity'}
)
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# Section 4: Forecasting
st.subheader(f"🔮 Forecast for: {forecast_item}")
idx = metadata[metadata['COMM_NAME'] == forecast_item].index[0]
series = price_df.iloc[idx]

# Prepare Prophet data
prophet_df = pd.DataFrame({
    'ds': dates,
    'y': series.values
})
model = Prophet(yearly_seasonality=True, interval_width=0.95)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
forecast = model.predict(future)

# === FIXED: Create Plotly forecast plot ===
fig3 = go.Figure()

# Historical data
fig3.add_trace(go.Scatter(
    x=forecast['ds'][:len(dates)],
    y=forecast['yhat'][:len(dates)],
    mode='markers',
    name='Historical Data',
    marker=dict(color='blue', size=6, opacity=0.7)
))

# Forecast trend line
fig3.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Predicted Trend',
    line=dict(color='red', width=3)
))

# Confidence interval (shaded area)
fig3.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    fill=None
))
fig3.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_lower'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.1)',
    showlegend=False
))

fig3.update_layout(
    title=f"{forecast_horizon}-Month Forecast for {forecast_item}",
    xaxis_title="Date",
    yaxis_title="Price Index",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig3, use_container_width=True)

# Show forecast table
st.write(f"📌 Next {forecast_horizon} months forecast (Predicted Trend ± 95% CI):")
forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
forecast_table['ds'] = forecast_table['ds'].dt.strftime('%Y-%m')
forecast_table.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
st.dataframe(forecast_table.round(2), use_container_width=True)

# Section 5: Download Options
st.subheader("📥 Export Data")
col_a, col_b = st.columns(2)

with col_a:
    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=price_df.to_csv(index=False),
        file_name="cleaned_commodity_prices.csv",
        mime="text/csv"
    )

with col_b:
    st.download_button(
        label="Download Forecast (CSV)",
        data=forecast_table.to_csv(index=False),
        file_name=f"forecast_{forecast_item.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Dashboard built with 🐍 Streamlit | Data: Commodity Price Indices (2012–2023)")