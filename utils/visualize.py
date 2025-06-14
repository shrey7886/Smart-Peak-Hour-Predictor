import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from predict_tft import predict_peak_hours
import logging

logger = logging.getLogger(__name__)

def display_weather_strip(df, title="🌦️ Weather Snapshot"):
    """Display a weather information strip with current and forecasted weather."""
    st.subheader(title)
    
    # Get the last 10 weather records
    weather_df = df[["timestamp", "weather_main", "temp", "humidity", "rain", "wind_speed"]].tail(10).copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    
    # Format the display
    weather_df["Time"] = weather_df["timestamp"].dt.strftime("%H:%M")
    weather_df["Date"] = weather_df["timestamp"].dt.strftime("%Y-%m-%d")
    weather_df["Temperature"] = weather_df["temp"].round(1).astype(str) + "°C"
    weather_df["Humidity"] = weather_df["humidity"].round(0).astype(str) + "%"
    weather_df["Rain"] = weather_df["rain"].round(1).astype(str) + " mm"
    weather_df["Wind"] = weather_df["wind_speed"].round(1).astype(str) + " m/s"
    
    # Create a styled dataframe
    display_cols = ["Date", "Time", "weather_main", "Temperature", "Humidity", "Rain", "Wind"]
    styled_df = weather_df[display_cols].rename(columns={"weather_main": "Weather"})
    
    # Add weather icons
    weather_icons = {
        "Clear": "☀️",
        "Clouds": "☁️",
        "Rain": "🌧️",
        "Snow": "❄️",
        "Thunderstorm": "⛈️",
        "Drizzle": "🌦️",
        "Mist": "🌫️"
    }
    styled_df["Weather"] = styled_df["Weather"].map(lambda x: f"{weather_icons.get(x, '')} {x}")
    
    # Display with custom styling
    st.dataframe(
        styled_df,
        hide_index=True,
        use_container_width=True
    )

def display_holiday_info(df, title="📅 Upcoming Holidays"):
    """Display upcoming holiday information with enhanced visualization."""
    st.subheader(title)
    
    # Get unique holidays
    holidays_df = df[["timestamp", "holiday_type", "is_holiday", "holiday_name", "days_until_holiday"]].copy()
    holidays_df["timestamp"] = pd.to_datetime(holidays_df["timestamp"])
    holidays_df = holidays_df[holidays_df["is_holiday"] == 1].drop_duplicates(subset=["timestamp", "holiday_type"])
    
    if not holidays_df.empty:
        # Format the display
        holidays_df["Date"] = holidays_df["timestamp"].dt.strftime("%Y-%m-%d")
        holidays_df["Day"] = holidays_df["timestamp"].dt.strftime("%A")
        holidays_df["Days Until"] = holidays_df["days_until_holiday"].apply(
            lambda x: f"{x} days" if x > 0 else "Today!"
        )
        
        # Create a styled dataframe
        display_cols = ["Date", "Day", "Days Until", "holiday_type", "holiday_name"]
        styled_df = holidays_df[display_cols].rename(columns={
            "holiday_type": "Type",
            "holiday_name": "Holiday Name"
        })
        
        # Add holiday icons and colors
        holiday_icons = {
            "Holiday": "🎉",
            "Weekend": "🌅",
            "Special Event": "🎊",
            "Regular": "📅"
        }
        
        holiday_colors = {
            "Holiday": "background-color: #FFE4E1",  # Misty Rose
            "Weekend": "background-color: #E6E6FA",  # Lavender
            "Special Event": "background-color: #FFD700",  # Gold
            "Regular": "background-color: #F0F8FF"  # Alice Blue
        }
        
        # Apply styling
        styled_df["Type"] = styled_df["Type"].map(lambda x: f"{holiday_icons.get(x, '')} {x}")
        
        # Sort by date
        styled_df = styled_df.sort_values("Date")
        
        # Display with custom styling
        st.dataframe(
            styled_df.style.applymap(
                lambda x: holiday_colors.get(x.split()[1] if isinstance(x, str) else x, ""),
                subset=["Type"]
            ),
            hide_index=True,
            use_container_width=True
        )
        
        # Add holiday impact metrics
        st.markdown("### 📊 Holiday Impact")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_transactions = df[df["is_holiday"] == 1]["transactions"].mean()
            st.metric("Avg Transactions on Holidays", f"{avg_transactions:.1f}")
        
        with col2:
            special_impact = df[df["holiday_type"] == "Special Event"]["transactions"].mean()
            st.metric("Special Event Impact", f"{special_impact:.1f}")
        
        with col3:
            next_holiday = holidays_df[holidays_df["days_until_holiday"] > 0].iloc[0] if not holidays_df[holidays_df["days_until_holiday"] > 0].empty else None
            if next_holiday is not None:
                st.metric(
                    "Next Holiday",
                    f"{next_holiday['holiday_name']}",
                    f"in {next_holiday['days_until_holiday']} days"
                )
    else:
        st.info("No upcoming holidays in the forecast period.")
        
    # Add holiday preparation tips
    st.markdown("### 💡 Holiday Preparation Tips")
    tips = {
        "Special Event": "📈 Increase staff and inventory for special shopping events",
        "Holiday": "🎯 Plan for higher customer traffic and special promotions",
        "Weekend": "⏰ Adjust operating hours for weekend shopping patterns"
    }
    
    for holiday_type, tip in tips.items():
        st.markdown(f"- **{holiday_icons.get(holiday_type, '')} {holiday_type}**: {tip}")

def create_weather_impact_chart(df, title="🌡️ Weather Impact on Transactions"):
    """Create a chart showing weather impact on transactions."""
    st.subheader(title)
    
    # Prepare data
    weather_impact = df.groupby("weather_main").agg({
        "transactions": ["mean", "std", "count"]
    }).round(2)
    weather_impact.columns = ["Average Transactions", "Std Dev", "Count"]
    weather_impact = weather_impact.reset_index()
    
    # Create the chart
    fig = px.bar(
        weather_impact,
        x="weather_main",
        y="Average Transactions",
        error_y="Std Dev",
        color="weather_main",
        title="Average Transactions by Weather Condition",
        labels={
            "weather_main": "Weather Condition",
            "Average Transactions": "Average Number of Transactions"
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_holiday_impact_chart(df, title="📊 Holiday Impact on Transactions"):
    """Create a chart showing holiday impact on transactions."""
    st.subheader(title)
    
    # Prepare data
    holiday_impact = df.groupby("holiday_type").agg({
        "transactions": ["mean", "std", "count"]
    }).round(2)
    holiday_impact.columns = ["Average Transactions", "Std Dev", "Count"]
    holiday_impact = holiday_impact.reset_index()
    
    # Create the chart
    fig = px.bar(
        holiday_impact,
        x="holiday_type",
        y="Average Transactions",
        error_y="Std Dev",
        color="holiday_type",
        title="Average Transactions by Day Type",
        labels={
            "holiday_type": "Day Type",
            "Average Transactions": "Average Number of Transactions"
        }
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_weather_holiday_dashboard(df):
    """Display a comprehensive weather and holiday dashboard."""
    st.markdown("### 📊 Weather & Holiday Analytics")
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        create_weather_impact_chart(df)
    
    with col2:
        create_holiday_impact_chart(df)
    
    # Display weather strip
    display_weather_strip(df)
    
    # Display holiday information
    display_holiday_info(df)

def display_shop_dashboard_v2(df: pd.DataFrame, shop_id: str, shop_name: str, skip_predictions: bool = False) -> None:
    """Display the main shop dashboard with all metrics and visualizations.
    
    Args:
        df: DataFrame containing shop data
        shop_id: ID of the shop
        shop_name: Name of the shop
        skip_predictions: If True, skip prediction-related visualizations
    """
    try:
        if df is None or df.empty:
            st.warning("No data available for this shop")
            return

        # Display shop name
        st.title(f"📊 Dashboard - {shop_name}")

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        with col1:
            avg_transactions = df["transactions"].mean()
            st.metric("Average Transactions", f"{avg_transactions:.1f}")

        with col2:
            total_transactions = df["transactions"].sum()
            st.metric("Total Transactions", f"{total_transactions:,}")

        with col3:
            peak_hour = df.loc[df["transactions"].idxmax()]
            st.metric(
                "Peak Hour",
                f"{peak_hour['timestamp'].strftime('%H:%M')}",
                f"{peak_hour['transactions']} transactions"
            )

        with col4:
            current_hour = df.iloc[-1]
            st.metric(
                "Current Hour",
                f"{current_hour['timestamp'].strftime('%H:%M')}",
                f"{current_hour['transactions']} transactions"
            )

        # Display transaction history
        display_transaction_history(df)

        # Display weather information if available
        if 'weather_condition' in df.columns and 'temperature' in df.columns:
            display_weather_info(df)
        else:
            st.info("Weather data not available. Using mock data for predictions.")
            # Add mock weather data
            df = add_weather_features(df, use_api=False)

        # Display holiday information
        display_holiday_info(df)

        # Make predictions if not skipped
        if not skip_predictions:
            try:
                predictions = predict_peak_hours(df)
                if predictions is not None:
                    display_forecast(df, predictions)
                    display_peak_hours(df, predictions)
            except Exception as e:
                logger.error(f"Error making predictions: {str(e)}")
                st.error("Error making predictions. Please check the logs.")
        else:
            st.info("⚠️ Predictions are disabled. Please train the model to enable predictions.")

    except Exception as e:
        logger.error(f"Error in shop dashboard: {str(e)}")
        st.error("An error occurred while displaying the dashboard")

def display_forecast(df, predictions, title="🔮 Forecasted Transactions"):
    """Display forecasted transactions with weather and holiday information."""
    st.subheader(title)
    
    # Create a copy of predictions with formatted columns
    forecast_df = predictions.copy()
    forecast_df["Hour"] = forecast_df.index
    forecast_df["Predicted_Transactions"] = forecast_df["predictions"].round(2)
    
    # Add weather and holiday information
    forecast_df = forecast_df.merge(
        df[["timestamp", "weather_main", "holiday_type", "temp"]],
        left_index=True,
        right_on="timestamp",
        how="left"
    )
    
    # Format the display
    forecast_df["Weather"] = forecast_df["weather_main"]
    forecast_df["Holiday_Type"] = forecast_df["holiday_type"]
    forecast_df["Temperature"] = forecast_df["temp"].round(1).astype(str) + "°C"
    
    # Add suggestions based on predictions
    def get_suggestion(row):
        if row["Predicted_Transactions"] > 23:
            return f"📈 Add staff/stock! ({row['Weather']}, {row['Temperature']}) - {row['Holiday_Type']}"
        else:
            return f"✅ Normal ({row['Weather']}, {row['Temperature']}) - {row['Holiday_Type']}"
    
    forecast_df["Suggestion"] = forecast_df.apply(get_suggestion, axis=1)
    
    # Display the forecast
    display_cols = ["Hour", "Predicted_Transactions", "Weather", "Holiday_Type", "Suggestion"]
    st.dataframe(
        forecast_df[display_cols],
        hide_index=True,
        use_container_width=True
    )
    
    # Display peak hours
    st.markdown("### 🔥 Peak Hours")
    peak_hours = forecast_df[forecast_df["Predicted_Transactions"] > 23].sort_values(
        "Predicted_Transactions", ascending=False
    )
    
    for _, row in peak_hours.iterrows():
        st.markdown(f"🕒 Hour {row['Hour']} - {row['Weather']}, {row['Holiday_Type']}")

def display_weather_info(df, title="🌦️ Weather Information"):
    """Display detailed weather information."""
    st.subheader(title)
    
    # Get the latest weather data
    latest_weather = df[["timestamp", "weather_main", "temp", "humidity", "rain", "wind_speed"]].iloc[-1]
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Temperature", f"{latest_weather['temp']:.1f}°C")
    with col2:
        st.metric("Humidity", f"{latest_weather['humidity']:.0f}%")
    with col3:
        st.metric("Rain", f"{latest_weather['rain']:.1f} mm")
    with col4:
        st.metric("Wind Speed", f"{latest_weather['wind_speed']:.1f} m/s")
    
    # Display weather trend
    st.markdown("### 📈 Weather Trend")
    weather_trend = df[["timestamp", "temp", "humidity", "rain", "wind_speed"]].tail(24)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Temperature", "Humidity", "Rain", "Wind Speed"))
    
    fig.add_trace(
        go.Scatter(x=weather_trend["timestamp"], y=weather_trend["temp"], name="Temperature"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=weather_trend["timestamp"], y=weather_trend["humidity"], name="Humidity"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=weather_trend["timestamp"], y=weather_trend["rain"], name="Rain"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=weather_trend["timestamp"], y=weather_trend["wind_speed"], name="Wind Speed"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_peak_hours(df, predictions, title="🔥 Peak Hours Analysis"):
    """Display analysis of peak hours with insights."""
    st.subheader(title)
    
    # Get peak hours
    peak_hours = predictions[predictions["predictions"] > 23].copy()
    
    if not peak_hours.empty:
        # Calculate peak hour statistics
        total_peak_hours = len(peak_hours)
        avg_peak_transactions = peak_hours["predictions"].mean()
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Peak Hours", total_peak_hours)
        with col2:
            st.metric("Average Peak Transactions", f"{avg_peak_transactions:.1f}")
        
        # Analyze weather impact on peak hours
        st.markdown("### 🌦️ Weather Impact on Peak Hours")
        weather_impact = peak_hours.merge(
            df[["timestamp", "weather_main"]],
            left_index=True,
            right_on="timestamp",
            how="left"
        )
        
        weather_counts = weather_impact["weather_main"].value_counts()
        fig = px.pie(
            values=weather_counts.values,
            names=weather_counts.index,
            title="Peak Hours by Weather Condition"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display peak hour recommendations
        st.markdown("### 💡 Peak Hour Recommendations")
        for _, row in peak_hours.iterrows():
            st.markdown(f"""
            - **Hour {row.name}**: 
              - Expected Transactions: {row['predictions']:.1f}
              - Weather: {df.loc[row.name, 'weather_main']}
              - Holiday Type: {df.loc[row.name, 'holiday_type']}
              - Recommendation: {'Increase staff and inventory' if row['predictions'] > 25 else 'Monitor closely'}
            """)
    else:
        st.info("No peak hours detected in the forecast period.")

def display_transaction_history(df, title="📊 Transaction History"):
    """Display historical transaction data with trends."""
    st.subheader(title)
    
    # Create a time series plot of transactions
    fig = go.Figure()
    
    # Add transaction line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["transactions"],
            name="Transactions",
            line=dict(color="blue")
        )
    )
    
    # Add moving average
    window = 24  # 24-hour moving average
    df["moving_avg"] = df["transactions"].rolling(window=window).mean()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["moving_avg"],
            name=f"{window}-Hour Moving Average",
            line=dict(color="red", dash="dash")
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Transaction History with Moving Average",
        xaxis_title="Time",
        yaxis_title="Number of Transactions",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display transaction statistics
    st.markdown("### 📈 Transaction Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Transactions", f"{df['transactions'].mean():.1f}")
    with col2:
        st.metric("Maximum Transactions", f"{df['transactions'].max():.0f}")
    with col3:
        st.metric("Minimum Transactions", f"{df['transactions'].min():.0f}")
    with col4:
        st.metric("Standard Deviation", f"{df['transactions'].std():.1f}")
    
    # Display hourly distribution
    st.markdown("### 🕒 Hourly Distribution")
    hourly_avg = df.groupby(df["timestamp"].dt.hour)["transactions"].mean()
    
    fig = px.bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        title="Average Transactions by Hour",
        labels={"x": "Hour of Day", "y": "Average Transactions"}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True) 