import pandas as pd
import streamlit as st
from analytics import get_vol_surface_data, plot_2d_skew_by_expiry


st.title("Implied Volatility Skew")


# User inputs
with st.sidebar.form("controls"):
    ticker = st.text_input("Ticker Symbol", value="AMZN")
    max_days = st.slider("Max Days to Expiry", min_value=30, max_value=60, value=45)
    st.form_submit_button("Generate Plot")


with st.spinner("Fetching data..."):
    # Fetch data
    df = get_vol_surface_data(ticker, max_days=max_days)

    # Generate plot
    fig = plot_2d_skew_by_expiry(df, ticker)

    # Display in Streamlit
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

        # Build Strike vs. Expiration IV table
        st.subheader("Implied Volatility (%) by Strike & Expiration")
        pivot_df = (
            df[df["days_to_expiry"] >= 5]
            .drop_duplicates(subset=["days_to_expiry", "strike"])
            .pivot(index="strike", columns="days_to_expiry", values="implied_volatility")
            .sort_index()
        )
        pivot_df.columns = [f"{days} Days" for days in pivot_df.columns]
        pivot_df.index.name = "Strike"
        display_df = pivot_df.apply(
            lambda col: col.map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}%")
        )
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning(f"No options data available to plot for {ticker}.")
