import pandas as pd
import streamlit as st
from alpaca_client import RateLimitError
from analytics import get_vol_surface_data, plot_2d_skew_by_expiry


# User inputs
with st.sidebar.form("controls"):
    ticker = st.text_input("Ticker Symbol", value="AMZN")
    max_days = st.slider("Max Days to Expiry", min_value=30, max_value=60, value=45)
    st.form_submit_button("Generate Plot")


with st.spinner("fetching data..."):
    try:
        # Fetch data
        df = get_vol_surface_data(ticker, max_days=max_days)

        # Generate plot
        fig = plot_2d_skew_by_expiry(df, ticker)
    except RateLimitError as e:
        st.warning(str(e))
        st.stop()

    # Display in Streamlit
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

        # Build Strike vs. Expiration Date IV table
        pivot_df = (
            df.drop_duplicates(subset=["expiration_date", "strike"])
            .pivot(index="strike", columns="expiration_date", values="implied_volatility")
            .sort_index()
        )
        pivot_df.index.name = "Strike"
        display_df = pivot_df.apply(
            lambda col: col.map(lambda x: "NA" if pd.isna(x) else f"{x:.2f}%")
        )

        # Color rows based on whether strike is above or at/below spot price
        spot_price = (df["strike"] / df["moneyness"]).iloc[0]

        def highlight_by_spot(row):
            color = (
                "background-color: rgba(46, 204, 113, 0.15)"
                if row.name > spot_price
                else "background-color: rgba(231, 76, 60, 0.15)"
            )
            return [color] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_by_spot, axis=1),
            use_container_width=True,
        )
    else:
        st.warning(f"no options data available to plot for {ticker}.")
