import pandas as pd
import streamlit as st
from alpaca_client import RateLimitError
from analytics import (
    get_multi_window_vrp,
    get_vol_surface_data,
    plot_2d_skew_by_expiry,
    plot_deltas_by_strike_and_expiry,
    plot_vrp_by_expiry,
)


# User inputs
with st.sidebar.form("controls"):
    ticker = st.text_input("Ticker Symbol", value="AMZN").strip().upper()
    max_days = st.slider("Max Days to Expiry", min_value=30, max_value=60, value=45)
    st.form_submit_button("Generate Plot")


with st.spinner("fetching data..."):
    try:
        # Fetch data
        df = get_vol_surface_data(ticker, max_days=max_days)
        vrp_df = get_multi_window_vrp(ticker, max_days=max_days)

        # Generate plots
        fig = plot_2d_skew_by_expiry(df, ticker)
        delta_fig = plot_deltas_by_strike_and_expiry(df, ticker)
        vrp_fig = plot_vrp_by_expiry(vrp_df, ticker)
    except RateLimitError as e:
        st.warning(str(e))
        st.stop()

    # Display in Streamlit
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

        if delta_fig is not None:
            st.plotly_chart(delta_fig, use_container_width=True)

        if vrp_fig is not None:
            st.plotly_chart(vrp_fig, use_container_width=True)

        # Color rows based on whether strike is an ATM/OTM call (>= atm_strike) or OTM put (< atm_strike)
        spot_price = (df["strike"] / df["moneyness"]).iloc[0]
        atm_strike = df.loc[(df["strike"] - spot_price).abs().idxmin(), "strike"]

        def highlight_by_spot(row):
            color = (
                "background-color: rgba(46, 204, 113, 0.15)"
                if row.name >= atm_strike
                else "background-color: rgba(231, 76, 60, 0.15)"
            )
            return [color] * len(row)

        # Tabbed tables per expiration date (all corresponding to max_days slider)
        expiries = sorted(df["expiration_date"].unique())
        tabs = st.tabs(expiries)

        for tab, expiry in zip(tabs, expiries):
            with tab:
                exp_df = (
                    df[df["expiration_date"] == expiry]
                    .drop_duplicates(subset=["strike"])
                    .sort_values("strike")
                    .set_index("strike")
                )
                exp_df.index.name = "Strike"

                display_df = pd.DataFrame(
                    {
                        "Delta": exp_df["delta"].map(
                            lambda x: "NA" if pd.isna(x) else f"{x:.3f}"
                        ),
                        "Price": exp_df["price"].map(
                            lambda x: "NA" if pd.isna(x) else f"${x:.2f}"
                        ),
                    },
                    index=exp_df.index,
                )

                st.dataframe(
                    display_df.style.apply(highlight_by_spot, axis=1),
                    use_container_width=True,
                )
    else:
        st.warning(f"no options data available to plot for {ticker}.")
