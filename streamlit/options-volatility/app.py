import streamlit as st
from my_functions import get_vol_surface_data, plot_2d_skew_by_expiry


st.title("Options Volatility Surface Skew")


# User inputs
ticker = st.sidebar.text_input("Ticker Symbol", value="AMZN")
max_days = st.sidebar.slider("Max Days to Expiry", 30, 45, 60)


if st.button("Generate Plot"):
    with st.spinner("Fetching data..."):
        # Fetch data
        df = get_vol_surface_data(ticker, max_days=max_days)
        
        # Generate plot
        fig = plot_2d_skew_by_expiry(df, ticker)
        
        # Display in Streamlit
        st.pyplot(fig)