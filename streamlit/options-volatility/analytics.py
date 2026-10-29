from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from alpaca_client import fetch_raw_options_chain, get_latest_price


# get next friday deltas from today that are at least 5 days out
def get_delta_next_fridays(n=6):

    today = datetime.today().date()
    next_fridays_days = []
    current_date = today

    while len(next_fridays_days) < n:
        current_date += timedelta(days=1)
        days_to_target = (current_date - today).days
        if current_date.weekday() == 4 and days_to_target >= 5:
            next_fridays_days.append(days_to_target)

    return next_fridays_days


# fetch option data for next 'max days',
# collect IV for OTM puts and calls,
# and limits strikes by moneyness.


def get_vol_surface_data(
    ticker_symbol: str,
    max_days: int = 60,
    risk_free_rate: float = 0.045,
    min_moneyness: float = 0.7,
    max_moneyness: float = 1.3
    ):

    today = datetime.today()
    surface_data = []

    # get friday target windows based on max_days
    num_fridays = (max_days // 7) + 2
    allowed_friday_windows = get_delta_next_fridays(num_fridays)

    for days_to_expiry in allowed_friday_windows:
        if days_to_expiry < 5 or days_to_expiry > max_days:
            continue

        try:
            chain_data = fetch_raw_options_chain(
                ticker_symbol, days_target=days_to_expiry
                )
            underlying_price = chain_data['underlying_price']
            t_years = chain_data['t_years']
            calls = chain_data['calls']
            puts = chain_data['puts']
            closest_expiry = chain_data['closest_expiry_str']
        except Exception as e:
            print(f"Could not fetch data for target {days_to_expiry} days via Alpaca: {e}")
            continue

        if puts.empty or calls.empty:
            continue

        # normalize column names
        if 'strike_price' in puts.columns and 'strike' not in puts.columns:
            puts = puts.rename(columns={'strike_price': 'strike'})
            calls = calls.rename(columns={'strike_price': 'strike'})
        if 'implied_volatility' in puts.columns and 'impliedVolatility' not in puts.columns:
            puts = puts.rename(columns={'implied_volatility': 'impliedVolatility'})
            calls = calls.rename(columns={'implied_volatility': 'impliedVolatility'})

        # filter and compute puts (strikes <= spot)
        filtered_puts = puts[(puts['strike'] <= underlying_price) & (puts['strike'] >= underlying_price * min_moneyness)]
        for _, row in filtered_puts.iterrows():
            iv = row.get('impliedVolatility', 0)
            if pd.isna(iv) or iv <= 0.001: continue

            surface_data.append({
                'days_to_expiry': days_to_expiry,
                'strike': row['strike'],
                'moneyness': row['strike'] / underlying_price,
                'implied_volatility': iv * 100
            })

        # filter and compute calls (strikes > spot)
        filtered_calls = calls[(calls['strike'] > underlying_price) & (calls['strike'] <= underlying_price * max_moneyness)]
        for _, row in filtered_calls.iterrows():
            iv = row.get('impliedVolatility', 0)
            if pd.isna(iv) or iv <= 0.001: continue

            surface_data.append({
                'days_to_expiry': days_to_expiry,
                'strike': row['strike'],
                'moneyness': row['strike'] / underlying_price,
                'implied_volatility': iv * 100
            })

    return pd.DataFrame(surface_data)


def plot_2d_skew_by_expiry(df: pd.DataFrame, ticker: str, max_lines: int = 8):
    """
    Plot 2D IV skew across expiry
    """

    if df.empty or 'days_to_expiry' not in df.columns:
        print("no data available to plot !!!")
        return None

    df = df[df['days_to_expiry'] >= 5].copy()
    if df.empty:
        print("no data available to plot !!!")
        return None

    fig = plt.figure(figsize=(10, 6))

    # fetch spot price for the vertical line
    underlying_price = get_latest_price(ticker)

    expiries = sorted(df['days_to_expiry'].unique())[:max_lines]
    num_expiries = len(expiries)
    colormap = plt.cm.plasma

    for i, days in enumerate(expiries):
        df_exp = df[df['days_to_expiry'] == days].sort_values(by='strike').drop_duplicates(subset=['strike'])
        color_idx = 0.85 - (i / max(1, num_expiries - 1)) * 0.75
        color = colormap(color_idx)

        x = df_exp['strike'].values
        y = df_exp['implied_volatility'].values

        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            plt.plot(x_smooth, y_smooth, color=color, linewidth=1.8, label=f'{days} Days')
            plt.plot(x, y, marker='o', color=color, linestyle='None', alpha=0.4, markersize=4)
        else:
            plt.plot(x, y, marker='o', color=color, linewidth=1.8, label=f'{days} Days')

    plt.axvline(x=underlying_price, color='red', linestyle='--',
                linewidth=1.5, label=f'spot price (${underlying_price:.2f})'
                )
    plt.title(f"{ticker} IV Skew by Expiration")
    plt.xlabel('strike')
    plt.ylabel('IV (%)')
    plt.legend(title='days to expiry', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig
