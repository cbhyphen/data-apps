from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import make_interp_spline

from alpaca_client import RateLimitError, fetch_raw_options_chain, get_latest_price


def get_delta_next_fridays(max_days: int = 60) -> list[int]:
    """
    get friday deltas from today between 5 and max_days out
    """
    today = datetime.today().date()
    return [
        days
        for days in range(5, max_days + 1)
        if (today + timedelta(days=days)).weekday() == 4
    ]


def get_vol_surface_data(
    ticker_symbol: str,
    max_days: int = 60,
    moneyness_bound: float = 0.30,
) -> pd.DataFrame:
    """
    fetch option data for next 'max_days',
    collect IV for OTM puts and calls,
    and limit strikes by moneyness.
    """
    surface_data = []
    underlying_price = get_latest_price(ticker_symbol)
    allowed_friday_windows = get_delta_next_fridays(max_days)

    for days_to_expiry in allowed_friday_windows:
        try:
            chain_data = fetch_raw_options_chain(
                ticker_symbol,
                days_target=days_to_expiry,
                underlying_price=underlying_price,
                moneyness_bound=moneyness_bound,
            )
            calls = chain_data['calls']
            puts = chain_data['puts']
            closest_expiry = chain_data['closest_expiry_str']
        except RateLimitError:
            raise
        except Exception as e:
            print(f"could not fetch data for target {days_to_expiry} days via Alpaca: {e}")
            continue

        if puts.empty or calls.empty:
            continue

        # combine OTM puts (strikes <= spot) and OTM calls (strikes > spot)
        otm_options = pd.concat([
            puts[puts['strike'] <= underlying_price],
            calls[calls['strike'] > underlying_price],
        ])

        for _, row in otm_options.iterrows():
            iv = row.get('impliedVolatility', 0)
            if pd.isna(iv) or iv <= 0.001:
                continue

            surface_data.append({
                'days_to_expiry': days_to_expiry,
                'expiration_date': closest_expiry,
                'strike': row['strike'],
                'moneyness': row['strike'] / underlying_price,
                'implied_volatility': iv * 100,
            })

    return pd.DataFrame(surface_data)


def plot_2d_skew_by_expiry(df: pd.DataFrame, ticker: str, max_lines: int = 8):
    """
    plot 2D IV skew across expiry
    """
    if df.empty or 'days_to_expiry' not in df.columns:
        print("no data available to plot !!!")
        return None

    # fetch spot price for the vertical line
    underlying_price = get_latest_price(ticker)

    expiries = sorted(df['days_to_expiry'].unique())[:max_lines]

    fig = go.Figure()
    colors = px.colors.sequential.Plasma_r

    for i, days in enumerate(expiries):
        df_exp = df[df['days_to_expiry'] == days].sort_values(by='strike').drop_duplicates(subset=['strike'])

        # safely map to color scale
        color_idx = int((i / max(1, len(expiries) - 1)) * (len(colors) - 1))
        color = colors[color_idx]

        x = df_exp['strike'].values
        y = df_exp['implied_volatility'].values

        if len(x) > 3:
            # spline interpolation for smooth curve
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)

            # add smooth line
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth,
                mode='lines',
                line=dict(color=color, width=2.5),
                name=f'{days} Days',
                legendgroup=f'{days}'
            ))
            # add actual data points
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=color, size=6, opacity=0.5),
                legendgroup=f'{days}',
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # fallback if not enough points for spline
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                line=dict(color=color, width=2.5),
                name=f'{days} Days'
            ))

    # vertical line for spot price
    fig.add_vline(x=underlying_price, line_dash="dash", line_color="red",
                  annotation_text=f"Spot: ${underlying_price:.2f}",
                  annotation_position="top right")

    # layout config
    fig.update_layout(
        title=f"{ticker} IV Skew by Expiration",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        hovermode="x unified",
        template="plotly_dark",  #  'plotly_white'
        legend=dict(title="Days to Expiry", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    return fig
