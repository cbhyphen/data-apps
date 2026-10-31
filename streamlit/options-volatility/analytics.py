from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import make_interp_spline

from alpaca_client import (
    RateLimitError,
    fetch_raw_options_chain,
    get_alpaca_ohlcv,
    get_latest_price,
)


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


def compute_yang_zhang(df, window, periods_per_year=252):
    """
    compute Yang-Zhang realized volatility estimator on daily OHLC data
    """

    o = pd.Series(df['open'].astype(float).values, index=df.index)
    h = pd.Series(df['high'].astype(float).values, index=df.index)
    l = pd.Series(df['low'].astype(float).values, index=df.index)
    c = pd.Series(df['close'].astype(float).values, index=df.index)

    c_prev = c.shift(1)

    log_oc = np.log(o / c_prev)
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    rs = log_ho * log_hc + log_lo * log_lc

    if window == 1:
        var_overnight = log_oc ** 2
        var_open_to_close = log_co ** 2
        var_rs = rs
        k = 0
    else:
        var_overnight = log_oc.rolling(window=window, min_periods=2).var(ddof=1)
        var_open_to_close = log_co.rolling(window=window, min_periods=2).var(ddof=1)
        var_rs = rs.rolling(window=window, min_periods=1).mean()
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_var = var_overnight + k * var_open_to_close + (1 - k) * var_rs
    latest_var = yz_var.iloc[-1]
    latest_vol = np.sqrt(latest_var * periods_per_year)

    return float(latest_vol)


def get_multi_window_vrp(
    ticker_symbol: str,
    max_days: int = 60,
    windows: list[int] | None = None,
    use_min_rv: bool = True,
    min_rv_window: int = 10,
    risk_free_rate: float = 0.045,
) -> pd.DataFrame:
    """
    calculate ATM VRP across multiple time frames.
    note on VRP is "ex-ante" (forward-looking snapshot) because
    looking at current option chains, not delta-hedged historical path.
    """
    if windows is None:
        windows = get_delta_next_fridays(max_days)

    max_window = max(max(windows), min_rv_window if use_min_rv else 0)
    lookback_days = int(max_window * 3)
    df_ohlc = get_alpaca_ohlcv(ticker_symbol, lookback_days)
    underlying_price = get_latest_price(ticker_symbol)

    results = []

    for w in windows:
        rv_window = max(min_rv_window, w) if use_min_rv else w
        try:
            rv_estimate = compute_yang_zhang(df_ohlc, window=rv_window)
        except Exception as e:
            print(f"Error calculating YZ RV for window {rv_window}: {e}")
            rv_estimate = np.nan

        try:
            chain_data = fetch_raw_options_chain(
                ticker_symbol,
                days_target=w,
                underlying_price=underlying_price,
            )
            spot = chain_data['underlying_price']
            t = chain_data['t_years']
            calls = chain_data['calls']
            puts = chain_data['puts']
            closest_expiry = chain_data['closest_expiry_str']
            actual_days = int(round(t * 365))

            # drop rows with missing delta or implied volatility
            calls = calls.dropna(subset=['delta', 'impliedVolatility']).copy()
            puts = puts.dropna(subset=['delta', 'impliedVolatility']).copy()

            # ATM IV
            atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]].iloc[0]
            atm_iv = atm_call['impliedVolatility']

        except RateLimitError:
            raise
        except Exception as e:
            print(f"Error fetching options for target window {w}d: {e}")
            closest_expiry = None
            actual_days = np.nan
            atm_iv = np.nan

        rv_pct = rv_estimate * 100 if not np.isnan(rv_estimate) else np.nan
        atm_iv_pct = atm_iv * 100 if not np.isnan(atm_iv) else np.nan

        results.append({
            'Target_Days': w,
            'Actual_Days': actual_days,
            'RV_%': rv_pct,
            'ATM_IV_%': atm_iv_pct,
            'ATM_VRP_%': atm_iv_pct - rv_pct,
        })

    return pd.DataFrame(results).round(2)


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


def plot_vrp_by_expiry(multi_vrp_df: pd.DataFrame, ticker: str):
    """
    plot ATM IV, Yang-Zhang RV, and VRP across target days to expiry
    """
    if multi_vrp_df.empty or 'ATM_VRP_%' not in multi_vrp_df.columns:
        print("no VRP data available to plot !!!")
        return None

    multi_vrp_df = multi_vrp_df.dropna(subset=['ATM_IV_%', 'RV_%', 'ATM_VRP_%'])
    if multi_vrp_df.empty:
        print("no VRP data available to plot !!!")
        return None

    fig = go.Figure()

    # VRP as bar plot
    vrp_colors = [
        'rgba(46, 204, 113, 0.6)' if val >= 0 else 'rgba(231, 76, 60, 0.6)'
        for val in multi_vrp_df['ATM_VRP_%']
    ]
    fig.add_trace(go.Bar(
        x=multi_vrp_df['Target_Days'],
        y=multi_vrp_df['ATM_VRP_%'],
        name='ATM VRP (%)',
        marker_color=vrp_colors,
        width=2,
    ))

    # line plot ATM IV
    fig.add_trace(go.Scatter(
        x=multi_vrp_df['Target_Days'],
        y=multi_vrp_df['ATM_IV_%'],
        mode='lines+markers',
        name='ATM IV (%)',
        line=dict(color='dodgerblue', width=2),
        marker=dict(symbol='circle', size=8),
    ))

    # line plot RV
    fig.add_trace(go.Scatter(
        x=multi_vrp_df['Target_Days'],
        y=multi_vrp_df['RV_%'],
        mode='lines+markers',
        name='RV (%)',
        line=dict(color='chocolate', width=2),
        marker=dict(symbol='square', size=8),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

    fig.update_layout(
        title=f"{ticker} VRP by Days to Expiry",
        xaxis_title="Days to Expiry",
        yaxis_title="Percent (%)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig
