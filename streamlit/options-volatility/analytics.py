import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from alpaca_client import (
    RateLimitError,
    fetch_raw_options_chain,
    get_alpaca_ohlcv,
    get_latest_price,
)
from utils import compute_yang_zhang, get_delta_next_fridays


def get_vol_surface_data(
    ticker_symbol: str,
    max_days: int = 60,
    moneyness_bound: float = 0.30,
) -> pd.DataFrame:
    """
    fetch option data for next 'max_days',
    collect IV and delta for OTM puts and calls,
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

        # use ATM call for the strike nearest to spot, OTM puts below it, and OTM calls above it
        atm_strike = calls.loc[(calls['strike'] - underlying_price).abs().idxmin(), 'strike']
        otm_options = pd.concat([
            puts[puts['strike'] < atm_strike],
            calls[calls['strike'] >= atm_strike],
        ])

        for _, row in otm_options.iterrows():
            iv = row.get('impliedVolatility', np.nan)
            iv_pct = iv * 100 if (pd.notna(iv) and iv > 0.001) else np.nan

            surface_data.append({
                'days_to_expiry': days_to_expiry,
                'expiration_date': closest_expiry,
                'strike': row['strike'],
                'moneyness': row['strike'] / underlying_price,
                'implied_volatility': iv_pct,
                'delta': row.get('delta', np.nan),
                'price': row.get('price', np.nan),
            })

    return pd.DataFrame(surface_data)


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
    lookback_days = max(120, int(max_window * 3))
    df_ohlc = get_alpaca_ohlcv(ticker_symbol, lookback_days)
    underlying_price = get_latest_price(ticker_symbol)

    results = []

    for w in windows:
        # convert calendar-day target window to trading days for Yang-Zhang RV
        trading_days = max(2, int(round(w * 252 / 365)))
        rv_window = max(min_rv_window, trading_days) if use_min_rv else trading_days
        try:
            rv_estimate = compute_yang_zhang(
                open_prices=df_ohlc['open'],
                high_prices=df_ohlc['high'],
                low_prices=df_ohlc['low'],
                close_prices=df_ohlc['close'],
                window=rv_window,
            )
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

            # ATM IV (only requires valid impliedVolatility; keeps delta intact in calls/puts)
            valid_calls = calls.dropna(subset=['impliedVolatility'])
            if valid_calls.empty:
                raise ValueError(f"no calls with valid IV for target window {w}d")

            atm_call = valid_calls.iloc[(valid_calls['strike'] - spot).abs().argsort()[:1]].iloc[0]
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
    if df.empty or 'expiration_date' not in df.columns:
        print("no data available to plot !!!")
        return None

    # fetch spot price for the vertical line
    underlying_price = get_latest_price(ticker)

    expiries = sorted(df['expiration_date'].unique())[:max_lines]
    all_strikes = sorted(df['strike'].unique())

    fig = go.Figure()
    colors = px.colors.sequential.Plasma_r

    for i, expiry in enumerate(expiries):
        df_exp = (
            df[df['expiration_date'] == expiry]
            .sort_values(by='strike')
            .drop_duplicates(subset=['strike'])
            .set_index('strike')
            .reindex(all_strikes)
        )
        if df_exp['implied_volatility'].dropna().empty:
            continue

        # safely map to color scale
        color_idx = int((i / max(1, len(expiries) - 1)) * (len(colors) - 1))
        color = colors[color_idx]

        fig.add_trace(go.Scatter(
            x=all_strikes,
            y=df_exp['implied_volatility'],
            mode='lines+markers',
            connectgaps=True,
            line=dict(color=color, width=2.5, shape='spline'),
            marker=dict(color=color, size=6, opacity=0.6),
            name=f'{expiry}',
            legendgroup=f'{expiry}',
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
        legend=dict(title="Expiration Date", yanchor="top", y=1, xanchor="left", x=1.02)
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
        title=f"{ticker} VRP by Expiration",
        xaxis=dict(
            title="Days to Expiry",
            tickmode="array",
            tickvals=multi_vrp_df['Target_Days'].tolist(),
        ),
        yaxis_title="Percent (%)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig


def plot_deltas_by_strike_and_expiry(df: pd.DataFrame, ticker: str, max_lines: int = 8):
    """
    plot OTM option absolute deltas across strike and expiration date
    """
    if df.empty or 'expiration_date' not in df.columns or 'delta' not in df.columns:
        print("no delta data available to plot !!!")
        return None

    df_valid = df.dropna(subset=['delta'])
    if df_valid.empty:
        print("no delta data available to plot !!!")
        return None

    underlying_price = get_latest_price(ticker)
    expiries = sorted(df_valid['expiration_date'].unique())[:max_lines]
    all_strikes = sorted(df['strike'].unique())

    fig = go.Figure()
    colors = px.colors.sequential.Viridis_r

    for i, expiry in enumerate(expiries):
        df_exp = (
            df[df['expiration_date'] == expiry]
            .sort_values(by='strike')
            .drop_duplicates(subset=['strike'])
            .set_index('strike')
            .reindex(all_strikes)
        )
        if df_exp['delta'].dropna().empty:
            continue

        color_idx = int((i / max(1, len(expiries) - 1)) * (len(colors) - 1))
        color = colors[color_idx]

        fig.add_trace(go.Scatter(
            x=all_strikes,
            y=np.abs(df_exp['delta']),
            mode='lines+markers',
            connectgaps=True,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=5, opacity=0.6),
            name=f'{expiry}',
            legendgroup=f'{expiry}',
        ))

    fig.add_vline(
        x=underlying_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Spot: ${underlying_price:.2f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{ticker} Absolute Delta by Expiration (OTM Puts and Calls)",
        xaxis_title="Strike Price",
        yaxis_title="Absolute Delta |Δ|",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(title="Expiration Date", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig
