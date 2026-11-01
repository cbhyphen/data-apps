from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


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


def compute_yang_zhang(
    open_prices: pd.Series,
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    window: int,
    periods_per_year: int = 252,
) -> float:
    """
    compute Yang-Zhang realized volatility estimator on daily OHLC series
    """
    lengths = {
        len(open_prices),
        len(high_prices),
        len(low_prices),
        len(close_prices),
    }
    if len(lengths) != 1:
        raise ValueError("open_prices, high_prices, low_prices, and close_prices must have the same length")

    n = lengths.pop()
    if n < 2:
        return np.nan

    o = pd.Series(open_prices.astype(float).values)
    h = pd.Series(high_prices.astype(float).values)
    l = pd.Series(low_prices.astype(float).values)
    c = pd.Series(close_prices.astype(float).values)

    c_prev = c.shift(1)

    log_oc = np.log(o / c_prev)
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)

    rs = log_ho * log_hc + log_lo * log_lc

    if window <= 1:
        var_overnight = log_oc ** 2
        var_open_to_close = log_co ** 2
        var_rs = rs
        k = 0.0
    else:
        var_overnight = log_oc.rolling(window=window, min_periods=2).var(ddof=1)
        var_open_to_close = log_co.rolling(window=window, min_periods=2).var(ddof=1)
        var_rs = rs.rolling(window=window, min_periods=1).mean()
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_var = var_overnight + k * var_open_to_close + (1 - k) * var_rs
    latest_var = yz_var.iloc[-1]
    if pd.isna(latest_var):
        return np.nan

    latest_vol = np.sqrt(max(0.0, float(latest_var)) * periods_per_year)
    return float(latest_vol)


def bs_price(
    sigma: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    is_call: bool,
) -> float:
    """
    compute Black-Scholes European option price
    """
    sqrt_t = np.sqrt(t_years)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        return float(spot * norm.cdf(d1) - strike * np.exp(-r * t_years) * norm.cdf(d2))
    return float(strike * np.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def bs_delta(
    sigma: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    is_call: bool,
) -> float:
    """
    compute Black-Scholes option delta
    """
    sqrt_t = np.sqrt(t_years)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1.0)


def bs_implied_vol(
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    r: float,
    is_call: bool,
) -> float | None:
    """
    solve Black-Scholes implied volatility from option price using Brent's method
    """
    if pd.isna(price) or price <= 0 or t_years <= 0:
        return None
    try:
        low_p = bs_price(0.001, spot, strike, t_years, r, is_call)
        high_p = bs_price(5.0, spot, strike, t_years, r, is_call)
        if low_p < price < high_p:
            return float(brentq(
                lambda s: bs_price(s, spot, strike, t_years, r, is_call) - price,
                0.001,
                5.0,
            ))
    except Exception:
        pass
    return None

