import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest


ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
MAX_CALLS_PER_MINUTE = 150  # safety buffer below alpaca 200/min free-tier limit


class RateLimitError(Exception):
    pass


@st.cache_resource
def _get_rate_limit_state():
    return deque(), threading.Lock()


def _record_api_call():
    """
    sliding-window rate limiter shared across all Streamlit sessions
    """
    timestamps, lock = _get_rate_limit_state()
    with lock:
        now = time.monotonic()
        while timestamps and now - timestamps[0] >= 60.0:
            timestamps.popleft()

        if len(timestamps) >= MAX_CALLS_PER_MINUTE:
            wait_seconds = int(60.0 - (now - timestamps[0])) + 1
            raise RateLimitError(
                f"Alpaca API rate limit reached ({MAX_CALLS_PER_MINUTE} calls/min). "
                f"please wait ~{wait_seconds}s and try again."
            )

        timestamps.append(now)


@st.cache_data(ttl=120, show_spinner=False)
def get_latest_price(ticker_symbol: str):
    """
    fetch latest spot price
    """
    _record_api_call()
    stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    trade_req = StockLatestTradeRequest(symbol_or_symbols=ticker_symbol)
    latest_trade = stock_client.get_stock_latest_trade(trade_req)
    underlying_price = latest_trade[ticker_symbol].price
    return underlying_price


@st.cache_data(ttl=120, show_spinner=False)
def fetch_raw_options_chain(
    ticker_symbol: str,
    days_target: int = 60,
    underlying_price: float | None = None,
    moneyness_bound: float = 0.30,
    ) -> dict:
    """
    get raw call and put option chains for expiration date closest to target days
    """

    opt_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # get spot price using latest trade if not provided.
    # avoids 0.0 ask price when market is closed.
    if underlying_price is None:
        underlying_price = get_latest_price(ticker_symbol)

    today = datetime.today().date()
    target_date = today + timedelta(days=days_target)

    strike_min = underlying_price * (1.0 - moneyness_bound)
    strike_max = underlying_price * (1.0 + moneyness_bound)

    # contracts request strictly around the target Friday window (including Thursday for holidays)
    # and bounded by strike range so pagination limits don't truncate puts or near-spot strikes
    contracts_req = GetOptionContractsRequest(
        underlying_symbols=[ticker_symbol],
        status="active",
        expiration_date_gte=target_date - timedelta(days=1),
        expiration_date_lte=target_date,
        strike_price_gte=f"{strike_min:.2f}",
        strike_price_lte=f"{strike_max:.2f}",
        limit=1000,
    )
    # query Metadata API to get contract symbols
    _record_api_call()
    contracts_meta = trading_client.get_option_contracts(contracts_req)

    if not contracts_meta.option_contracts:
        raise ValueError(f"no options contracts found for {ticker_symbol} around {target_date}")

    # prefer exact Friday match, falling back to Thursday if Friday is a market holiday
    closest_expiry = min(
        {c.expiration_date for c in contracts_meta.option_contracts},
        key=lambda exp: abs((exp - target_date).days),
    )

    t_years = (closest_expiry - today).days / 365.0

    # filter symbols by closest expiry; prefer $5 strike increments when enough exist for spline fitting
    expiry_contracts = [
        c for c in contracts_meta.option_contracts
        if c.expiration_date == closest_expiry
        and strike_min <= float(c.strike_price) <= strike_max
    ]
    target_symbols = [
        c.symbol for c in expiry_contracts if float(c.strike_price) % 5 == 0
    ]
    if len(target_symbols) < 8:
        target_symbols = [c.symbol for c in expiry_contracts]

    if not target_symbols:
        raise ValueError(f"no strikes found within bounds for expiry {closest_expiry}")

    # fetch pricing / greeks ONLY for kept symbols (batched in chunks of 100 per Alpaca API limit)
    snapshots = {}
    for i in range(0, len(target_symbols), 100):
        _record_api_call()
        snap_req = OptionSnapshotRequest(symbol_or_symbols=target_symbols[i : i + 100])
        snapshots.update(opt_client.get_option_snapshot(snap_req))

    records = []
    for sym, snapshot in snapshots.items():
        match = re.match(r'^([A-Za-z]+)(\d{6})([CP])(\d{8})$', sym)
        if not match:
            continue

        _, date_str, opt_type, strike_str = match.groups()
        contract_type = 'call' if opt_type == 'C' else 'put'
        strike = int(strike_str) / 1000.0

        iv = snapshot.implied_volatility
        greeks = snapshot.greeks
        delta = greeks.delta if greeks else np.nan

        records.append({
            'symbol': sym,
            'expiration_date': closest_expiry,
            'contract_type': contract_type,
            'strike': strike,
            'impliedVolatility': iv if iv is not None else np.nan,
            'delta': delta if delta is not None else np.nan,
        })

    chain_df = pd.DataFrame(records)

    if chain_df.empty:
         raise ValueError(f"parsed option chain is empty for {ticker_symbol} at target expiry {closest_expiry}")

    # separate by contract type
    calls = chain_df[chain_df['contract_type'] == 'call'].copy()
    puts = chain_df[chain_df['contract_type'] == 'put'].copy()

    # format output to match expected columns
    closest_expiry_str = closest_expiry.strftime("%Y-%m-%d")

    return {
        'underlying_price': underlying_price,
        't_years': t_years,
        'calls': calls,
        'puts': puts,
        'closest_expiry_str': closest_expiry_str
    }
