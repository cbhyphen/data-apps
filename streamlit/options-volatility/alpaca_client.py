from datetime import datetime, timedelta
import os
import re

import numpy as np
import pandas as pd

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest


ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]


def get_latest_price(ticker_symbol: str):
    # fetch spot price for the vertical line
    stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    trade_req = StockLatestTradeRequest(symbol_or_symbols=ticker_symbol)
    latest_trade = stock_client.get_stock_latest_trade(trade_req)
    underlying_price = latest_trade[ticker_symbol].price
    return underlying_price


# get raw call and put option chains for expiration date closest to target days
def fetch_raw_options_chain(
    ticker_symbol: str,
    days_target: int = 60
    ):

    opt_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # get spot price using using latest trade.
    # avoids 0.0 ask price when market is closed.
    underlying_price = get_latest_price(ticker_symbol)

    today = datetime.today().date()
    target_date = today + timedelta(days=days_target)

    # contracts request strictly around the target expiration window
    # instead of starting from today, preventing pagination limits from hiding later dates
    contracts_req = GetOptionContractsRequest(
        underlying_symbols=[ticker_symbol],
        status="active",
        expiration_date_gte=target_date - timedelta(days=3),
        expiration_date_lte=target_date + timedelta(days=3)
    )
    # query Metadata API to get contract symbols
    contracts_meta = trading_client.get_option_contracts(contracts_req)

    if not contracts_meta.option_contracts:
        raise ValueError(f"no options contracts found for {ticker_symbol} around {target_date}")

    # find closest expiration date
    unique_dates = set(c.expiration_date for c in contracts_meta.option_contracts)
    exp_dates = list(unique_dates)
    days_diffs = [abs((exp - today).days - days_target) for exp in exp_dates]
    closest_idx = np.argmin(days_diffs)
    closest_expiry = exp_dates[closest_idx]

    t_years = (closest_expiry - today).days / 365.0

    # filter symbols by closest expiry and moneyness bounds (+/- 35%)
    strike_min = underlying_price * 0.65
    strike_max = underlying_price * 1.35
    target_symbols = []

    for c in contracts_meta.option_contracts:
        if c.expiration_date == closest_expiry:
            strike = float(c.strike_price)
            if strike_min <= strike <= strike_max:
                target_symbols.append(c.symbol)

    if not target_symbols:
        raise ValueError(f"no strikes found within bounds for expiry {closest_expiry}")

    # fetch pricing / greeks ONLY for kept symbols.
    # drastically reduces payload compared to the full chain.
    snap_req = OptionSnapshotRequest(symbol_or_symbols=target_symbols)
    snapshots = opt_client.get_option_snapshot(snap_req)

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
