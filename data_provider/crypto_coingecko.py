# -*- coding: utf-8 -*-
"""
CoinGecko 币圈数据源（降级实现）

目标：
- 在 Binance 被网络/风控限制时，仍尽可能提供币种日线数据
- 返回结构对齐 `crypto_binance.py`：open/high/low/close/volume/amount/pct_chg

说明：
- CoinGecko 的 `/market_chart` 日线接口不直接提供 OHLC，因此 open/high/low 采用 close 的近似填充
- 对于 `get_crypto_quote`，仅保证 price/change_pct 可用，其它字段返回 None 以便上层 prompt 使用 N/A
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_DEFAULT_VS_CURRENCY = "usd"

# 小范围常见币种映射：覆盖 CI 最常见的 BTC/ETH/SOL 等
_SYMBOL_TO_COINGECKO_ID: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "DOT": "polkadot",
    "TRX": "tron",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
}


def _safe_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("CoinGecko 请求失败: url=%s params=%s error=%s", url, params, exc)
        raise


def _extract_base_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    for suffix in ("USDT", "USD", "BUSD", "USDC", "TUSD", "EUR", "GBP"):
        if s.endswith(suffix) and len(s) > len(suffix):
            return s[: -len(suffix)]
    return s


def _resolve_coin_id(base_symbol: str) -> str:
    base_symbol = (base_symbol or "").strip().upper()
    if not base_symbol:
        raise ValueError("base_symbol 不能为空")

    mapped = _SYMBOL_TO_COINGECKO_ID.get(base_symbol)
    if mapped:
        return mapped

    # 兜底：搜索
    query = base_symbol.lower()
    url = f"{_COINGECKO_BASE}/search"
    data = _safe_get_json(url, params={"query": query}, timeout=10.0)
    coins = data.get("coins") or []
    if not coins:
        raise ValueError(f"CoinGecko 搜索无结果: query={query}")

    # 优先匹配 symbol（不保证完全一致，但能提升命中率）
    for c in coins:
        sym = (c.get("symbol") or "").strip().upper()
        if sym == base_symbol:
            cid = (c.get("id") or "").strip()
            if cid:
                return cid

    # 否则取第一个
    cid = (coins[0].get("id") or "").strip()
    if not cid:
        raise ValueError(f"CoinGecko 搜索返回无 id: query={query}")
    return cid


def get_crypto_klines(
    symbol: str,
    interval: str = "1d",
    limit: int = 90,
) -> pd.DataFrame:
    """
    获取币种 K 线数据（日线近似），对齐股票日线结构。

    返回列对齐 Binance 实现：
    - date: pandas.Timestamp（UTC 时间）
    - open/high/low/close: float
    - volume: float（基础币种成交量的近似）
    - amount: float（计价币种成交额的近似）
    - pct_chg: float（相邻收盘价涨跌幅，%）
    """
    _ = interval  # CoinGecko 日线仅支持近似，因此此参数目前不做区分

    symbol_norm = (symbol or "").strip().upper()
    if not symbol_norm:
        raise ValueError("symbol 不能为空")

    base_symbol = _extract_base_symbol(symbol_norm)
    coin_id = _resolve_coin_id(base_symbol)

    # CoinGecko market_chart 的 days 参数：单位天，返回每日数据点
    # limit 可能大于天数上限：这里做保守截断，避免参数异常
    safe_days = max(1, min(int(limit), 365))

    url = f"{_COINGECKO_BASE}/coins/{coin_id}/market_chart"
    data = _safe_get_json(
        url,
        params={
            "vs_currency": _DEFAULT_VS_CURRENCY,
            "days": safe_days,
            "interval": "daily",
        },
        timeout=15.0,
    )

    prices: List[List[Any]] = data.get("prices") or []
    volumes: List[List[Any]] = data.get("total_volumes") or []
    if not prices:
        raise ValueError(f"CoinGecko 返回空 prices: symbol={symbol_norm}, coin_id={coin_id}")

    vol_by_ts: Dict[int, float] = {}
    for ts, v in volumes:
        try:
            vol_by_ts[int(ts)] = float(v)
        except (TypeError, ValueError):
            continue

    prices_sorted = sorted(prices, key=lambda x: x[0])
    prices_slice = prices_sorted[-max(1, int(limit)) :]

    rows: List[Dict[str, Any]] = []
    prev_close: Optional[float] = None

    for ts, price in prices_slice:
        ts_int = int(ts)
        dt = datetime.utcfromtimestamp(ts_int / 1000.0)
        close = float(price)

        amount_usd = vol_by_ts.get(ts_int)
        if amount_usd is None:
            amount_usd = 0.0

        volume_base = (amount_usd / close) if close > 0 else 0.0

        open_price = prev_close if prev_close is not None else close
        high_price = close
        low_price = close

        pct_chg = 0.0 if prev_close is None or prev_close == 0 else (close / prev_close - 1.0) * 100.0
        pct_chg = round(float(pct_chg), 4)

        rows.append(
            {
                "date": dt,
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close),
                "volume": float(volume_base),
                "amount": float(amount_usd),
                "pct_chg": pct_chg,
            }
        )

        prev_close = close

    df = pd.DataFrame(rows)
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


def get_crypto_quote(symbol: str) -> Dict[str, Any]:
    """
    获取币种 24 小时行情快照（尽力提供）。

    这里保证：
    - price：USD 最新价
    - change_pct：24h 涨跌幅（%）

    high/low/volume/amount：返回 None（上层 prompt 会显示 N/A）
    """
    symbol_norm = (symbol or "").strip().upper()
    if not symbol_norm:
        raise ValueError("symbol 不能为空")

    base_symbol = _extract_base_symbol(symbol_norm)
    coin_id = _resolve_coin_id(base_symbol)

    url = f"{_COINGECKO_BASE}/simple/price"
    data = _safe_get_json(
        url,
        params={
            "ids": coin_id,
            "vs_currencies": _DEFAULT_VS_CURRENCY,
            "include_24hr_change": "true",
        },
        timeout=10.0,
    )

    coin = (data or {}).get(coin_id) or {}
    try:
        price = float(coin.get(_DEFAULT_VS_CURRENCY))
    except (TypeError, ValueError):
        price = None

    change_key = f"{_DEFAULT_VS_CURRENCY}_24h_change"
    try:
        change_pct = float(coin.get(change_key))
    except (TypeError, ValueError):
        change_pct = None

    return {
        "symbol": symbol_norm,
        "price": price,
        "change_pct": change_pct,
        "change_amount": None,
        "high": None,
        "low": None,
        "volume": None,
        "amount": None,
    }


__all__ = ["get_crypto_klines", "get_crypto_quote"]

