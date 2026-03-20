# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BINANCE_REST_BASE = "https://api.binance.com"


def _safe_request(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Any:
    """
    统一封装 Binance GET 请求，返回 JSON。
    调用方负责解释返回结构。
    """
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Binance 请求失败: url=%s params=%s error=%s", url, params, exc)
        raise

def get_crypto_klines(
     symbol: str,
     interval: str = "1d",
     limit: int = 90,
 ) -> pd.DataFrame:
     """
     获取币种 K 线数据（日线为主），对齐股票日线结构。

     Args:
         symbol: 交易对，如 'BTCUSDT'、'ETHUSDT'
         interval: Binance K 线周期，默认 '1d'
         limit: 返回条数，默认 90

     Returns:
         DataFrame，包含列：
         - date: pandas.Timestamp（UTC 时间）
         - open/high/low/close: float
         - volume: float（基础币种成交量）
         - amount: float（计价币种成交额，quote volume）
         - pct_chg: float（相邻收盘价涨跌幅，%）
     """
     symbol = (symbol or "").strip().upper()
     if not symbol:
         raise ValueError("symbol 不能为空")

     url = f"{_BINANCE_REST_BASE}/api/v3/klines"
     raw = _safe_request(
         url,
         params={
             "symbol": symbol,
             "interval": interval,
             "limit": max(1, min(int(limit), 1000)),
         },
     )

     if not raw:
         raise ValueError(f"Binance 返回空 K 线数据: symbol={symbol}")

     rows: List[Dict[str, Any]] = []
     for item in raw:
         # 参考官方返回结构：https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
         open_time = int(item[0])  # ms
         open_price = float(item[1])
         high_price = float(item[2])
         low_price = float(item[3])
         close_price = float(item[4])
         volume_base = float(item[5])
         quote_volume = float(item[7])

         rows.append(
             {
                 "date": datetime.utcfromtimestamp(open_time / 1000.0),
                 "open": open_price,
                 "high": high_price,
                 "low": low_price,
                 "close": close_price,
                 "volume": volume_base,
                 "amount": quote_volume,
             }
         )

     df = pd.DataFrame(rows)
     df = df.sort_values("date", ascending=True).reset_index(drop=True)

     # 计算简单的日收益率 pct_chg：相邻收盘价之比
     df["pct_chg"] = df["close"].pct_change().fillna(0.0) * 100.0
     df["pct_chg"] = df["pct_chg"].round(4)

     return df

def get_crypto_quote(symbol: str) -> Dict[str, Any]:
    """
    获取币种 24 小时行情快照，用于报告中的当前价格与涨跌幅展示。

    Args:
        symbol: 交易对，如 'BTCUSDT'

    Returns:
        dict，字段示例：
        - symbol: 交易对
        - price: 最新价
        - change_pct: 24h 涨跌幅（%）
        - change_amount: 24h 涨跌额
        - high: 24h 最高价
        - low: 24h 最低价
        - volume: 24h 成交量（基础币种）
        - amount: 24h 成交额（计价币种）
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol 不能为空")

    url = f"{_BINANCE_REST_BASE}/api/v3/ticker/24hr"
    data = _safe_request(url, params={"symbol": symbol})

    try:
        last_price = float(data.get("lastPrice"))
    except (TypeError, ValueError):
        last_price = None

    try:
        price_change = float(data.get("priceChange"))
    except (TypeError, ValueError):
        price_change = None

    try:
        price_change_pct = float(data.get("priceChangePercent"))
    except (TypeError, ValueError):
        price_change_pct = None

    try:
        high_price = float(data.get("highPrice"))
    except (TypeError, ValueError):
        high_price = None

    try:
        low_price = float(data.get("lowPrice"))
    except (TypeError, ValueError):
        low_price = None

    try:
        vol_base = float(data.get("volume"))
    except (TypeError, ValueError):
        vol_base = None

    try:
        vol_quote = float(data.get("quoteVolume"))
    except (TypeError, ValueError):
        vol_quote = None

    return {
        "symbol": symbol,
        "price": last_price,
        "change_pct": price_change_pct,
        "change_amount": price_change,
        "high": high_price,
        "low": low_price,
        "volume": vol_base,
        "amount": vol_quote,
    }


__all__ = ["get_crypto_klines", "get_crypto_quote"]

