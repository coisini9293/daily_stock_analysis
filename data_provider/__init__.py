# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（动态调整）：
【配置了 TUSHARE_TOKEN 时】
1. TushareFetcher (Priority 0) - 🔥 最高优先级（动态提升）
2. EfinanceFetcher (Priority 0) - 同优先级
3. AkshareFetcher (Priority 1) - 来自 akshare 库
4. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
5. BaostockFetcher (Priority 3) - 来自 baostock 库
6. YfinanceFetcher (Priority 4) - 来自 yfinance 库

【未配置 TUSHARE_TOKEN 时】
1. EfinanceFetcher (Priority 0) - 最高优先级，来自 efinance 库
2. AkshareFetcher (Priority 1) - 来自 akshare 库
3. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
4. TushareFetcher (Priority 2) - 来自 tushare 库（不可用）
5. BaostockFetcher (Priority 3) - 来自 baostock 库
6. YfinanceFetcher (Priority 4) - 来自 yfinance 库

提示：优先级数字越小越优先，同优先级按初始化顺序排列
"""

import logging

from .base import BaseFetcher, DataFetcherManager
from .efinance_fetcher import EfinanceFetcher
from .akshare_fetcher import AkshareFetcher, is_hk_stock_code
from .tushare_fetcher import TushareFetcher
from .pytdx_fetcher import PytdxFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher
from .crypto_binance import (
    get_crypto_klines as _binance_get_crypto_klines,
    get_crypto_quote as _binance_get_crypto_quote,
)
from .crypto_coingecko import (
    get_crypto_klines as _coingecko_get_crypto_klines,
    get_crypto_quote as _coingecko_get_crypto_quote,
)
from .us_index_mapping import is_us_index_code, is_us_stock_code, get_us_index_yf_symbol, US_INDEX_MAPPING

logger = logging.getLogger(__name__)


def _parse_crypto_provider_list(provider_str: str) -> list[str]:
    """
    支持逗号分隔的 provider 列表，决定 fallback 顺序。
    例如：`binance,coingecko`
    """
    raw = (provider_str or "").strip().lower()
    if not raw:
        return ["binance"]
    items = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return items or ["binance"]


def _resolve_crypto_providers() -> list[str]:
    # 懒加载，避免在 import data_provider 时触发配置初始化链
    from src.config import get_config

    cfg = get_config()
    return _parse_crypto_provider_list(getattr(cfg, "crypto_data_provider", "binance"))


def get_crypto_klines(
    symbol: str,
    interval: str = "1d",
    limit: int = 90,
):
    """
    获取币圈 K 线（带故障切换）。
    fallback 顺序由 `CRYPTO_DATA_PROVIDER` 决定（可配置为 `binance,coingecko`）。
    """
    providers = _resolve_crypto_providers()
    last_exc: Exception | None = None

    for p in providers:
        try:
            if p == "binance":
                return _binance_get_crypto_klines(symbol, interval=interval, limit=limit)
            if p in ("coingecko", "coin_gecko"):
                return _coingecko_get_crypto_klines(symbol, interval=interval, limit=limit)

            raise ValueError(f"Unknown crypto provider: {p}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Crypto provider failed: provider=%s symbol=%s err=%s", p, symbol, exc)

    # 若全部 provider 都失败，抛出最后一次异常，便于上层定位
    if last_exc:
        raise last_exc
    raise ValueError(f"No crypto providers configured: {providers}")


def get_crypto_quote(symbol: str):
    """获取币圈 24h 行情快照（带故障切换）。"""
    providers = _resolve_crypto_providers()
    last_exc: Exception | None = None

    for p in providers:
        try:
            if p == "binance":
                return _binance_get_crypto_quote(symbol)
            if p in ("coingecko", "coin_gecko"):
                return _coingecko_get_crypto_quote(symbol)
            raise ValueError(f"Unknown crypto provider: {p}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Crypto quote provider failed: provider=%s symbol=%s err=%s", p, symbol, exc)

    if last_exc:
        raise last_exc
    raise ValueError(f"No crypto providers configured: {providers}")

__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'EfinanceFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'PytdxFetcher',
    'BaostockFetcher',
    'YfinanceFetcher',
    'get_crypto_klines',
    'get_crypto_quote',
    'is_us_index_code',
    'is_us_stock_code',
    'is_hk_stock_code',
    'get_us_index_yf_symbol',
    'US_INDEX_MAPPING',
]
