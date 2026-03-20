# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from data_provider import get_crypto_klines, get_crypto_quote
from src.analyzer import AnalysisResult, GeminiAnalyzer
from src.config import Config, get_config
from src.enums import ReportType
from src.notification import NotificationService
from src.search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class CryptoContext:
    """轻量级币种分析上下文（对齐股票分析需要的关键信息）。"""

    symbol: str
    date: str
    today: Dict[str, Any]
    yesterday: Dict[str, Any]
    realtime: Dict[str, Any]


class CryptoAnalysisPipeline:
    """
    币圈分析流水线（与股票分析解耦，MVP 版本）。

    使用方式：
        pipeline = CryptoAnalysisPipeline(config, analyzer, search_service, notifier)
        results = pipeline.run(dry_run=False, send_notification=True)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        analyzer: Optional[GeminiAnalyzer] = None,
        search_service: Optional[SearchService] = None,
        notifier: Optional[NotificationService] = None,
    ) -> None:
        self.config = config or get_config()
        self.analyzer = analyzer or GeminiAnalyzer()
        self.notifier = notifier or NotificationService()
        self.search_service = search_service or SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            brave_keys=self.config.brave_api_keys,
            serpapi_keys=self.config.serpapi_keys,
            minimax_keys=self.config.minimax_api_keys,
            searxng_base_urls=self.config.searxng_base_urls,
            searxng_public_instances_enabled=self.config.searxng_public_instances_enabled,
            news_max_age_days=self.config.news_max_age_days,
            news_strategy_profile=getattr(self.config, "news_strategy_profile", "short"),
        )

    def _build_crypto_context(self, symbol: str) -> Optional[CryptoContext]:
        """
        拉取单个币种的 K 线与 24h 行情，构建 GeminiAnalyzer 可消费的上下文。
        """
        try:
            # 取最近 ~60 根日 K 线，用于计算 MA 与 pct_chg
            df: pd.DataFrame = get_crypto_klines(symbol, interval="1d", limit=60)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Crypto] 获取 K 线失败: %s (%s)", symbol, exc)
            return None

        if df is None or df.empty:
            logger.warning("[Crypto] %s K 线数据为空，跳过", symbol)
            return None

        df = df.sort_values("date", ascending=True).reset_index(drop=True)

        # 计算 MA 与简单量能指标（与股票日线结构尽量对齐）
        df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
        df["ma10"] = df["close"].rolling(window=10, min_periods=1).mean()
        df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
        # volume_ratio: 今日成交量 / 5 日均量（shift 1），含义近似“放量倍数”
        avg_vol5 = df["volume"].rolling(window=5, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / avg_vol5.shift(1)
        df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

        today_row = df.iloc[-1].to_dict()
        yesterday_row = df.iloc[-2].to_dict() if len(df) >= 2 else {}

        # 尝试获取 24h 行情快照
        realtime: Dict[str, Any] = {}
        try:
            quote = get_crypto_quote(symbol)
            if quote:
                realtime = {
                    "name": symbol,
                    "price": quote.get("price"),
                    "change_pct": quote.get("change_pct"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "amount": quote.get("amount"),
                    "volume": quote.get("volume"),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Crypto] 获取 24h 行情失败（忽略）: %s (%s)", symbol, exc)

        ctx = CryptoContext(
            symbol=symbol,
            date=today_row["date"].strftime("%Y-%m-%d")
            if isinstance(today_row.get("date"), datetime)
            else str(today_row.get("date")),
            today=today_row,
            yesterday=yesterday_row,
            realtime=realtime,
        )
        return ctx

    def _convert_to_stock_like_context(self, ctx: CryptoContext) -> Dict[str, Any]:
        """
        将 CryptoContext 转为“类股票”上下文结构，尽量复用 GeminiAnalyzer 的提示词逻辑。

        说明：
        - code 使用交易对符号（如 BTCUSDT）；
        - stock_name 也暂时使用符号本身，后续可扩展为友好名称（如 BTC/USDT）。
        """
        today = ctx.today or {}
        yesterday = ctx.yesterday or {}
        realtime = ctx.realtime or {}

        context: Dict[str, Any] = {
            "code": ctx.symbol,
            "stock_name": ctx.symbol,
            "date": ctx.date,
            "today": {
                "open": today.get("open"),
                "high": today.get("high"),
                "low": today.get("low"),
                "close": today.get("close"),
                "volume": today.get("volume"),
                "amount": today.get("amount"),
                "pct_chg": today.get("pct_chg"),
                "ma5": today.get("ma5"),
                "ma10": today.get("ma10"),
                "ma20": today.get("ma20"),
            },
            "yesterday": (
                {
                    "close": yesterday.get("close"),
                    "volume": yesterday.get("volume"),
                }
                if yesterday
                else {}
            ),
            "data_missing": False,
        }

        # 价格涨跌与量能变化（复用股票逻辑）
        try:
            if (
                yesterday
                and today.get("close") is not None
                and yesterday.get("close") not in (None, 0)
            ):
                yc = float(yesterday["close"])
                tc = float(today["close"])
                context["price_change_ratio"] = round((tc - yc) / yc * 100.0, 2)
        except Exception:  # noqa: BLE001
            pass

        try:
            if (
                yesterday
                and today.get("volume") is not None
                and yesterday.get("volume") not in (None, 0)
            ):
                yv = float(yesterday["volume"])
                tv = float(today["volume"])
                context["volume_change_ratio"] = round(tv / yv, 2)
        except Exception:  # noqa: BLE001
            pass

        # 简化版 MA 状态描述
        ma5 = today.get("ma5") or 0
        ma10 = today.get("ma10") or 0
        ma20 = today.get("ma20") or 0
        price = today.get("close") or realtime.get("price") or 0
        if price and ma5 and ma10 and ma20:
            if price > ma5 > ma10 > ma20 > 0:
                ma_status = "多头排列 📈"
            elif price < ma5 < ma10 < ma20 and ma20 > 0:
                ma_status = "空头排列 📉"
            else:
                ma_status = "震荡整理 ↔️"
            context["ma_status"] = ma_status

        # 将 realtime 装入与股票一致的结构
        if realtime:
            context["realtime"] = {
                "name": ctx.symbol,
                "price": realtime.get("price"),
                "change_pct": realtime.get("change_pct"),
                "high": realtime.get("high"),
                "low": realtime.get("low"),
                "volume": realtime.get("volume"),
                "amount": realtime.get("amount"),
            }

        # 透传当前运行时的新闻窗口设置，避免与全局配置不一致
        context["news_window_days"] = getattr(self.search_service, "news_window_days", 3)

        # 标记资产类型，便于后续扩展 prompt（当前版本 GeminiAnalyzer 仍按股票提示词处理）
        context["asset_type"] = "crypto_spot"
        return context

    def _analyze_single_symbol(self, symbol: str) -> Optional[AnalysisResult]:
        """分析单个币种，返回 AnalysisResult。"""
        ctx = self._build_crypto_context(symbol)
        if not ctx:
            return None

        context = self._convert_to_stock_like_context(ctx)

        # 搜索相关新闻 / 舆情：复用 SearchService，但使用英文语境更适合币圈
        news_context = None
        try:
            news_resp = self.search_service.search_stock_news(
                stock_code=symbol,
                stock_name=symbol,
                max_results=5,
            )
            if news_resp and news_resp.results:
                news_context = news_resp.to_context(max_results=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Crypto] %s 新闻搜索失败: %s", symbol, exc)

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[Crypto] LLM 分析器未可用，跳过 %s", symbol)
            return None

        try:
            result = self.analyzer.analyze(context, news_context=news_context)
            logger.info(
                "[Crypto] %s 分析完成: %s, 评分 %s",
                symbol,
                result.operation_advice,
                result.sentiment_score,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("[Crypto] %s 分析失败: %s", symbol, exc)
            return None

    def run(
        self,
        dry_run: bool = False,
        send_notification: bool = True,
        report_type: Optional[ReportType] = None,
    ) -> List[AnalysisResult]:
        """
        运行币圈分析主流程。

        Args:
            dry_run: True 时仅拉取数据不调用 LLM。
            send_notification: 是否通过现有通知通道发送「币圈日报」。
            report_type: 报告类型，默认复用全局 REPORT_TYPE。
        """
        symbols = getattr(self.config, "crypto_symbol_list", []) or []
        if not getattr(self.config, "crypto_enabled", False):
            logger.info("[Crypto] CRYPTO_ENABLED=false，跳过币圈分析")
            return []
        if not symbols:
            logger.info("[Crypto] 未配置 CRYPTO_SYMBOL_LIST，跳过币圈分析")
            return []

        logger.info("[Crypto] 开始分析 %d 个币种: %s", len(symbols), ", ".join(symbols))
        results: List[AnalysisResult] = []

        for symbol in symbols:
            symbol = (symbol or "").strip().upper()
            if not symbol:
                continue
            logger.info("========== [Crypto] 开始处理 %s ==========", symbol)
            if dry_run:
                # 仅验证数据拉取链路可用
                _ = self._build_crypto_context(symbol)
                continue
            result = self._analyze_single_symbol(symbol)
            if result:
                results.append(result)

        if not results or dry_run:
            return results

        # 生成并发送币圈日报（与股票链路解耦，单独一份）
        try:
            rt = report_type or ReportType.from_str(
                getattr(self.config, "report_type", "simple")
            )
            report_md = self.notifier.generate_aggregate_report(
                results, rt, report_date=None
            )
            # 为了在推送中清晰区分，这里在内容前加一个简单标题
            header = f"# 🪙 Crypto Daily ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            full_content = header + report_md
            if send_notification and self.notifier.is_available():
                ok = self.notifier.send(full_content)
                if ok:
                    logger.info("[Crypto] 币圈日报推送成功")
                else:
                    logger.warning("[Crypto] 币圈日报推送失败")
        except Exception as exc:  # noqa: BLE001
            logger.error("[Crypto] 生成或推送币圈日报失败: %s", exc)

        return results

