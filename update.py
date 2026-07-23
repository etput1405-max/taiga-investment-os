from __future__ import annotations

import io
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

JST = timezone(timedelta(hours=9))
DATA_FILE = Path("data.json")
HISTORY_FILE = Path("history.json")

MAX_STOCKS = int(os.getenv("MAX_STOCKS", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "70"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "130"))

MARKETS: dict[str, str] = {
    "^N225": "日経平均",
    "1306.T": "TOPIX連動ETF",
    "JPY=X": "ドル円",
    "^TNX": "米国10年金利",
    "^VIX": "VIX",
    "CL=F": "WTI原油",
    "GC=F": "金",
    "HG=F": "銅",
    "SOXX": "米国半導体ETF",
}

FALLBACK: dict[str, str] = {
    "1605.T": "INPEX", "1928.T": "積水ハウス", "2914.T": "JT",
    "4063.T": "信越化学", "4502.T": "武田薬品", "4519.T": "中外製薬",
    "4568.T": "第一三共", "6098.T": "リクルート", "6301.T": "コマツ",
    "6501.T": "日立", "6758.T": "ソニーG", "6861.T": "キーエンス",
    "6902.T": "デンソー", "7203.T": "トヨタ", "7267.T": "ホンダ",
    "7741.T": "HOYA", "7974.T": "任天堂", "8001.T": "伊藤忠",
    "8031.T": "三井物産", "8035.T": "東京エレクトロン", "8058.T": "三菱商事",
    "8306.T": "三菱UFJ", "8316.T": "三井住友FG", "8411.T": "みずほFG",
    "8766.T": "東京海上", "9432.T": "NTT", "9433.T": "KDDI",
    "9983.T": "ファーストリテイリング", "9984.T": "ソフトバンクG",
}

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TaigaInvestmentOS/1.0)"}


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def get_universe() -> dict[str, str]:
    """JPX掲載の上場銘柄一覧を取得。取得できなければ固定リストを使用。"""
    try:
        response = requests.get(JPX_PAGE, timeout=30, headers=HEADERS)
        response.raise_for_status()
        links = re.findall(r'href="([^"]+\.(?:xls|xlsx))"', response.text, re.I)
        absolute_links = [
            ("https://www.jpx.co.jp" + link if link.startswith("/") else link)
            for link in links
        ]

        for url in absolute_links:
            try:
                file_response = requests.get(url, timeout=60, headers=HEADERS)
                file_response.raise_for_status()
                frame = pd.read_excel(io.BytesIO(file_response.content))

                code_column = next(
                    col for col in frame.columns
                    if "コード" in str(col) or "Code" in str(col)
                )
                name_column = next(
                    col for col in frame.columns
                    if "銘柄名" in str(col) or "Issue name" in str(col)
                )
                market_column = next(
                    (
                        col for col in frame.columns
                        if "市場・商品区分" in str(col) or "Market" in str(col)
                    ),
                    None,
                )

                if market_column is not None:
                    is_stock = frame[market_column].astype(str).str.contains(
                        "プライム|スタンダード|グロース|Prime|Standard|Growth",
                        regex=True,
                        na=False,
                    )
                    frame = frame[is_stock]

                universe: dict[str, str] = {}
                for _, row in frame.iterrows():
                    code = str(row[code_column]).split(".")[0].strip()
                    if re.fullmatch(r"\d{4}", code):
                        universe[f"{code}.T"] = str(row[name_column]).strip()

                if len(universe) > 1_000:
                    return universe
            except Exception:
                continue
    except Exception:
        pass

    return FALLBACK.copy()


def percent_change(series: pd.Series, days: int) -> float | None:
    values = series.dropna()
    if len(values) <= days:
        return None
    base = float(values.iloc[-days - 1])
    latest = float(values.iloc[-1])
    if base == 0:
        return None
    return (latest / base - 1) * 100


def z_score(series: pd.Series) -> pd.Series:
    deviation = float(series.std(ddof=0))
    if not deviation or math.isnan(deviation):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / deviation


def download_close(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed_batches: list[str] = []

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        batch_frame: pd.DataFrame | None = None

        for attempt in range(3):
            try:
                raw = yf.download(
                    batch,
                    period="1y",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="column",
                    timeout=45,
                )
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        batch_frame = raw["Close"]
                    else:
                        batch_frame = raw[["Close"]].rename(
                            columns={"Close": batch[0]}
                        )
                    break
            except Exception:
                time.sleep(2 ** attempt)

        if batch_frame is None or batch_frame.empty:
            failed_batches.extend(batch)
        else:
            frames.append(batch_frame)

        time.sleep(1)

    if not frames:
        return pd.DataFrame(), failed_batches

    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index().ffill()
    return close, failed_batches


def build_market(close: pd.DataFrame) -> list[dict[str, Any]]:
    market: list[dict[str, Any]] = []
    for symbol, name in MARKETS.items():
        if symbol not in close.columns:
            continue
        series = close[symbol].dropna()
        if series.empty:
            continue
        market.append(
            {
                "symbol": symbol,
                "name": name,
                "value": round(float(series.iloc[-1]), 4),
                "change5": round(percent_change(series, 5) or 0, 2),
                "change20": round(percent_change(series, 20) or 0, 2),
                "change60": round(percent_change(series, 60) or 0, 2),
            }
        )
    return market


def build_regime(market: list[dict[str, Any]]) -> dict[str, Any]:
    values = {row["symbol"]: row for row in market}
    nikkei = values.get("^N225", {}).get("change20", 0)
    vix = values.get("^VIX", {}).get("change20", 0)
    copper = values.get("HG=F", {}).get("change20", 0)
    semiconductors = values.get("SOXX", {}).get("change20", 0)

    score = float(
        np.clip(
            50
            + nikkei * 0.65
            - max(vix, 0) * 0.30
            + copper * 0.18
            + semiconductors * 0.18,
            0,
            100,
        )
    )
    label = "リスクオン" if score >= 65 else "リスクオフ" if score <= 35 else "中立"
    return {
        "score": round(score, 1),
        "label": label,
        "nikkei20": nikkei,
        "vix20": vix,
    }


def main() -> None:
    universe = get_universe()
    selected = dict(list(sorted(universe.items()))[:MAX_STOCKS])

    stock_close, batch_failures = download_close(list(selected))
    market_close, _ = download_close(list(MARKETS))

    market = build_market(market_close)
    regime = build_regime(market)
    nikkei20 = float(regime.get("nikkei20", 0))

    raw: list[dict[str, Any]] = []
    unavailable: list[str] = []

    for ticker, name in selected.items():
        if ticker not in stock_close.columns:
            unavailable.append(ticker)
            continue

        series = stock_close[ticker].dropna()
        if len(series) < MIN_HISTORY:
            unavailable.append(ticker)
            continue

        returns = series.pct_change().dropna()
        price = float(series.iloc[-1])
        high = float(series.tail(252).max())
        low = float(series.tail(252).min())

        raw.append(
            {
                "ticker": ticker,
                "name": name,
                "price": price,
                "mom5": percent_change(series, 5),
                "mom20": percent_change(series, 20),
                "mom60": percent_change(series, 60),
                "mom120": percent_change(series, 120),
                "drawdown": (price / high - 1) * 100,
                "from_low": (price / low - 1) * 100 if low else 0,
                "vol20": float(returns.tail(20).std(ddof=0) * math.sqrt(252) * 100),
                "volume_proxy": abs(float(returns.tail(5).mean())) * 100,
            }
        )

    if not raw:
        previous = load_json(DATA_FILE, {})
        previous["last_error"] = "株価データを取得できませんでした"
        previous["attempted_at"] = datetime.now(JST).isoformat(timespec="seconds")
        save_json(DATA_FILE, previous)
        raise RuntimeError("株価データを取得できませんでした")

    frame = pd.DataFrame(raw).set_index("ticker")
    factors = pd.DataFrame(index=frame.index)

    factors["momentum"] = (
        z_score(frame["mom5"].fillna(0)) * 0.15
        + z_score(frame["mom20"].fillna(0)) * 0.30
        + z_score(frame["mom60"].fillna(0)) * 0.35
        + z_score(frame["mom120"].fillna(0)) * 0.20
    )
    factors["pullback"] = z_score(-abs(frame["drawdown"].fillna(0) + 12))
    factors["stability"] = z_score(-frame["vol20"].fillna(frame["vol20"].median()))
    factors["relative"] = z_score(frame["mom60"].fillna(0) - nikkei20)
    factors["recovery"] = z_score(
        frame["from_low"].fillna(0).clip(lower=0, upper=80)
    )

    composite = (
        factors["momentum"] * 0.34
        + factors["pullback"] * 0.24
        + factors["stability"] * 0.14
        + factors["relative"] * 0.20
        + factors["recovery"] * 0.08
    )

    frame["score"] = (composite.rank(pct=True) * 100).round(1)
    frame["momentum_score"] = (factors["momentum"].rank(pct=True) * 100).round(1)
    frame["pullback_score"] = (factors["pullback"].rank(pct=True) * 100).round(1)
    frame["stability_score"] = (factors["stability"].rank(pct=True) * 100).round(1)
    frame["relative_score"] = (factors["relative"].rank(pct=True) * 100).round(1)

    median_volatility = float(frame["vol20"].median())
    ranking: list[dict[str, Any]] = []

    for ticker, row in frame.sort_values("score", ascending=False).iterrows():
        score = float(row["score"])
        signal = (
            "最優先候補" if score >= 90
            else "買い場候補" if score >= 75
            else "監視" if score >= 50
            else "見送り"
        )

        reasons: list[str] = []
        if (row["mom20"] or 0) > 0 and (row["mom60"] or 0) > 0:
            reasons.append("短中期上向き")
        if -20 <= row["drawdown"] <= -5:
            reasons.append("適度な押し目")
        if (row["mom60"] or 0) > nikkei20:
            reasons.append("日経平均より強い")
        if row["vol20"] < median_volatility:
            reasons.append("値動き比較的安定")
        if (row["mom5"] or 0) > 0 and (row["mom20"] or 0) < 0:
            reasons.append("反転初動候補")

        ranking.append(
            {
                "ticker": ticker,
                "name": str(row["name"]),
                "price": round(float(row["price"]), 2),
                "score": score,
                "signal": signal,
                "mom5": round(float(row["mom5"]), 2) if pd.notna(row["mom5"]) else None,
                "mom20": round(float(row["mom20"]), 2) if pd.notna(row["mom20"]) else None,
                "mom60": round(float(row["mom60"]), 2) if pd.notna(row["mom60"]) else None,
                "mom120": round(float(row["mom120"]), 2) if pd.notna(row["mom120"]) else None,
                "drawdown": round(float(row["drawdown"]), 2),
                "vol20": round(float(row["vol20"]), 2),
                "momentum_score": float(row["momentum_score"]),
                "pullback_score": float(row["pullback_score"]),
                "stability_score": float(row["stability_score"]),
                "relative_score": float(row["relative_score"]),
                "reasons": reasons or ["市場内の相対順位で抽出"],
            }
        )

    now = datetime.now(JST).isoformat(timespec="seconds")
    payload = {
        "version": "1.0",
        "updated_at": now,
        "regime": regime,
        "market": market,
        "ranking": ranking,
        "meta": {
            "universe": len(universe),
            "requested": len(selected),
            "success": len(raw),
            "failed": len(selected) - len(raw),
            "batch_failures": len(batch_failures),
        },
        "last_error": None,
    }
    save_json(DATA_FILE, payload)

    history = load_json(HISTORY_FILE, [])
    history.insert(
        0,
        {
            "updated_at": now,
            "regime": regime,
            "top": [
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "score": row["score"],
                    "signal": row["signal"],
                }
                for row in ranking[:10]
            ],
            "success": len(raw),
        },
    )
    save_json(HISTORY_FILE, history[:120])


if __name__ == "__main__":
    main()
