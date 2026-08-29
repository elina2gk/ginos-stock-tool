#!/usr/bin/env python3
"""
AI Stock & ETF Selector – Rich Interactive App with Live Refresh
===============================================================
Modern browser-based tool for general investors.

New in this version:
- Auto-refresh every 30s / 60s / 2min (or Off)
- Very prominent current price + daily change %
- Rich profiles, scores, charts, CSV export

Launch:
  python3 -m streamlit run stock_selector_app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import logging
from typing import List, Dict, Optional

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Gino's Stock and ETF Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _expected_password() -> str:
    try:
        if hasattr(st, "secrets") and "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except Exception:
        pass
    return os.environ.get("APP_PASSWORD", "GinoTool2026")


def require_login():
    if st.session_state.get("authenticated"):
        return True

    st.title("Gino's Stock and ETF Tool")
    st.subheader("This app is password protected")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Unlock", type="primary"):
        if pwd == _expected_password():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()


# App is open; password gate disabled.

# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------
POPULAR_ETFS = [
    "VOO", "SPY", "IVV", "VTI", "ITOT", "QQQ", "QQQM", "VGT", "XLK", "ARKK",
    "SCHD", "VIG", "VYM", "DVY", "DGRO", "VXUS", "IXUS", "VEA", "IEMG", "VWO",
    "BND", "AGG", "TLT", "IEF", "VCIT", "XLF", "XLE", "XLV", "XLI", "SMH",
    "BOTZ", "IJH", "IJR", "VB", "VBR",
]

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    "ABBV", "MRK", "KO", "PEP", "COST", "AVGO", "AMD", "CRM", "NFLX",
    "ADBE", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "LRCX",
    "NOW", "INTU", "ISRG", "BKNG", "SBUX", "NEE", "RTX", "CAT", "GE",
    "IBM", "AMGN", "PFE", "TMO", "DHR", "ACN", "LIN", "PM", "UPS",
]

# ---------------------------------------------------------------------------
# Gino's Corner – Portfolio persistence
# ---------------------------------------------------------------------------
import json
import os

PORTFOLIO_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "ginos_corner_portfolio.json")


def load_portfolio() -> list:
    """Load saved holdings from Desktop JSON file."""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []


ALERTS_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "ginos_alert_settings.json")
CARRIER_SMS = {
    "AT&T": "txt.att.net",
    "Verizon": "vtext.com",
    "T-Mobile": "tmomail.net",
    "Sprint": "messaging.sprintpcs.com",
    "Google Fi": "msg.fi.google.com",
}


def load_alert_settings() -> dict:
    defaults = {
        "email": "",
        "phone": "",
        "carrier": "AT&T",
        "email_on": False,
        "sms_on": False,
        "threshold_pct": 1.0,
        "watchlist": "",
        "watchlist_stocks": [],
        "watchlist_etfs": [],
        "last_sent": {},
        "alert_log": [],
    }
    try:
        if os.path.exists(ALERTS_FILE):
            with open(ALERTS_FILE, "r") as f:
                data = json.load(f)
                defaults.update(data or {})
    except Exception:
        pass
    return defaults


def save_alert_settings(settings: dict):
    try:
        with open(ALERTS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save alert settings: {e}")


def parse_watchlist(text: str) -> list:
    raw = (text or "").replace(",", " ").upper().split()
    return list(dict.fromkeys([t.strip() for t in raw if t.strip()]))


def check_watchlist_email_alerts(settings) -> list:
    """Email when a watchlist ticker is down >= threshold vs prior close."""
    tickers = list(dict.fromkeys(
        list(settings.get("watchlist_stocks") or [])
        + list(settings.get("watchlist_etfs") or [])
        + parse_watchlist(settings.get("watchlist", ""))
    ))
    if not tickers:
        return []
    thresh = float(settings.get("threshold_pct") or 1.0)
    quotes = fetch_live_quotes(tickers)
    last_sent = settings.get("last_sent") or {}
    today = datetime.now().strftime("%Y-%m-%d")
    hits = []
    for t in tickers:
        q = quotes.get(t) or {}
        chg = q.get("change_pct")
        price = q.get("price")
        if chg is None or float(chg) > -thresh:
            continue
        key = f"WL-{t}-{today}"
        if last_sent.get(key):
            continue
        hits.append((t, price, float(chg), key))
    if not hits:
        return []
    lines = [
        "Watchlist alert from Gino's Stock and ETF Tool",
        "A stock is at least 1% below its last market price (prior close).",
        "",
    ]
    for t, price, chg, _ in hits:
        lines.append(f"{t}: ${price:.2f}  ({chg:+.2f}%)" if price else f"{t}: {chg:+.2f}%")
    status = "logged"
    if settings.get("email_on") and settings.get("email"):
        try:
            status = send_alert_message([settings["email"].strip()], "Watchlist: stock down 1%+", "\n".join(lines))
        except Exception as e:
            status = f"send failed: {e}"
    else:
        status = "saved to Alerts page (email not connected)"

    log = list(settings.get("alert_log") or [])
    for t, price, chg, key in hits:
        last_sent[key] = datetime.now().isoformat()
        log.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": t,
            "price": price,
            "change_pct": chg,
            "email_status": status,
            "message": f"{t} is {chg:+.2f}% vs last market close" + (f" at ${price:.2f}" if price else ""),
        })
    settings["last_sent"] = last_sent
    settings["alert_log"] = log[:100]
    save_alert_settings(settings)
    return [(t, price, chg, status) for t, price, chg, _ in hits]


def _smtp_config():
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    port = int(os.environ.get("SMTP_PORT", "587"))
    try:
        if hasattr(st, "secrets"):
            host = st.secrets.get("SMTP_HOST", host)
            user = st.secrets.get("SMTP_USER", user)
            password = st.secrets.get("SMTP_PASS", password)
            port = int(st.secrets.get("SMTP_PORT", port))
    except Exception:
        pass
    return host, user, password, port


def send_alert_message(to_addrs, subject, body) -> str:
    import smtplib
    from email.mime.text import MIMEText
    host, user, password, port = _smtp_config()
    if not host or not user or not password:
        return "Email is not configured. Add SMTP_HOST, SMTP_USER, SMTP_PASS in Streamlit Secrets."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to_addrs)
    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, to_addrs, msg.as_string())
    return "sent"


def maybe_send_drop_alerts(holdings, quotes, settings):
    """Alert when a holding is down at least threshold % today or vs cost."""
    thresh = float(settings.get("threshold_pct") or 1.0)
    last_sent = settings.get("last_sent") or {}
    today = datetime.now().strftime("%Y-%m-%d")
    hits = []
    for h in holdings:
        t = str(h.get("ticker", "")).upper()
        q = quotes.get(t) or {}
        price = q.get("price")
        chg = q.get("change_pct")
        shares = float(h.get("shares") or 0)
        cost = float(h.get("cost_basis") or 0)
        pl_pct = None
        if price and cost > 0:
            pl_pct = (float(price) - cost) / cost * 100
        down = False
        why = []
        if chg is not None and float(chg) <= -thresh:
            down = True
            why.append(f"day {float(chg):+.2f}%")
        if pl_pct is not None and pl_pct <= -thresh:
            down = True
            why.append(f"vs cost {pl_pct:+.2f}%")
        if not down:
            continue
        key = f"{t}-{today}"
        if last_sent.get(key):
            continue
        hits.append((t, price, chg, pl_pct, shares, cost, ", ".join(why), key))

    if not hits:
        return settings, []

    lines = ["Holding drop alert from Gino's Stock and ETF Tool:", ""]
    for t, price, chg, pl_pct, shares, cost, why, _ in hits:
        lines.append(
            f"{t}: ${price if price else '—'} | {why} | {shares} shares @ ${cost:.2f}"
        )
    body = "\n".join(lines)
    subject = "Alert: holding down 1%+"
    to_addrs = []
    if settings.get("email_on") and settings.get("email"):
        to_addrs.append(settings["email"].strip())
    if settings.get("sms_on") and settings.get("phone"):
        phone = "".join(ch for ch in str(settings["phone"]) if ch.isdigit())
        if phone:
            domain = CARRIER_SMS.get(settings.get("carrier") or "AT&T", "txt.att.net")
            to_addrs.append(f"{phone}@{domain}")
    status = "no destination"
    if to_addrs:
        try:
            status = send_alert_message(to_addrs, subject, body)
        except Exception as e:
            status = f"send failed: {e}"
    log = list(settings.get("alert_log") or [])
    for item in hits:
        last_sent[item[7]] = datetime.now().isoformat()
        log.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": item[0],
            "price": item[1],
            "change_pct": item[2],
            "email_status": status if to_addrs else "saved to Alerts page (email not connected)",
            "message": f"{item[0]} is {item[6]}",
        })
    settings["last_sent"] = last_sent
    settings["alert_log"] = log[:100]
    save_alert_settings(settings)
    return settings, [(item[0], item[6], status) for item in hits]


def save_portfolio(holdings: list):
    """Save holdings to Desktop JSON file."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(holdings, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save portfolio: {e}")


def fetch_upcoming_earnings(days: int = 7) -> pd.DataFrame:
    rows = []
    today = datetime.now().date()
    for i in range(days):
        d = today + pd.Timedelta(days=i)
        # skip weekends
        if d.weekday() >= 5:
            continue
        data = _http_get_json(
            f"https://api.nasdaq.com/api/calendar/earnings?date={d.strftime('%Y-%m-%d')}"
        )
        for item in ((data or {}).get("data") or {}).get("rows") or []:
            rows.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Symbol": item.get("symbol"),
                "Company": item.get("name"),
                "Time": str(item.get("time") or "").replace("time-", "").replace("-", " "),
                "EPS Est.": item.get("epsForecast"),
                "Last Year EPS": item.get("lastYearEPS"),
                "Market Cap": item.get("marketCap"),
            })
    return pd.DataFrame(rows)


def fetch_ipo_calendar() -> Dict[str, pd.DataFrame]:
    month = datetime.now().strftime("%Y-%m")
    data = (_http_get_json(f"https://api.nasdaq.com/api/ipo/calendar?date={month}") or {}).get("data") or {}
    out = {}
    for key, title in [("upcoming", "Upcoming"), ("priced", "Recently priced"), ("filed", "Filed")]:
        block = data.get(key) or {}
        raw = block.get("rows") if isinstance(block, dict) else []
        recs = []
        for item in raw or []:
            recs.append({
                "Symbol": item.get("proposedTickerSymbol") or "—",
                "Company": item.get("companyName"),
                "Exchange": item.get("proposedExchange"),
                "Price": item.get("proposedSharePrice"),
                "Shares": item.get("sharesOffered"),
                "Value": item.get("dollarValueOfSharesOffered"),
                "Date": item.get("pricedDate") or item.get("filedDate"),
                "Status": item.get("dealStatus") or title,
            })
        out[title] = pd.DataFrame(recs)
    return out



# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    data = {}
    try:
        df = yf.download(tickers, period=period, group_by="ticker",
                         auto_adjust=True, progress=False, threads=True)
        if len(tickers) == 1:
            t = tickers[0]
            if not df.empty:
                d = df.copy()
                d.columns = [c.lower() for c in d.columns]
                data[t] = d
        else:
            for t in tickers:
                try:
                    if t in df.columns.get_level_values(0):
                        sub = df[t].copy()
                        sub.columns = [c.lower() for c in sub.columns]
                        if len(sub.dropna()) > 40:
                            data[t] = sub
                except Exception:
                    continue
    except Exception:
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period=period, auto_adjust=True)
                if not hist.empty and len(hist) > 40:
                    hist.columns = [c.lower() for c in hist.columns]
                    data[t] = hist
            except Exception:
                continue
    return data


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and force lowercase names."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Prefer the price field level if present
        level0 = [str(c).lower() for c in out.columns.get_level_values(0)]
        if "close" in level0:
            out.columns = [str(c).lower() for c in out.columns.get_level_values(0)]
        else:
            out.columns = [str(c[-1]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    return out


def _quote_from_frame(df: pd.DataFrame) -> Optional[Dict]:
    df = _normalize_ohlc(df)
    if "close" not in df.columns:
        return None
    closes = pd.to_numeric(df["close"], errors="coerce").dropna()
    if closes.empty:
        return None
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
    chg = last - prev
    chg_pct = (chg / prev) * 100 if prev else 0.0
    return {"price": last, "change": chg, "change_pct": chg_pct, "prev_close": prev}


def _http_get_json(url: str) -> Optional[dict]:
    """Use stdlib urllib so Mac LibreSSL / urllib3 issues don't blank prices."""
    try:
        import json as _json
        from urllib.request import Request, urlopen
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        with urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return _json.loads(raw)
    except Exception:
        return None



def _parse_money(text) -> Optional[float]:
    if text is None:
        return None
    try:
        s = str(text).replace("$", "").replace(",", "").replace("%", "").replace("+", "").strip()
        if not s or s.upper() in ("N/A", "NA", "--"):
            return None
        return float(s)
    except Exception:
        return None


COMMON_NAMES = {
    "APPLE": "AAPL", "NVIDIA": "NVDA", "MICROSOFT": "MSFT", "AMAZON": "AMZN",
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "META": "META", "FACEBOOK": "META",
    "TESLA": "TSLA", "NETFLIX": "NFLX", "INTEL": "INTC", "AMD": "AMD",
    "BERKSHIRE": "BRK-B", "JPMORGAN": "JPM", "VISA": "V", "WALMART": "WMT",
    "COSTCO": "COST", "COKE": "KO", "EXXON": "XOM",
    "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM",
    "TAIWAN SEMICONDUCTOR MANUFACTURING": "TSM",
    "COREWEAVE": "CRWV",
    "SK HYNIX": "SKHY", "SKHYNIX": "SKHY", "HYNIX": "SKHY",
}


def requests_quote(text: str) -> str:
    try:
        from urllib.parse import quote
        return quote(str(text))
    except Exception:
        return str(text).replace(" ", "%20")


def resolve_ticker(raw: str) -> str:
    """Convert a company name like APPLE into a ticker like AAPL."""
    text = (raw or "").strip().upper().replace(".", "-")
    if not text:
        return text
    if text in COMMON_NAMES:
        return COMMON_NAMES[text]
    try:
        data = _http_get_json(
            f"https://query1.finance.yahoo.com/v1/finance/search?q={requests_quote(raw)}"
        )
        quotes = (data or {}).get("quotes") or []
        for item in quotes:
            sym = item.get("symbol")
            qtype = (item.get("quoteType") or "").upper()
            if sym and qtype in ("EQUITY", "ETF", "MUTUALFUND", "INDEX"):
                return str(sym).upper()
        if quotes and quotes[0].get("symbol"):
            return str(quotes[0]["symbol"]).upper()
    except Exception:
        pass
    return text


def _quote_from_nasdaq(t: str) -> Optional[Dict]:
    for asset in ("stocks", "etf"):
        data = _http_get_json(f"https://api.nasdaq.com/api/quote/{t}/info?assetclass={asset}")
        primary = ((data or {}).get("data") or {}).get("primaryData") or {}
        price = _parse_money(primary.get("lastSalePrice"))
        if price is None:
            continue
        raw_chg = str(primary.get("netChange") or "")
        raw_pct = str(primary.get("percentageChange") or "")
        chg = _parse_money(raw_chg) or 0.0
        chg_pct = _parse_money(raw_pct) or 0.0
        if raw_chg.startswith("-"):
            chg = -abs(chg)
        elif raw_chg.startswith("+"):
            chg = abs(chg)
        if raw_pct.startswith("-"):
            chg_pct = -abs(chg_pct)
        elif raw_pct.startswith("+"):
            chg_pct = abs(chg_pct)
        return {
            "price": price,
            "change": float(chg),
            "change_pct": float(chg_pct),
            "prev_close": price - float(chg),
            "source": "Nasdaq",
        }
    return None


def _quote_from_cnbc(t: str) -> Optional[Dict]:
    data = _http_get_json(
        "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
        f"?symbols={t}&requestMethod=itv&noform=1&partnerId=2&output=json"
    )
    quotes = ((data or {}).get("FormattedQuoteResult") or {}).get("FormattedQuote") or []
    if not quotes:
        return None
    item = quotes[0]
    price = _parse_money(item.get("last"))
    if price is None:
        return None
    chg = _parse_money(item.get("change")) or 0.0
    chg_pct = _parse_money(item.get("change_pct")) or 0.0
    raw_chg = str(item.get("change") or "")
    raw_pct = str(item.get("change_pct") or "")
    if raw_chg.startswith("-"):
        chg = -abs(chg)
    if raw_pct.startswith("-"):
        chg_pct = -abs(chg_pct)
    return {
        "price": price,
        "change": float(chg),
        "change_pct": float(chg_pct),
        "prev_close": price - float(chg),
        "source": "CNBC",
    }


def fetch_one_quote(ticker: str) -> Optional[Dict]:
    """Nasdaq first, then CNBC, then Yahoo as last resort."""
    t = resolve_ticker(ticker)
    q = _quote_from_nasdaq(t)
    if q:
        return q
    q = _quote_from_cnbc(t)
    if q:
        return q
    try:
        data = _http_get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{t}?range=5d&interval=1d"
        )
        result = (data or {}).get("chart", {}).get("result") or []
        if result:
            meta = result[0].get("meta") or {}
            price = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            if price is not None:
                price = float(price)
                prev = float(prev) if prev is not None else price
                chg = price - prev
                chg_pct = (chg / prev) * 100 if prev else 0.0
                return {"price": price, "change": chg, "change_pct": chg_pct, "prev_close": prev, "source": "Yahoo"}
    except Exception:
        pass
    return None




@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_quotes(tickers: List[str]) -> Dict[str, Dict]:
    """Latest price + daily change. Fetches in parallel for speed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    quotes = {}
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return quotes

    def _job(sym):
        return sym, fetch_one_quote(sym)

    with ThreadPoolExecutor(max_workers=min(8, len(clean))) as pool:
        futures = [pool.submit(_job, t) for t in clean]
        for fut in as_completed(futures):
            try:
                sym, q = fut.result()
                if q:
                    quotes[sym] = q
            except Exception:
                continue
    return quotes


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df = df.copy()
    df["sma_20"] = SMAIndicator(close, 20).sma_indicator()
    df["sma_50"] = SMAIndicator(close, 50).sma_indicator()
    df["sma_200"] = SMAIndicator(close, 200).sma_indicator() if len(df) > 200 else np.nan
    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["rsi"] = RSIIndicator(close, 14).rsi()
    bb = BollingerBands(close)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)
    df["ret_12m"] = close.pct_change(min(252, len(df)-1))
    df["volatility_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_rich_info(ticker: str) -> Dict:
    info = {
        "ticker": ticker, "name": ticker, "type": "Unknown",
        "sector": None, "industry": None, "category": None,
        "market_cap": None, "trailing_pe": None, "dividend_yield": None,
        "52w_high": None, "52w_low": None, "description": None,
        "recommendation": None, "target_mean": None, "num_analysts": None,
    }
    try:
        t = yf.Ticker(ticker)
        raw = t.info or {}
        info["name"] = raw.get("longName") or raw.get("shortName") or ticker
        qtype = (raw.get("quoteType") or "").upper()
        info["type"] = "ETF" if qtype == "ETF" else ("Stock" if qtype in ("EQUITY", "STOCK") else (qtype.title() or "Unknown"))
        info["sector"] = raw.get("sector") or raw.get("category")
        info["industry"] = raw.get("industry")
        info["category"] = raw.get("category")
        info["market_cap"] = raw.get("marketCap")
        info["trailing_pe"] = raw.get("trailingPE")
        info["dividend_yield"] = raw.get("dividendYield")
        info["52w_high"] = raw.get("fiftyTwoWeekHigh")
        info["52w_low"] = raw.get("fiftyTwoWeekLow")
        info["description"] = raw.get("longBusinessSummary")
        try:
            rec = getattr(t, "recommendations_summary", None)
            if rec is not None and hasattr(rec, "empty") and not rec.empty:
                latest = rec.iloc[0]
                sb = latest.get("strongBuy", 0) or 0
                b = latest.get("buy", 0) or 0
                h = latest.get("hold", 0) or 0
                s = latest.get("sell", 0) or 0
                ss = latest.get("strongSell", 0) or 0
                total = sb + b + h + s + ss
                if total > 0:
                    info["num_analysts"] = int(total)
                    bullish = sb + b
                    info["recommendation"] = "Bullish" if bullish/total >= 0.70 else ("Neutral-Bullish" if bullish/total >= 0.45 else "Mixed/Cautious")
        except Exception:
            pass
        try:
            targets = getattr(t, "analyst_price_targets", None)
            if isinstance(targets, dict):
                info["target_mean"] = targets.get("mean")
        except Exception:
            pass
    except Exception:
        pass
    return info


def score_symbol(df: pd.DataFrame, ticker: str, is_etf: bool = False) -> Dict:
    if len(df) < 50:
        return {"ticker": ticker, "score": 0, "error": "Insufficient data"}
    latest, prev = df.iloc[-1], df.iloc[-2]
    score = 50.0
    reasons = []
    price = latest["close"]
    sma20, sma50, sma200 = latest.get("sma_20"), latest.get("sma_50"), latest.get("sma_200")

    if pd.notna(sma20) and pd.notna(sma50):
        if price > sma20 > sma50:
            score += 9; reasons.append("Strong uptrend (Price > SMA20 > SMA50)")
        elif price > sma50:
            score += 4; reasons.append("Above medium-term average")
        elif price < sma20 < sma50:
            score -= 6; reasons.append("Downtrend structure")
    if pd.notna(sma200):
        if price > sma200:
            score += 5; reasons.append("Above 200-day SMA")
        else:
            score -= 4; reasons.append("Below 200-day SMA")

    ret_1m = latest.get("ret_1m") or 0
    ret_3m = latest.get("ret_3m") or 0
    ret_6m = latest.get("ret_6m") or 0
    mom = 0
    if ret_1m > 0.05: mom += 5
    elif ret_1m > 0: mom += 2
    elif ret_1m < -0.08: mom -= 4
    if ret_3m > 0.10: mom += 7
    elif ret_3m > 0.03: mom += 3
    elif ret_3m < -0.10: mom -= 5
    if ret_6m > 0.15: mom += 6
    elif ret_6m > 0.05: mom += 2
    elif ret_6m < -0.15: mom -= 4
    score += mom
    if mom >= 10: reasons.append(f"Strong momentum ({ret_3m*100:.1f}% 3M)")
    elif mom <= -5: reasons.append("Weak momentum")

    rsi = latest.get("rsi", 50)
    if 40 <= rsi <= 62: score += 4; reasons.append("RSI healthy")
    elif 30 <= rsi < 40: score += 6; reasons.append("RSI near oversold")
    elif rsi > 75: score -= 5; reasons.append("RSI overbought")

    macd_hist = latest.get("macd_hist", 0)
    prev_hist = prev.get("macd_hist", 0)
    if macd_hist > 0 and macd_hist > prev_hist: score += 6; reasons.append("MACD expanding positively")
    elif macd_hist > 0: score += 3; reasons.append("MACD above signal")
    elif macd_hist < 0 and macd_hist < prev_hist: score -= 4; reasons.append("MACD deteriorating")

    bb_pct = latest.get("bb_pct", 0.5)
    if bb_pct < 0.15: score += 5; reasons.append("Near lower Bollinger")
    elif bb_pct > 0.92: score -= 3; reasons.append("Near upper Bollinger")

    vol20 = latest.get("volatility_20", 0.25)
    if not is_etf and vol20 and vol20 > 0.50: score -= 5; reasons.append("High volatility")
    elif vol20 and vol20 < 0.16: score += 2; reasons.append("Low volatility")
    if is_etf and ticker in ["VOO", "SPY", "IVV", "VTI", "QQQ", "SCHD", "BND", "AGG"]:
        score += 2; reasons.append("Core high-quality ETF")

    score = max(0.0, min(100.0, score))
    action = "STRONG BUY" if score >= 72 else "BUY" if score >= 60 else "HOLD" if score >= 48 else "WATCH" if score >= 35 else "AVOID"

    return {
        "ticker": ticker, "score": round(score, 1), "action": action,
        "price": round(float(price), 2),
        "rsi": round(float(rsi), 1) if pd.notna(rsi) else None,
        "ret_1m_pct": round(ret_1m * 100, 1), "ret_3m_pct": round(ret_3m * 100, 1),
        "ret_6m_pct": round(ret_6m * 100, 1),
        "ret_12m_pct": round((latest.get("ret_12m") or 0) * 100, 1),
        "volatility": round(float(vol20) * 100, 1) if pd.notna(vol20) else None,
        "reasons": reasons[:6], "df": df,
    }


def resolve_universe(mode_name: str, custom: str):
    if mode_name == "Custom tickers":
        raw = (custom or "").replace(",", " ").upper().split()
        tickers = list(dict.fromkeys([t.strip() for t in raw if t.strip()]))
        etf_set = set(POPULAR_ETFS)
    elif mode_name == "ETFs only":
        tickers = POPULAR_ETFS.copy()
        etf_set = set(POPULAR_ETFS)
    elif mode_name == "Stocks only":
        tickers = DEFAULT_STOCKS.copy()
        etf_set = set()
    else:
        tickers = DEFAULT_STOCKS + POPULAR_ETFS
        etf_set = set(POPULAR_ETFS)
    return tickers, etf_set


def run_market_scan(tickers, period, min_score, etf_set) -> pd.DataFrame:
    """Fast score-only scan used by Discover."""
    hist_data = fetch_history(tickers, period=period)
    results = []
    for ticker, raw in hist_data.items():
        try:
            df = compute_indicators(raw)
            is_etf = ticker in etf_set
            scored = score_symbol(df, ticker, is_etf=is_etf)
            if "error" in scored or scored["score"] < min_score:
                continue
            scored["type"] = "ETF" if is_etf else "Stock"
            scored["name"] = ticker
            results.append(scored)
        except Exception:
            continue
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    return out


def create_chart(ticker: str, df: pd.DataFrame, score: float, action: str):
    df = df.dropna(subset=["close"]).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.2, 0.25],
                        subplot_titles=(f"{ticker}  •  Score {score}  •  {action}", "Volume", "RSI & MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    for col, color, name in [("sma_20", "orange", "SMA 20"), ("sma_50", "blue", "SMA 50"), ("sma_200", "purple", "SMA 200")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line=dict(color=color, width=1.3)), row=1, col=1)
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.65), row=2, col=1)
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#ab47bc", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.45, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.45, row=3, col=1)
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(color="#42a5f5", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal", line=dict(color="#ff9800", width=1.2)), row=3, col=1)
    fig.update_layout(height=720, template="plotly_white", xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=40, r=20, t=60, b=30))
    return fig


def csv_link(df, filename, label="Download CSV"):
    from urllib.parse import quote
    csv = df.to_csv(index=False)
    st.markdown(
        f'<a download="{filename}" href="data:text/csv;charset=utf-8,{quote(csv)}" '
        f'style="display:inline-block;margin:8px 0;padding:10px 16px;background:#111111;'
        f'color:#ffd100;border:1px solid #ffd100;text-decoration:none;border-radius:6px;'
        f'font-weight:600;">{label}</a>',
        unsafe_allow_html=True,
    )


def format_mktcap(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    try:
        v = float(val)
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9: return f"${v/1e9:.1f}B"
        if v >= 1e6: return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"
    except Exception:
        return "—"


def color_change(val):
    if val is None: return ""
    return "normal" if val >= 0 else "inverse"


# ---------------------------------------------------------------------------
# Robinhood-style full-page layout (no left frame)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .stDeployButton, [data-testid="stToolbar"], #MainMenu {display: none !important;}
    [data-testid="stElementToolbar"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="StyledToolbar"],
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    button[title="Download as CSV"],
    button[aria-label="Download as CSV"] {
        display: none !important;
    }
    .stApp, [data-testid="stAppViewContainer"], section[data-testid="stMain"] {
        background-color: #000000 !important;
        overflow: visible !important;
        height: auto !important;
        pointer-events: auto !important;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span, .stMarkdown, .stCaption, .stText {
        color: #ffffff !important;
    }
    [data-testid="stCaption"], [data-testid="stWidgetLabel"], small, .stAlert p {
        color: #ffe680 !important;
    }
    .stSpinner, [data-testid="stStatusWidget"] { color: #ffffff !important; }
    .stMetric {background: #1a1a1a; border-radius: 12px; padding: 8px 10px; border-left: 4px solid #e10600;}
    button[kind="primary"] {background-color: #e10600 !important; color: #ffffff !important;}
    .stButton > button, [data-testid="stForm"] button, [data-testid="stBaseButton-secondary"] {
        background-color: #1a1a1a !important;
        color: #ffd100 !important;
        border: 1px solid #ffd100 !important;
    }
    .stButton > button p, .stButton > button span,
    [data-testid="stForm"] button p, [data-testid="stForm"] button span {
        color: #ffd100 !important;
    }
    button[kind="primary"] p, button[kind="primary"] span {
        color: #ffffff !important;
    }
    div[data-testid="stHorizontalBlock"] button {
        padding: 0.1rem 0.45rem !important;
        min-height: 26px !important;
        height: auto !important;
        font-size: 12px !important;
        line-height: 1.2 !important;
        border-radius: 4px !important;
        background-color: #111111 !important;
        color: #ffd100 !important;
        border: 1px solid #ffd100 !important;
    }
    div[data-testid="stHorizontalBlock"] button p,
    div[data-testid="stHorizontalBlock"] button span,
    div[data-testid="stHorizontalBlock"] button div {
        color: #ffd100 !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #e10600 !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover p,
    div[data-testid="stHorizontalBlock"] button:hover span {
        color: #ffffff !important;
    }
    div[data-testid="stDownloadButton"] button {
        min-height: 40px !important;
        height: auto !important;
        white-space: normal !important;
        overflow: visible !important;
        width: 100% !important;
        background-color: #1a1a1a !important;
        color: #ffd100 !important;
        border: 1px solid #ffd100 !important;
    }
    div[data-testid="stDownloadButton"] button p {
        color: #ffd100 !important;
        white-space: normal !important;
    }
    [data-baseweb="tab"] {color: #ffffff;}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {color: #ffffff !important;}
    .stSelectbox label, .stSlider label, .stTextInput label, .stCheckbox label {color: #ffffff !important;}
    div[data-baseweb="select"] {color: #000000;}

    [aria-selected="true"] {color: #ffd100 !important; border-bottom: 2px solid #ffd100 !important;}
    div[data-testid="stImage"] {
        width: 100vw !important;
        max-width: 100vw !important;
        margin-left: calc(50% - 50vw) !important;
        margin-right: calc(50% - 50vw) !important;
    }
    div[data-testid="stImage"] img {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        display: block !important;
        width: 100vw !important;
        max-width: 100vw !important;
        height: 210px !important;
        max-height: 210px !important;
        object-fit: cover !important;
        object-position: left center !important;
        background: #000;
    }
    .gino-pad { padding-left: 1.2rem; padding-right: 1.2rem; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100% !important;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-bottom: 1.2rem !important;
        }
        img.desktop-banner { display: none !important; }
        div[data-testid="stRadio"] {
            position: sticky; top: 0; z-index: 20;
            background: #000; padding: 4px 0 8px 0;
        }
        div[data-testid="stRadio"] > div { flex-wrap: wrap !important; gap: 4px !important; }
        div[data-testid="stRadio"] label {
            padding: 4px 8px !important;
            background: #161616 !important;
            border: 1px solid #444 !important;
            border-radius: 999px !important;
        }
        div[data-testid="stRadio"] label p {
            color: #ffd100 !important;
            font-size: 12px !important;
            line-height: 1.2 !important;
        }
        .stButton > button, [data-testid="stForm"] button {
            min-height: 34px !important;
            font-size: 13px !important;
            padding: 4px 10px !important;
        }
        input, textarea, select { font-size: 16px !important; }
        [data-testid="stDataFrame"] { overflow-x: auto !important; }
        h1, h2, h3 { font-size: 1.1rem !important; }
        .mobile-title {
            display: block !important;
            color: #f5e6b8 !important;
            font-family: Georgia, "Times New Roman", Times, serif !important;
            font-size: 17px !important;
            font-weight: 600 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            padding: 10px 2px 6px 2px !important;
            text-decoration: none !important;
        }
        a.mobile-title:visited { color: #f5e6b8 !important; }
        a:visited, a:visited span, a:visited p { color: #e10600 !important; }
        a { color: #ffd100 !important; }
    }
    .mobile-title { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

_img_roots = [
    os.path.dirname(os.path.abspath(__file__)),
    os.getcwd(),
    os.path.join(os.path.expanduser("~"), "Desktop"),
    "/home/workdir/artifacts",
]
_banner_candidates = []
for _root in _img_roots:
    _banner_candidates.extend([
        os.path.join(_root, "gino_banner.png"),
        os.path.join(_root, "gino_photo.png"),
        os.path.join(_root, "Gino_Photo.png"),
        os.path.join(_root, "gino_photo.PNG"),
    ])

_banner = next((p for p in _banner_candidates if os.path.exists(p)), None)
_photo_full = next(
    (
        p
        for p in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gino_photo.png"),
            os.path.join(os.path.expanduser("~"), "Desktop", "gino_photo.png"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "gino_banner.png"),
            os.path.join(os.path.expanduser("~"), "Desktop", "gino_banner.png"),
            _banner,
        ]
        if p and os.path.exists(p)
    ),
    None,
)
_portrait = False
try:
    _pv = st.query_params.get("portrait", "")
    if isinstance(_pv, list):
        _pv = _pv[0] if _pv else ""
    _portrait = str(_pv).lower() in ("1", "true", "yes")
except Exception:
    _portrait = False
if _portrait:
    st.markdown(
        """
        <style>
        header, [data-testid="stToolbar"], [data-testid="stHeader"] {display:none !important;}
        .block-container {padding: 0.4rem !important;}
        div[data-testid="stImage"], div[data-testid="stImage"] img {
            display: block !important;
            visibility: visible !important;
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
            object-fit: contain !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if _photo_full:
        import base64
        with open(_photo_full, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode("ascii")
        _ext = "jpeg" if _photo_full.lower().endswith((".jpg", ".jpeg")) else "png"
        st.markdown(
            f"<img src='data:image/{_ext};base64,{_b64}' style='width:100%;height:auto;display:block;' />",
            unsafe_allow_html=True,
        )
    else:
        st.error("Photo file not found. Name it gino_photo.png and keep it in the same folder as stock_selector_app.py")

    if st.button("Close", key="close_portrait"):
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()
    st.stop()

if _banner:
    try:
        st.image(_banner, use_container_width=True)
    except TypeError:
        st.image(_banner, use_column_width=True)
st.markdown(
    "<a class='mobile-title' href='?portrait=1' target='_blank' rel='noopener'>Gino's Stock and ETF Tool</a>",
    unsafe_allow_html=True,
)

refresh_sec = 30

if refresh_sec > 0:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")
    else:
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {refresh_sec * 1000});</script>",
            unsafe_allow_html=True,
        )

PAGES = ["Discover", "Leaders", "Screener", "My Watchlist", "Alerts", "Earnings", "Portfolio"]
PAGE_MAP = {"Earnings": "Earnings & IPOs"}
if "page" not in st.session_state:
    st.session_state.page = "Discover"
if st.session_state.page == "Earnings & IPOs":
    st.session_state.page = "Earnings"

picked = st.radio(
    "Page",
    PAGES,
    index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="page_radio",
)
st.session_state.page = picked
page = PAGE_MAP.get(picked, picked)
if page != "Discover":
    if st.button("Refresh now", key="manual_refresh"):
        st.session_state.pop("gino_quotes", None)
        st.rerun()

if st.session_state.get("_last_page") != page:
    st.session_state._last_page = page
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        const goTop = () => {
          try {
            const doc = window.parent.document;
            const nav = doc.querySelector('[data-testid="stHorizontalBlock"]');
            if (nav) { nav.scrollIntoView({behavior:'instant', block:'start'}); }
            const main = doc.querySelector('section.main');
            if (main) main.scrollTo(0, Math.max(0, (nav ? nav.offsetTop - 8 : 0)));
          } catch(e) {}
          try { window.parent.scrollTo(0,0); } catch(e) {}
          try { window.scrollTo(0,0); } catch(e) {}
        };
        goTop();
        setTimeout(goTop, 50);
        </script>
        """,
        height=0,
    )
_alert_cfg = load_alert_settings()
if _alert_cfg.get("watchlist") or _alert_cfg.get("watchlist_stocks") or _alert_cfg.get("watchlist_etfs"):
    try:
        check_watchlist_email_alerts(_alert_cfg)
    except Exception:
        pass




# Defaults used by Screener / Leaders (controls live on the Screener tab)
mode = st.session_state.get("mode", "All (Stocks + ETFs)")
custom_input = st.session_state.get("custom_input", "AAPL, MSFT, NVDA, QQQ, VOO, SCHD")
period = st.session_state.get("period", "1y")
top_n = st.session_state.get("top_n", 10)
min_score = st.session_state.get("min_score", 0)
show_charts = st.session_state.get("show_charts", True)
charts_count = st.session_state.get("charts_count", 4)
run_button = bool(st.session_state.get("run_scan", False))



# ========================= GINO'S CORNER =========================
if page == "Portfolio":
    st.header("📌 Gino's Corner")
    st.markdown("Track your personal holdings with live prices and accurate gain/loss.")
    alerts = load_alert_settings()
    with st.expander("Drop alerts (email / text)", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            alerts["email"] = st.text_input("Email", value=alerts.get("email", ""))
            alerts["email_on"] = st.checkbox("Send email", value=bool(alerts.get("email_on")))
        with a2:
            alerts["phone"] = st.text_input("Mobile number", value=alerts.get("phone", ""))
            alerts["sms_on"] = st.checkbox("Send text", value=bool(alerts.get("sms_on")))
        with a3:
            alerts["carrier"] = st.selectbox(
                "Carrier (for texts)",
                list(CARRIER_SMS.keys()),
                index=list(CARRIER_SMS.keys()).index(alerts.get("carrier", "AT&T"))
                if alerts.get("carrier") in CARRIER_SMS else 0,
            )
            alerts["threshold_pct"] = st.number_input("Alert if down at least %", min_value=0.1, max_value=20.0, value=float(alerts.get("threshold_pct") or 1.0), step=0.1)
        if st.button("Save alert settings"):
            save_alert_settings(alerts)
            st.success("Alert settings saved")
        st.caption("Email/text need SMTP settings in Streamlit Secrets: SMTP_HOST, SMTP_USER, SMTP_PASS. Texts use your carrier email gateway. One alert per stock per day.")


    # Load portfolio into session state
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()

    # --- Add new holding form ---
    st.subheader("Add or Update a Holding")
    with st.form("add_holding_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            _choices = [""] + [f"{s}  (Stock)" for s in sorted(set(DEFAULT_STOCKS))] + [f"{e}  (ETF)" for e in sorted(set(POPULAR_ETFS))]
            picked = st.selectbox("Ticker", _choices, help="Start typing to find a symbol")
            typed = st.text_input("Or type a ticker / name", placeholder="AAPL or APPLE")
            new_ticker = (typed or (picked.split()[0] if picked else "")).upper().strip()

        with c2:
            new_shares = st.number_input("Shares", min_value=0.0, value=1.0, step=1.0, format="%.4f")
        with c3:
            new_cost = st.number_input("Purchase Price ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        with c4:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("➕ Add / Update")

        if submitted:
            if not new_ticker:
                st.error("Please enter a ticker.")
            elif new_shares <= 0:
                st.error("Shares must be greater than zero.")
            elif new_cost < 0:
                st.error("Purchase price cannot be negative.")
            else:
                # Convert names like APPLE -> AAPL
                resolved = resolve_ticker(new_ticker)
                updated = False
                for h in st.session_state.portfolio:
                    if h["ticker"] == resolved or h["ticker"] == new_ticker:
                        h["ticker"] = resolved
                        h["shares"] = float(new_shares)
                        h["cost_basis"] = float(new_cost)
                        updated = True
                        break
                if not updated:
                    st.session_state.portfolio.append({
                        "ticker": resolved,
                        "shares": float(new_shares),
                        "cost_basis": float(new_cost),
                    })
                save_portfolio(st.session_state.portfolio)
                msg = f"{'Updated' if updated else 'Added'} {resolved}"
                if resolved != new_ticker:
                    msg += f" (from {new_ticker})"
                st.success(msg)
                st.rerun()

    # --- Current holdings table ---
    st.subheader("My Holdings")
    holdings = st.session_state.portfolio

    if not holdings:
        st.info("No holdings yet. Add your first stock or ETF above.")
    else:
        # Local name cleanup only (APPLE -> AAPL). No web lookup on page load.
        for h in holdings:
            raw = str(h.get("ticker", "")).strip().upper()
            if raw in COMMON_NAMES:
                h["ticker"] = COMMON_NAMES[raw]
        save_portfolio(st.session_state.portfolio)
        tickers_held = [str(h["ticker"]).strip().upper() for h in holdings]

        if "gino_quotes" not in st.session_state:
            st.session_state.gino_quotes = {}
        if "gino_quotes_time" not in st.session_state:
            st.session_state.gino_quotes_time = None

        # Keep Gino's Corner prices current when auto-refresh is on
        if refresh_sec > 0:
            st.session_state.gino_quotes = fetch_live_quotes(tickers_held)
            st.session_state.gino_quotes_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if st.button("🔄 Refresh Live Prices", key="refresh_gino_prices"):
            with st.spinner("Updating live prices from Nasdaq / CNBC…"):
                fresh = {}
                errors = []
                for sym in tickers_held:
                    try:
                        q = fetch_one_quote(sym)
                        if q:
                            fresh[sym] = q
                        else:
                            errors.append(f"{sym}: no quote returned")
                    except Exception as e:
                        errors.append(f"{sym}: {e}")
                st.session_state.gino_quotes = fresh
                st.session_state.gino_quote_errors = errors
                st.session_state.gino_quotes_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

        live = st.session_state.get("gino_quotes") or {}
        if st.session_state.get("gino_quotes_time"):
            st.caption(f"Last price update: {st.session_state.gino_quotes_time}")
        if live:
            try:
                maybe_send_drop_alerts(holdings, live, alerts)
            except Exception:
                pass

        if not live:
            st.info("Holdings are loaded. Click **Refresh Live Prices** to fill Live Price, Market Value, and Gain/Loss.")
        missing = [t for t in tickers_held if live and t not in live]
        if missing:
            st.warning(
                "Could not fetch a live price for: " + ", ".join(missing) +
                ". Click **Refresh Live Prices** again."
            )
        if live:
            srcs = sorted({(live[t].get("source") or "?") for t in live})
            st.success("Prices loaded from: " + ", ".join(srcs))
            with st.expander("See raw quote details"):
                st.write(live)
        errs = st.session_state.get("gino_quote_errors") or []
        if errs:
            with st.expander("Price lookup errors"):
                for e in errs:
                    st.write(e)



        rows = []
        total_cost = 0.0
        total_value = 0.0

        for h in holdings:
            t = str(h["ticker"]).strip().upper()
            shares = float(h["shares"])
            cost = float(h["cost_basis"])
            q = live.get(t, {})
            price = q.get("price")
            source = q.get("source")


            market_value = shares * price if price is not None else None
            cost_total = shares * cost
            gain_dol = (market_value - cost_total) if market_value is not None else None
            gain_pct = (gain_dol / cost_total * 100) if (gain_dol is not None and cost_total > 0) else None

            if cost_total:
                total_cost += cost_total
            if market_value is not None:
                total_value += market_value

            rows.append({
                "Ticker": t,
                "Shares": shares,
                "Cost/Share": cost,
                "Cost Basis": round(cost_total, 2),
                "Live Price": round(price, 2) if price is not None else None,
                "Market Value": round(market_value, 2) if market_value is not None else None,
                "Gain/Loss $": round(gain_dol, 2) if gain_dol is not None else None,
                "Gain/Loss %": round(gain_pct, 2) if gain_pct is not None else None,
                "Source": source or "",
            })

        pdf = pd.DataFrame(rows)
        st.dataframe(pdf, use_container_width=True, hide_index=True)

        # Totals
        total_gain = total_value - total_cost if total_value and total_cost else None
        total_gain_pct = (total_gain / total_cost * 100) if (total_gain is not None and total_cost > 0) else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Cost Basis", f"${total_cost:,.2f}")
        m2.metric("Total Market Value", f"${total_value:,.2f}" if total_value else "—")
        if total_gain is not None:
            m3.metric("Total Gain/Loss $", f"${total_gain:+,.2f}",
                      delta=f"{total_gain_pct:+.2f}%" if total_gain_pct is not None else None,
                      delta_color="normal" if total_gain >= 0 else "inverse")
        else:
            m3.metric("Total Gain/Loss $", "—")
        m4.metric("Holdings", len(holdings))

        # Edit / Remove
        st.markdown("#### Edit or Remove a Holding")
        edit_ticker = st.selectbox("Select ticker to edit or remove", [""] + [h["Ticker"] for h in rows])
        if edit_ticker:
            current = next((h for h in holdings if h["ticker"] == edit_ticker), None)
            if current:
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    e_shares = st.number_input("Shares", value=float(current["shares"]), min_value=0.0, step=1.0, key="edit_shares")
                with ec2:
                    e_cost = st.number_input("Purchase Price", value=float(current["cost_basis"]), min_value=0.0, step=0.01, key="edit_cost")
                with ec3:
                    st.write("")
                    st.write("")
                    if st.button("💾 Save Changes"):
                        current["shares"] = float(e_shares)
                        current["cost_basis"] = float(e_cost)
                        save_portfolio(st.session_state.portfolio)
                        st.success(f"Updated {edit_ticker}")
                        st.rerun()
                    if st.button("🗑️ Remove Holding"):
                        st.session_state.portfolio = [h for h in st.session_state.portfolio if h["ticker"] != edit_ticker]
                        save_portfolio(st.session_state.portfolio)
                        st.success(f"Removed {edit_ticker}")
                        st.rerun()

        # Download portfolio CSV
        csv_link(pdf, f"ginos_corner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "Download CSV")

    st.caption("Portfolio data is saved on your Desktop as ginos_corner_portfolio.json")

# ========================= MY WATCHLIST =========================
elif page == "My Watchlist":
    st.subheader("My Watchlist")
    st.caption("Choose symbols as you type. Keep stocks and ETFs on separate lists. Email when a name is 1% below the last close.")
    wl_cfg = load_alert_settings()
    if not isinstance(wl_cfg.get("watchlist_stocks"), list):
        wl_cfg["watchlist_stocks"] = parse_watchlist(wl_cfg.get("watchlist", ""))
    if not isinstance(wl_cfg.get("watchlist_etfs"), list):
        wl_cfg["watchlist_etfs"] = []

    stock_choices = sorted(set(DEFAULT_STOCKS + list(COMMON_NAMES.values())))
    etf_choices = sorted(set(POPULAR_ETFS))
    all_choices = [""] + [f"{s}  (Stock)" for s in stock_choices] + [f"{e}  (ETF)" for e in etf_choices]
    if "watchlist_stocks" not in st.session_state:
        st.session_state.watchlist_stocks = list(wl_cfg.get("watchlist_stocks") or [])
    if "watchlist_etfs" not in st.session_state:
        st.session_state.watchlist_etfs = list(wl_cfg.get("watchlist_etfs") or [])
    wl_cfg["watchlist_stocks"] = st.session_state.watchlist_stocks
    wl_cfg["watchlist_etfs"] = st.session_state.watchlist_etfs

    with st.form("watchlist_add_form", clear_on_submit=False):
        picked = st.selectbox("Type or choose a symbol", all_choices)
        typed = st.text_input("Or type a ticker").strip().upper()
        c1, c2 = st.columns(2)
        add_stock = c1.form_submit_button("Add to Stock watchlist")
        add_etf = c2.form_submit_button("Add to ETF watchlist")
    chosen = typed or (picked.split()[0] if picked else "")
    if add_stock or add_etf:
        if not chosen:
            st.error("Choose a symbol in the dropdown or type a ticker, then click Add.")
        else:
            if add_stock:
                if chosen not in st.session_state.watchlist_stocks:
                    st.session_state.watchlist_stocks.append(chosen)
                st.session_state.watchlist_etfs = [x for x in st.session_state.watchlist_etfs if x != chosen]
            else:
                if chosen not in st.session_state.watchlist_etfs:
                    st.session_state.watchlist_etfs.append(chosen)
                st.session_state.watchlist_stocks = [x for x in st.session_state.watchlist_stocks if x != chosen]
            wl_cfg["watchlist_stocks"] = st.session_state.watchlist_stocks
            wl_cfg["watchlist_etfs"] = st.session_state.watchlist_etfs
            save_alert_settings(wl_cfg)
            st.success(f"Added {chosen} to the {'stock' if add_stock else 'ETF'} watchlist")
            st.rerun()



    s_col, e_col = st.columns(2)
    with s_col:
        st.markdown("**Stock watchlist**")
        if not st.session_state.watchlist_stocks:
            st.caption("None yet")
        else:
            for t in list(st.session_state.watchlist_stocks):
                r1, r2 = st.columns([3, 1])
                r1.write(t)
                if r2.button("Remove", key=f"rm_stock_{t}"):
                    st.session_state.watchlist_stocks = [x for x in st.session_state.watchlist_stocks if x != t]
                    wl_cfg["watchlist_stocks"] = st.session_state.watchlist_stocks
                    wl_cfg["watchlist_etfs"] = st.session_state.watchlist_etfs
                    save_alert_settings(wl_cfg)
                    st.rerun()
    with e_col:
        st.markdown("**ETF watchlist**")
        if not st.session_state.watchlist_etfs:
            st.caption("None yet")
        else:
            for t in list(st.session_state.watchlist_etfs):
                r1, r2 = st.columns([3, 1])
                r1.write(t)
                if r2.button("Remove", key=f"rm_etf_{t}"):
                    st.session_state.watchlist_etfs = [x for x in st.session_state.watchlist_etfs if x != t]
                    wl_cfg["watchlist_stocks"] = st.session_state.watchlist_stocks
                    wl_cfg["watchlist_etfs"] = st.session_state.watchlist_etfs
                    save_alert_settings(wl_cfg)
                    st.rerun()


    watched = wl_cfg["watchlist_stocks"] + wl_cfg["watchlist_etfs"]
    if watched:
        quotes = fetch_live_quotes(watched)
        rows = []
        for t in watched:
            q = quotes.get(t) or {}
            rows.append({
                "Ticker": t,
                "List": "ETF" if t in wl_cfg["watchlist_etfs"] else "Stock",
                "Live Price": q.get("price"),
                "Day %": q.get("change_pct"),
                "Source": q.get("source"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    w1, w2, w3 = st.columns(3)
    with w1:
        wl_cfg["email"] = st.text_input("Alert email", value=wl_cfg.get("email", ""), key="wl_email")
    with w2:
        wl_cfg["email_on"] = st.checkbox("Email me when a name is down 1%", value=bool(wl_cfg.get("email_on")), key="wl_email_on")
    with w3:
        wl_cfg["threshold_pct"] = st.number_input("Drop % trigger", min_value=0.1, max_value=20.0, value=float(wl_cfg.get("threshold_pct") or 1.0), step=0.1, key="wl_thresh")
    if st.button("Save alert email settings", key="save_wl_email"):
        wl_cfg["watchlist_stocks"] = st.session_state.watchlist_stocks
        wl_cfg["watchlist_etfs"] = st.session_state.watchlist_etfs
        save_alert_settings(wl_cfg)
        st.success("Watchlist and email settings saved")


elif page == "Alerts":
    st.subheader("Alerts")
    st.caption("Drop alerts from your watchlist. Email is optional; alerts are stored here either way.")
    cfg = load_alert_settings()
    log = cfg.get("alert_log") or []
    if st.button("Clear alerts"):
        cfg["alert_log"] = []
        save_alert_settings(cfg)
        st.rerun()
    if not log:
        st.info("No alerts yet.")
    else:
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

# ========================= DISCOVER =========================
elif page == "Discover":
    st.subheader("Discover strong names")

    st.markdown("Scan for uptrends, healthy RSI, and strong 3-month momentum.")
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Universe", ["All (Stocks + ETFs)", "Stocks only", "ETFs only"], key="mode")
    with c2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1, key="period")
    with c3:
        min_score = st.slider("Min score", 0, 80, 50, key="min_score")
    if st.button("Find strong stocks & ETFs", type="primary"):
        tickers_u, etf_set_u = resolve_universe(mode, custom_input)
        with st.spinner(f"Scanning {len(tickers_u)} symbols on Discover…"):
            scanned = run_market_scan(tickers_u, period, min_score, etf_set_u)
        st.session_state["results_df"] = scanned
        if scanned.empty:
            st.warning("No symbols passed the filters. Lower Min score and try again.")
        else:
            st.success(f"Found {len(scanned)} ideas")
    df_prev = st.session_state.get("results_df")
    if df_prev is not None and not getattr(df_prev, "empty", True):
        show_cols = [c for c in ["ticker", "name", "type", "score", "action", "ret_3m_pct", "rsi"] if c in df_prev.columns]
        st.dataframe(df_prev[show_cols].head(20), use_container_width=True, hide_index=True)
    else:
        st.markdown("Click **Find strong stocks & ETFs**.")



# ========================= LEADERS =========================
elif page == "Leaders":
    st.subheader("Leaders")
    df_res = st.session_state.get("results_df")
    if df_res is None or getattr(df_res, "empty", True):
        st.info("Run a scan on the Discover or Screener tab first.")
    else:
        st.markdown("**Highest scores**")
        cols = [c for c in ["ticker", "name", "type", "score", "action", "ret_3m_pct", "rsi", "live_price", "live_change_pct"] if c in df_res.columns]
        st.dataframe(df_res[cols].head(12), use_container_width=True, hide_index=True)
        if "ret_3m_pct" in df_res.columns:
            st.markdown("**Best 3-month momentum**")
            st.dataframe(df_res.sort_values("ret_3m_pct", ascending=False)[cols].head(8), use_container_width=True, hide_index=True)
        if "rsi" in df_res.columns:
            setups = df_res[(df_res["rsi"] >= 40) & (df_res["rsi"] <= 62)]
            st.markdown("**Healthy RSI setups (40–62)**")
            if setups.empty:
                st.write("None in the last scan.")
            else:
                st.dataframe(setups[cols].head(8), use_container_width=True, hide_index=True)
        etfs = df_res[df_res.get("type", "") == "ETF"] if "type" in df_res.columns else pd.DataFrame()
        if not etfs.empty:
            st.markdown("**Strong ETFs**")
            st.dataframe(etfs[cols].head(8), use_container_width=True, hide_index=True)

# ========================= SELECTOR TAB =========================
elif page == "Screener":
    top_n = st.slider("Show top N", 5, 25, int(st.session_state.get("top_n", 10)), key="top_n")
    show_charts = st.checkbox("Show charts", value=True, key="show_charts")
    charts_count = st.slider("Charts", 1, 8, 4, key="charts_count") if show_charts else 0
    if st.button("Run screener", type="primary"):
        run_button = True
    if mode == "Custom tickers":
        raw = custom_input.replace(",", " ").upper().split()
        tickers = list(dict.fromkeys([t.strip() for t in raw if t.strip()]))
        etf_set = set(POPULAR_ETFS)
    elif mode == "ETFs only":
        tickers = POPULAR_ETFS.copy()
        etf_set = set(POPULAR_ETFS)
    elif mode == "Stocks only":
        tickers = DEFAULT_STOCKS.copy()
        etf_set = set()
    else:
        tickers = DEFAULT_STOCKS + POPULAR_ETFS
        etf_set = set(POPULAR_ETFS)

    if not tickers:
        st.error("No valid tickers.")
        st.stop()

    # Live quotes strip — updates on auto-refresh without a full analysis
    watch = tickers[:8]
    live_quotes = fetch_live_quotes(watch)
    st.session_state["selector_quotes"] = live_quotes
    st.session_state["selector_quotes_time"] = datetime.now().strftime("%H:%M:%S")
    if live_quotes:
        st.subheader("Live Prices (near real-time)")
        cols = st.columns(min(6, len(live_quotes)))
        for i, (t, q) in enumerate(list(live_quotes.items())[:6]):
            with cols[i]:
                delta_str = f"{q['change']:+.2f} ({q['change_pct']:+.2f}%)"
                st.metric(label=t, value=f"${q['price']:.2f}", delta=delta_str,
                          delta_color="normal" if q["change_pct"] >= 0 else "inverse")
        st.caption(
            f"Updated {st.session_state.get('selector_quotes_time', '')} from Nasdaq/CNBC. "
            "Auto-refresh is on. Weekend prices stay at Friday’s close."
        )

    # Full analysis — only when you click the button.
    # Reuse saved results on auto-refresh so the page stays fast.
    if run_button:
        skip_analysis = False
    elif "results_df" in st.session_state:
        df_res = st.session_state["results_df"]
        st.caption("Showing last analysis.")
        skip_analysis = True
    else:
        st.info("Use Discover or Run screener to scan symbols.")
        skip_analysis = True
        df_res = pd.DataFrame()
        results = None
        hist_data = {}


    if skip_analysis:
        hist_data = {}
        results = None
    else:
      with st.spinner(f"Analyzing {len(tickers)} symbols…"):
        hist_data = fetch_history(tickers, period=period)
        results = []
        progress = st.progress(0)
        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers))
            if ticker not in hist_data:
                continue
            try:
                df = compute_indicators(hist_data[ticker])
                is_etf = ticker in etf_set
                scored = score_symbol(df, ticker, is_etf=is_etf)
                if "error" in scored or scored["score"] < min_score:
                    continue
                rich = get_rich_info(ticker)
                if rich.get("type") in ("ETF", "Stock"):
                    scored["type"] = rich["type"]
                else:
                    scored["type"] = "ETF" if is_etf else "Stock"
                scored["name"] = rich.get("name") or ticker
                scored["sector"] = rich.get("sector")
                scored["industry"] = rich.get("industry")
                scored["category"] = rich.get("category")
                scored["market_cap"] = rich.get("market_cap")
                scored["trailing_pe"] = rich.get("trailing_pe")
                scored["dividend_yield"] = rich.get("dividend_yield")
                scored["52w_high"] = rich.get("52w_high")
                scored["52w_low"] = rich.get("52w_low")
                scored["description"] = rich.get("description")
                scored["recommendation"] = rich.get("recommendation")
                scored["target_mean"] = rich.get("target_mean")
                scored["num_analysts"] = rich.get("num_analysts")
                # Overlay live quote if available
                if ticker in live_quotes:
                    scored["live_price"] = live_quotes[ticker]["price"]
                    scored["live_change_pct"] = live_quotes[ticker]["change_pct"]
                results.append(scored)
            except Exception:
                continue
        progress.empty()

        if not results:
            st.warning("No symbols passed the filters.")
            df_res = pd.DataFrame()
        else:
            df_res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
            df_res.index = df_res.index + 1
            st.session_state["results_df"] = df_res
            st.session_state["run_scan"] = False

    if df_res is None or getattr(df_res, "empty", True):
        st.caption("No screener table yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Symbols analyzed", len(hist_data) if hist_data else len(df_res))
        c2.metric("Ideas shown", len(df_res))
        c3.metric("Top score", f"{df_res.iloc[0]['score']}")
        c4.metric("Updated", datetime.now().strftime("%H:%M:%S"))


    if df_res is None or getattr(df_res, "empty", True):
        pass
    else:
        st.subheader("Ranked Results")



    # Build display with live price if present
    show_cols = ["ticker", "name", "type", "sector", "score", "action", "price"]
    if "live_price" in df_res.columns:
        show_cols += ["live_price", "live_change_pct"]
    show_cols += ["ret_1m_pct", "ret_3m_pct", "ret_6m_pct", "rsi", "recommendation"]
    show_cols = [c for c in show_cols if c in df_res.columns]

    rename = {
        "ticker": "Ticker", "name": "Name", "type": "Type", "sector": "Sector / Category",
        "score": "Score", "action": "Action", "price": "Score Price",
        "live_price": "Live Price", "live_change_pct": "Day %",
        "ret_1m_pct": "1M %", "ret_3m_pct": "3M %", "ret_6m_pct": "6M %",
        "rsi": "RSI", "recommendation": "Analyst",
    }
    table_df = df_res[show_cols].rename(columns=rename)
    st.dataframe(table_df, use_container_width=True, height=380)

    csv_link(table_df, f"ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "Download ranking CSV")

    # Detailed profiles
    st.subheader("Detailed Profiles & Charts")
    for idx, row in df_res.head(max(top_n, charts_count)).iterrows():
        live_txt = ""
        if "live_price" in row and pd.notna(row.get("live_price")):
            chg = row.get("live_change_pct", 0) or 0
            live_txt = f"  •  Live ${row['live_price']:.2f} ({chg:+.2f}%)"
        with st.expander(f"#{idx}  {row['ticker']} — {row.get('name','')}  |  Score {row['score']} ({row['action']}){live_txt}",
                         expanded=(idx <= 2)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Type:** {row.get('type','—')}")
                st.markdown(f"**Sector / Category:** {row.get('sector') or row.get('category') or '—'}")
                if row.get("industry"): st.markdown(f"**Industry:** {row['industry']}")
                st.markdown(f"**Price (at score):** ${row['price']}")
                if "live_price" in row and pd.notna(row.get("live_price")):
                    st.markdown(f"**Live Price:** ${row['live_price']:.2f}  ({row.get('live_change_pct',0):+.2f}%)")
                st.markdown(f"**Market Cap:** {format_mktcap(row.get('market_cap'))}")
                pe = row.get("trailing_pe")
                st.markdown(f"**Trailing P/E:** {pe:.1f}" if pe else "**Trailing P/E:** —")
                dy = row.get("dividend_yield")
                st.markdown(f"**Div Yield:** {dy*100:.2f}%" if dy else "**Div Yield:** —")
                st.markdown(f"**52-Week:** {row.get('52w_low') or '—'} – {row.get('52w_high') or '—'}")
                if row.get("recommendation"):
                    st.markdown(f"**Analyst Lean:** {row['recommendation']}" +
                                (f" ({row['num_analysts']} analysts)" if row.get("num_analysts") else ""))
            with col_b:
                st.markdown("**Performance**")
                st.markdown(f"1M: {row['ret_1m_pct']}% 3M: {row['ret_3m_pct']}% "
                            f"6M: {row['ret_6m_pct']}% 12M: {row.get('ret_12m_pct','—')}%")
                st.markdown(f"**RSI:** {row['rsi']} **Volatility:** {row['volatility']}%")
                st.markdown("**Score Drivers**")
                for r in row.get("reasons", []):
                    st.markdown(f"- {r}")
            if row.get("description"):
                st.markdown("**Description**")
                st.write(row["description"][:1100] + ("…" if len(str(row["description"])) > 1100 else ""))
            if show_charts and idx <= charts_count:
                try:
                    fig = create_chart(row["ticker"], row["df"], row["score"], row["action"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart unavailable: {e}")

    # Full detailed CSV
    detail_rows = []
    for _, row in df_res.iterrows():
        detail_rows.append({
            "Ticker": row["ticker"], "Name": row.get("name"), "Type": row.get("type"),
            "Sector": row.get("sector"), "Industry": row.get("industry"),
            "Score": row["score"], "Action": row["action"], "Price": row["price"],
            "Live Price": row.get("live_price"), "Day %": row.get("live_change_pct"),
            "Market Cap": row.get("market_cap"), "Trailing PE": row.get("trailing_pe"),
            "Div Yield": row.get("dividend_yield"), "52W High": row.get("52w_high"),
            "52W Low": row.get("52w_low"), "1M %": row["ret_1m_pct"], "3M %": row["ret_3m_pct"],
            "6M %": row["ret_6m_pct"], "12M %": row.get("ret_12m_pct"), "RSI": row["rsi"],
            "Volatility %": row["volatility"], "Analyst": row.get("recommendation"),
            "Reasons": " | ".join(row.get("reasons", [])),
            "Description": (row.get("description") or "")[:400],
        })
    detail_df = pd.DataFrame(detail_rows)
    csv_link(detail_df, f"detailed_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "Download detailed CSV")

    st.markdown("---")
    st.caption("Educational tool only. Free market data is typically delayed. Not personalized investment advice.")

# ========================= EARNINGS & IPOS =========================
elif page == "Earnings & IPOs":
    st.subheader("Upcoming earnings")
    st.caption("Source: Nasdaq earnings calendar")
    try:
        earn = fetch_upcoming_earnings(8)
        if earn.empty:
            st.info("No earnings found for the next several market days.")
        else:
            st.dataframe(earn, use_container_width=True, hide_index=True)
            csv_link(earn, f"earnings_{datetime.now().strftime('%Y%m%d')}.csv", "Download earnings CSV")
    except Exception as e:
        st.warning(f"Earnings calendar unavailable: {e}")

    st.subheader("IPOs")
    st.caption("Source: Nasdaq IPO calendar for this month")
    try:
        ipo_map = fetch_ipo_calendar()
        for title, df in ipo_map.items():
            st.markdown(f"**{title}**")
            if df.empty:
                st.write("None listed right now.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"IPO calendar unavailable: {e}")

st.caption("Prices refresh about every 30 seconds.")


