"""AMW AI Custom Stock Screener™ - Streamlit dashboard (analysis only, no order execution).

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.patterns import NOT_IMPLEMENTED
from src.backtest.engine import run_backtest
from src.config import TIMEFRAMES, default_settings, validate_settings
from src.data.provider import DiskCache, YFinanceProvider
from src.data.universe import custom_universe, load_nifty500
from src.data.validation import clean_and_validate, usable
from src.screener.export import export_name, to_csv_bytes
from src.screener.scanner import ScanResult, analyse_symbol, apply_filters, rows_frame, run_scan
from src.ui.charts import build_chart

ROOT = Path(__file__).resolve().parent
logging.basicConfig(filename=ROOT / "amw_screener.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="AMW AI Custom Stock Screener™", layout="wide", page_icon="📈")

# ------------------------------------------------------------------ state
if "cfg" not in st.session_state:
    st.session_state.cfg = default_settings()
if "scan" not in st.session_state:
    st.session_state.scan = None
cfg = st.session_state.cfg


def provider() -> YFinanceProvider:
    return YFinanceProvider(cfg, DiskCache(ROOT / cfg["data"]["cache_dir"], cfg["data"]["ttl_minutes"]))


@st.cache_data(ttl=3600, show_spinner=False)
def get_universe(refresh_key: int):
    u = load_nifty500(default_settings(), ROOT, force_refresh=refresh_key > 0)
    return u.symbols, u.source, u.retrieved_at, u.is_fallback, u.errors


st.title("📈 AMW AI Custom Stock Screener™")
st.caption("फक्त तांत्रिक विश्लेषण · कोणतीही ऑर्डर दिली जात नाही · Score = research label, यशाची संभाव्यता नाही · डेटा: Yahoo Finance (delayed असू शकतो)")

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Universe")
uni_choice = sb.radio("Universe निवडा", ["Nifty 500", "Custom watchlist"], horizontal=True)
if "uni_refresh" not in st.session_state:
    st.session_state.uni_refresh = 0
if uni_choice == "Nifty 500":
    if sb.button("🔄 Constituent list refresh"):
        st.session_state.uni_refresh += 1
    sym_df, src, when, is_fb, errs = get_universe(st.session_state.uni_refresh)
    if is_fb:
        sb.error(f"⚠️ {src} — {len(sym_df)} symbols. अधिकृत Nifty 500 यादी मिळाली नाही. "
                 "NSE वरून `ind_nifty500list.csv` डाउनलोड करून `data/` फोल्डरमध्ये ठेवा आणि refresh करा.")
    else:
        sb.success(f"स्रोत: {src}\n\n{len(sym_df)} symbols · {when}")
    for e in errs:
        sb.caption(f"• {e}")
    limit = sb.number_input("Scan करायचे symbols (पहिले N; 0 = सर्व)", 0, len(sym_df), min(10, len(sym_df)), step=10)
    scan_df = sym_df if limit == 0 else sym_df.head(int(limit))
else:
    txt = sb.text_area("Symbols (comma/space; उदा. RELIANCE, TCS, INFY)", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK")
    cu = custom_universe(txt)
    for e in cu.errors:
        sb.warning(e)
    scan_df = cu.symbols
sym_count = len(scan_df)

tfs = sb.multiselect("Timeframes", TIMEFRAMES, default=TIMEFRAMES)
if not tfs:
    sb.error("किमान एक timeframe निवडा")
c1, c2 = sb.columns(2)
start = c1.button("▶️ Scan", type="primary", disabled=not tfs or sym_count == 0)
resume = c2.button("⏯️ Continue", disabled=st.session_state.scan is None or not st.session_state.scan.stopped_early)
sb.caption("Scan थांबवण्यासाठी वरच्या उजव्या कोपऱ्यातील **Stop** दाबा; पूर्ण झालेले batches जपले जातात व **Continue** ने पुढे चालतो.")
sb.caption(f"अंदाजे requests: {int(np.ceil(sym_count / cfg['data']['batch_size'])) * max(1, len(tfs))} (batch × timeframe)")

if start or resume:
    if start:
        st.session_state.scan = ScanResult(started=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"))
        st.session_state.scan_tfs = tfs
        st.session_state.scan_universe = scan_df
    res = st.session_state.scan
    res.stopped_early = True  # stays True if the run is interrupted
    bar = st.progress(0.0, text="Scan सुरू…")
    run_scan(st.session_state.scan_universe, st.session_state.scan_tfs, cfg, provider(),
             progress=lambda p, m: bar.progress(min(1.0, p), text=f"Scan: {m}"), resume=res)
    res.stopped_early = res.symbols_done < res.total
    bar.empty()

# ------------------------------------------------------------------ filters
sb.header("Filters")
res: ScanResult | None = st.session_state.scan
ranked = rows_frame(res) if res else pd.DataFrame()
f = {}
f["direction"] = sb.selectbox("दिशा", ["Both", "Bullish", "Bearish"])
f["min_score"] = sb.slider("Minimum score", 0, 100, 0)
f["min_coverage"] = sb.slider("Minimum evidence coverage %", 0, 100, 0)
f["setup_family"] = sb.multiselect("Setup family", ["Pullback", "Reversal", "Trend Continuation", "Breakout", "Pattern"])
if sb.checkbox("RSI range filter"):
    f["rsi"] = sb.slider("RSI", 0, 100, (30, 70))
if sb.checkbox("Relative volume filter"):
    f["min_rvol"] = sb.number_input("Min RVOL", 0.0, 20.0, 1.5, 0.1)
if sb.checkbox("Price range filter"):
    f["price"] = sb.slider("Price", 0, 100000, (50, 10000))
if sb.checkbox("Liquidity filter (avg volume, 20 bars)"):
    f["min_avg_volume"] = sb.number_input("Min avg volume", 0, 100_000_000, 100_000, 10_000)
if not ranked.empty and ranked.industry.fillna("").str.len().gt(0).any():
    f["industry"] = sb.multiselect("Sector/Industry (NSE list)", sorted(x for x in ranked.industry.dropna().unique() if x))
else:
    sb.caption("Sector filter: विश्वासार्ह sector डेटा उपलब्ध नाही (custom/fallback list).")
f["data_ok_only"] = sb.checkbox("फक्त सर्व timeframes OK डेटा")
f["search"] = sb.text_input("Symbol / company शोधा")
filtered = apply_filters(ranked, f) if not ranked.empty else ranked

tabs = st.tabs(["Overview", "Ranked Results", "Stock Detail", "Multi-Timeframe Matrix", "Data Quality", "Backtest (optional)", "Settings"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    if res is None:
        st.info("डावीकडून universe व timeframes निवडा आणि **Scan** दाबा. आधी 5-10 symbols वर चाचणी करा.")
    else:
        q = pd.DataFrame(res.quality)
        k = st.columns(4)
        k[0].metric("Scanned symbols", f"{res.symbols_done}/{res.total}")
        k[1].metric("Valid-data symbols", len(ranked))
        k[2].metric("Bullish candidates", int((ranked.direction == "Bullish").sum()) if len(ranked) else 0)
        k[3].metric("Bearish candidates", int((ranked.direction == "Bearish").sum()) if len(ranked) else 0)
        k = st.columns(4)
        k[0].metric("Strong candidates", int((ranked.label == "Strong Candidate").sum()) if len(ranked) else 0)
        k[1].metric("A1 candidates", int((ranked.label == "A1 Candidate").sum()) if len(ranked) else 0)
        k[2].metric("Data errors", int(q.status.isin(["UNAVAILABLE", "INVALID", "INSUFFICIENT", "ERROR"]).sum()) if len(q) else 0)
        k[3].metric("Stale data", int((q.status == "STALE").sum()) if len(q) else 0)
        st.caption(f"Scan सुरू: {res.started} · शेवट: {res.finished or '—'}" + (" · ⚠️ अर्धवट scan (Continue दाबा)" if res.stopped_early else ""))
        st.warning("Labels (A1/Strong/Watchlist) हे नियम-आधारित confluence दर्शवतात; ते नफ्याची संभाव्यता नाहीत. "
                   "Out-of-sample calibration केलेले नाही.")

# ------------------------------------------------------------------ ranked
with tabs[1]:
    if filtered.empty:
        st.info("निकाल नाहीत (scan करा किंवा filters सैल करा).")
    else:
        cols = ["symbol", "company", "direction", "score", "label", "coverage_pct", "setups", "alignment", "bars_time",
                "close", "rsi14", "rvol", "atr14", "atr_pct", "data_status", "missing_evidence", "warnings"]
        st.dataframe(filtered[cols], width="stretch", hide_index=True,
                     column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.1f"),
                                    "close": st.column_config.NumberColumn(format="%.2f"),
                                    "rsi14": st.column_config.NumberColumn(format="%.1f"),
                                    "rvol": st.column_config.NumberColumn(format="%.2f"),
                                    "atr14": st.column_config.NumberColumn(format="%.2f"),
                                    "atr_pct": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("⬇️ Filtered results CSV", to_csv_bytes(filtered), export_name("ranked_results"), "text/csv")
        with st.expander("Bullish / Bearish / Neutral स्वतंत्र ranking"):
            for d in ("Bullish", "Bearish", "Neutral"):
                sub = filtered[filtered.direction == d]
                st.markdown(f"**{d}** ({len(sub)})")
                st.dataframe(sub[["symbol", "score", "label", "setups", "coverage_pct"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ detail
with tabs[2]:
    options = list(filtered.ticker) if not filtered.empty else list(scan_df.ticker)
    if not options:
        st.info("Symbols नाहीत.")
    else:
        c = st.columns([2, 1, 1])
        tk = c[0].selectbox("Stock", options)
        dtf = c[1].selectbox("Timeframe", TIMEFRAMES, index=0)
        nb = c[2].number_input("Bars", 50, 1000, 200, 50)
        o = st.columns(6)
        show = {"ema": o[0].checkbox("EMA", True), "bb": o[1].checkbox("Bollinger"), "vwap": o[2].checkbox("VWAP"),
                "zones": o[3].checkbox("Zones", True), "sr": o[4].checkbox("S/R", True), "patterns": o[5].checkbox("Patterns", True),
                "swings": True, "refs": st.checkbox("Setup reference levels (entry/invalidation/target refs)", True)}
        raw = provider().fetch_many([tk], dtf).get(tk)
        analyses, reps = analyse_symbol({dtf: raw}, tk, [dtf], cfg)
        rep, a = reps[dtf], analyses[dtf]
        st.caption(f"Data status: **{rep.status}** · bars {rep.bars} · {rep.first} → {rep.last} · " + "; ".join(rep.issues))
        if a is None:
            st.error("या timeframe साठी वापरण्यायोग्य डेटा नाही — chart/analysis दाखवत नाही (डेटा बनवलेला नाही).")
        else:
            st.plotly_chart(build_chart(a, int(nb), show), width="stretch")
            st.caption("Entry/invalidation/target हे फक्त नियम-आधारित reference levels आहेत — trade instruction किंवा हमी नाही.")
            L = a.last
            m = st.columns(6)
            for col, (lab, key, fmt) in zip(m, [("Close", "Close", "{:.2f}"), ("RSI14", "rsi14", "{:.1f}"), ("RVOL", "rvol", "{:.2f}"),
                                                ("ATR%", "atr_pct", "{:.2f}"), ("ADX", "adx", "{:.1f}"), ("MACD hist", "macd_hist", "{:.2f}")]):
                col.metric(lab, fmt.format(L[key]) if L.get(key) is not None else "NA")
            st.markdown(f"**Trend:** {a.trend_label} ({a.trend_score:+.2f})" if a.trend_score is not None else "**Trend:** NA")
            st.markdown(f"**Structure trend (last close-based break):** {a.structure_trend or 'none'} · **Volatility:** {a.compression or 'NA'}")
            ix = a.df.index
            T = lambda i: str(ix[i])[:16]  # noqa: E731
            ev_rows = ([{"time": T(e.i), "type": e.type, "detail": f"{e.direction} close beyond {e.level:.2f}"} for e in a.events[-6:]]
                       + [{"time": T(c_["i"]), "type": "Candle", "detail": f"{c_['pattern']} {c_['bias']} [{c_['status']}] confirm: {c_['confirm']} · invalid: {c_['invalid']}"} for c_ in a.candles]
                       + [{"time": T(s["i"]), "type": "Liquidity sweep", "detail": f"{s['side']} @ {s['level']:.2f} (hypothesis)"} for s in a.liquidity["sweeps"][-4:]])
            st.subheader("Timestamped evidence")
            st.dataframe(pd.DataFrame(ev_rows).sort_values("time", ascending=False) if ev_rows else pd.DataFrame(), hide_index=True, width="stretch")
            st.subheader("Setups")
            st.dataframe(pd.DataFrame(a.setups) if a.setups else pd.DataFrame({"info": ["No setup"]}), hide_index=True, width="stretch")
            st.subheader("Patterns")
            if a.patterns:
                st.dataframe(pd.DataFrame([{k: v for k, v in p.items() if k != "points"} for p in a.patterns]), hide_index=True, width="stretch")
            st.caption(f"Pattern detection मर्यादा: नियम-आधारित heuristics; {', '.join(NOT_IMPLEMENTED)} implement केलेले नाही.")
            st.subheader("Zones & S/R")
            st.dataframe(pd.DataFrame(a.zones + a.sr), hide_index=True, width="stretch")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ OHLCV + indicators CSV", to_csv_bytes(a.df), export_name(f"{tk}_{dtf}_ohlcv"), "text/csv")
            an = pd.DataFrame(ev_rows + [{"time": s["time"], "type": f"Setup {s['setup']}", "detail": f"{s['direction']} [{s['status']}] " + " | ".join(s["evidence"])} for s in a.setups])
            d2.download_button("⬇️ Analysis CSV", to_csv_bytes(an), export_name(f"{tk}_{dtf}_analysis"), "text/csv")

# ------------------------------------------------------------------ matrix
with tabs[3]:
    if res and res.matrix:
        mdf = pd.DataFrame(res.matrix)
        if not filtered.empty:
            mdf = mdf[mdf.ticker.isin(filtered.ticker)]
        st.dataframe(mdf, hide_index=True, width="stretch")
        st.download_button("⬇️ Multi-timeframe matrix CSV", to_csv_bytes(mdf), export_name("mtf_matrix"), "text/csv")
    else:
        st.info("Scan नंतर matrix दिसेल.")

# ------------------------------------------------------------------ quality
with tabs[4]:
    if res and res.quality:
        q = pd.DataFrame(res.quality)
        only_bad = st.checkbox("फक्त समस्या असलेले", True)
        if only_bad:
            q = q[(q.status != "OK") | (q.issues.fillna("") != "")]
        st.dataframe(q, hide_index=True, width="stretch")
        st.download_button("⬇️ Data quality CSV", to_csv_bytes(pd.DataFrame(res.quality)), export_name("data_quality"), "text/csv")
        if res.evidence:
            st.download_button("⬇️ Setup evidence & score explanation CSV", to_csv_bytes(pd.DataFrame(res.evidence)),
                               export_name("setup_evidence"), "text/csv")
    else:
        st.info("Scan नंतर डेटा-गुणवत्ता अहवाल दिसेल.")

# ------------------------------------------------------------------ backtest
with tabs[5]:
    st.warning("Backtest हा scanner score पासून वेगळा आहे. Survivorship bias (सध्याचे constituents), Yahoo डेटा त्रुटी, "
               "आणि गृहितके (next-bar open entry, stop-first) लक्षात घ्या. भूतकाळ ≠ भविष्य.")
    c = st.columns(4)
    btk = c[0].selectbox("Stock", list(scan_df.ticker) or ["RELIANCE.NS"], key="bt_tk")
    btf = c[1].selectbox("Timeframe", ["1D", "1H"], key="bt_tf")
    fam = c[2].multiselect("Setups", ["Pullback", "Reversal", "Trend Continuation", "Breakout", "Pattern"], key="bt_fam")
    stepb = c[3].number_input("Evaluation step (bars)", 1, 10, 1)
    if st.button("Backtest चालवा"):
        raw = provider().fetch_many([btk], btf).get(btk)
        clean, rep = clean_and_validate(raw, btk, btf, cfg)
        if not usable(rep):
            st.error(f"डेटा वापरण्यायोग्य नाही: {rep.status}")
        else:
            with st.spinner("प्रत्येक bar वर फक्त भूतकाळातील डेटा वापरून पुन्हा analysis… (काही मिनिटे लागू शकतात)"):
                bt = run_backtest(clean, btf, cfg, fam or None, step=int(stepb))
            st.write(f"Train/Test split: {bt['split_time']}")
            st.dataframe(pd.DataFrame({k: bt[k] for k in ("all", "train", "test")}).T, width="stretch")
            st.write(f"Benchmark (buy & hold, same period): {bt['benchmark_buy_hold_pct']}%")
            st.dataframe(bt["trades"], hide_index=True, width="stretch")
            with st.expander("गृहितके"):
                st.code(bt["assumptions"])

# ------------------------------------------------------------------ settings
with tabs[6]:
    st.subheader("Scoring weights")
    new = copy.deepcopy(cfg)
    wc = st.columns(3)
    for i, (k, v) in enumerate(cfg["scoring"]["weights"].items()):
        new["scoring"]["weights"][k] = wc[i % 3].number_input(k, 0, 100, int(v), key=f"w_{k}")
    st.caption(f"एकूण weight: {sum(new['scoring']['weights'].values())} (score नेहमी 0-100 मध्ये normalize होतो)")
    st.subheader("Labels")
    lc = st.columns(3)
    for i, k in enumerate(("a1", "strong", "watch")):
        new["scoring"]["labels"][k] = lc[i].number_input(k, 0, 100, int(cfg["scoring"]["labels"][k]), key=f"l_{k}")
    st.subheader("Analysis thresholds")
    ac = st.columns(3)
    a = new["analysis"]
    a["swing_k"] = ac[0].number_input("swing_k", 1, 10, int(a["swing_k"]))
    a["breakout_rvol"] = ac[1].number_input("breakout_rvol", 0.5, 10.0, float(a["breakout_rvol"]), 0.1)
    a["zone_proximity_atr"] = ac[2].number_input("zone_proximity_atr", 0.1, 5.0, float(a["zone_proximity_atr"]), 0.1)
    a["max_pullback_distance_atr"] = ac[0].number_input("max_pullback_distance_atr", 0.1, 5.0, float(a["max_pullback_distance_atr"]), 0.1)
    a["pattern_tolerance_atr"] = ac[1].number_input("pattern_tolerance_atr", 0.1, 3.0, float(a["pattern_tolerance_atr"]), 0.1)
    a["require_retest"] = ac[2].checkbox("Breakout साठी retest आवश्यक", bool(a["require_retest"]))
    a["primary_timeframe"] = ac[0].selectbox("Primary timeframe", TIMEFRAMES, index=TIMEFRAMES.index(a["primary_timeframe"]))
    st.subheader("Data / rate limits")
    dc = st.columns(3)
    new["data"]["batch_size"] = dc[0].number_input("batch_size", 1, 100, int(cfg["data"]["batch_size"]))
    new["data"]["pause_between_batches"] = dc[1].number_input("pause (s)", 0.0, 30.0, float(cfg["data"]["pause_between_batches"]), 0.5)
    new["data"]["max_retries"] = dc[2].number_input("max_retries", 0, 5, int(cfg["data"]["max_retries"]))
    b1, b2 = st.columns(2)
    if b1.button("✅ Settings लागू करा"):
        try:
            validate_settings(new)
            st.session_state.cfg = new
            st.success("लागू झाले. नवीन scores साठी पुन्हा Scan करा.")
        except ValueError as e:
            st.error(f"अवैध settings: {e}")
    if b2.button("↩️ Restore defaults"):
        st.session_state.cfg = default_settings()
        for k in [k for k in st.session_state.keys() if k.startswith(("w_", "l_"))]:
            del st.session_state[k]
        st.rerun()
