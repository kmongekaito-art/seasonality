# -*- coding: utf-8 -*-
"""
S&P500 Hybrid Seasonal System (Markos Katsanos, Stocks & Commodities V.40:04)
Backtest en Streamlit con comparación vs Buy & Hold del SPY.
Implementación fiel del código AFL de AmiBroker del artículo y de los
documentos SEASONAL-LONG / SEASONAL-SELL.

Ejecutar:  streamlit run app.py
"""
import io
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Hybrid Seasonal System", page_icon="📈", layout="wide")


def _st_major_minor() -> tuple[int, int]:
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (1, 32)


_MODERN_WIDTH = _st_major_minor() >= (1, 50)


def show_plotly(fig, **kwargs):
    if _MODERN_WIDTH:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    return st.plotly_chart(fig, **kwargs)


def show_dataframe(data, **kwargs):
    if _MODERN_WIDTH:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    return st.dataframe(data, **kwargs)


MESES = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
         7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
         11: "Noviembre", 12: "Diciembre"}

# ============================================================================
# 1. SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuración")

    st.header("📊 Datos")
    ticker = st.text_input("Ticker subyacente", "SPY")
    vix_ticker = st.text_input("Ticker del VIX", "^VIX")
    hoy = dt.date.today()
    rango_fechas = st.date_input(
        "Rango de datos (desde - hasta)",
        (dt.date(1994, 1, 1), hoy),
        min_value=dt.date(1990, 1, 1), max_value=hoy)
    if isinstance(rango_fechas, (tuple, list)) and len(rango_fechas) == 2:
        fecha_ini, fecha_fin = rango_fechas
    else:
        fecha_ini = rango_fechas[0] if isinstance(rango_fechas, (tuple, list)) else rango_fechas
        fecha_fin = hoy

    st.header("🧭 Estrategia")
    sell_month = st.selectbox("Mes de salida estacional (SELLMONTH)",
                              list(MESES.keys()), index=7,
                              format_func=lambda m: MESES[m])
    vix_up_max = st.number_input("Umbral subida VIX % (VIXUPMAX)", 10.0, 150.0, 60.0, 5.0)
    k_mult = st.number_input("Multiplicador ATR/VIX (K)", 0.5, 3.0, 1.5, 0.05)
    crit_vfi = st.number_input("Nivel crítico VFI (CRIT)", -100.0, 0.0, -20.0, 5.0)

    st.header("💰 Backtest")
    capital = st.number_input("Capital inicial ($)", 1000.0, 1e9, 100_000.0, 1000.0)
    pct_equity = st.slider("% del capital por operación", 10, 100, 100)
    comision = st.number_input("Comisión ($ por acción)", 0.0, 1.0, 0.01, 0.01)
    usar_ajustados = st.checkbox("Precios ajustados por dividendo", False)

    st.header("📐 Indicadores del gráfico")
    INDICADORES = ["VFI (130) + MA(VFI,10)", "VIX", "VIXUP % (subida VIX 25d)",
                   "ATR (15)", "ATRUP % (subida ATR 25d)", "RSI (14)", "Volumen"]
    n_ind = st.number_input("Número de indicadores", 0, 6, 3)
    sel_ind = []
    for i in range(int(n_ind)):
        sel_ind.append(st.selectbox(f"Indicador {i+1}", INDICADORES,
                                    index=min(i, len(INDICADORES) - 1),
                                    key=f"ind_sel_{i}"))

    st.header("🎨 Gráfico")
    rango_grafico = st.selectbox("Rango mostrado", ["Máximo", "20 años", "10 años",
                                                    "5 años", "2 años"], index=2)
    mostrar_marcas = st.checkbox("Marcas BUY / SELL", True)
    sma_velas = st.checkbox("SMA 50/200 sobre velas", False)
    vix_velas = st.checkbox("VIX sobre velas (eje derecho)", False)


# ============================================================================
# 2. DATOS
# ============================================================================
@st.cache_data(ttl=3600, show_spinner="Descargando datos de Yahoo Finance…")
def fetch(sym: str, vix_sym: str, ini: dt.date, fin: dt.date,
          ajustados: bool) -> pd.DataFrame:
    def _dl(s: str) -> pd.DataFrame:
        df = yf.download(s, start=ini, end=fin + pd.Timedelta(days=1),
                         progress=False, auto_adjust=ajustados)
        if df is None or df.empty:
            raise ValueError(f"No hay datos para {s}")
        if isinstance(df.columns, pd.MultiIndex):
            for level in range(df.columns.nlevels):
                names = set(df.columns.get_level_values(level))
                if {"Open", "High", "Low", "Close"}.issubset(names):
                    df = df.copy()
                    df.columns = df.columns.get_level_values(level)
                    break
            else:
                if s in df.columns.get_level_values(-1):
                    df = df.xs(s, axis=1, level=-1)
                else:
                    df = df.droplevel(-1, axis=1)
        return df

    px = _dl(sym)
    vx = _dl(vix_sym)
    df = px[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["VIX"] = vx["Close"].reindex(df.index).ffill()
    return df.dropna(subset=["Open", "High", "Low", "Close", "VIX"])


# ============================================================================
# 3. INDICADORES (traducción fiel del AFL)
# ============================================================================
def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    """ATR de AmiBroker: suavizado de Wilder."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_indicadores(df: pd.DataFrame, vfi_period: int = 130) -> pd.DataFrame:
    d = df.copy()
    d["Month"] = d.index.month

    # --- Volatilidad relativa (VIX y ATR), mínimos/máximos de 25 días ----------
    vix = d["VIX"]
    d["VIXUP"] = (vix / vix.rolling(25).min().shift(1) - 1) * 100      # Ref(LLV,-1)
    d["VIXDN"] = (vix / vix.rolling(25).max().shift(1) - 1) * 100      # Ref(HHV,-1)
    atr15 = wilder_atr(d, 15)
    d["ATR"] = atr15
    d["ATRUP"] = (atr15 / atr15.rolling(25).min().shift(1) - 1) * 100
    d["ATRDN"] = (atr15 / atr15.rolling(25).max().shift(1) - 1) * 100

    # --- VFI de Katsanos (130 días) --------------------------------------------
    avg = (d["High"] + d["Low"] + d["Close"]) / 3.0                   # Avg de AFL
    inter = np.log(avg) - np.log(avg.shift(1))
    vinter = inter.rolling(30).std(ddof=0)
    cutoff = 0.2 * vinter * d["Close"]
    vave = d["Volume"].rolling(vfi_period).mean().shift(1)            # Ref(MA(V,p),-1)
    vc = np.minimum(d["Volume"], 2.5 * vave)
    mf = avg - avg.shift(1)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0))
    vfi1 = pd.Series(vcp, index=d.index).rolling(vfi_period).sum() / vave
    d["VFI"] = vfi1.ewm(span=3, adjust=False).mean()                  # EMA(VFI1,3)
    d["MA_VFI_10"] = d["VFI"].rolling(10).mean()

    # --- Extras -------------------------------------------------------------------
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    return d


def construir_senales(d: pd.DataFrame, sell_month: int, vix_up_max: float,
                      k_mult: float, crit: float) -> pd.DataFrame:
    d = d.copy()
    volcond = (((d["VIXUP"] < vix_up_max) | (d["ATRUP"] < k_mult * vix_up_max))
               & (d["VFI"] > crit))                                   # VOLCONDITION
    ventana = (d["Month"] >= 10) | (d["Month"] < sell_month)
    d["VENTANA"] = ventana
    d["VOLCONDITION"] = volcond
    # BUY = ventana estacional AND Ref(VOLCONDITION, -1)
    d["BUY"] = ventana & volcond.shift(1).fillna(False)
    d["SELL_SEASONAL"] = d["Month"] == sell_month
    # SELLVOLATILITY = VIXUP > 2 * VIXUPMAX   (pánico de volatilidad)
    d["PANIC_VIX"] = d["VIXUP"] > 2 * vix_up_max
    # SELLMF = Cross(CRIT, VFI) AND MA(VFI,10) descendente → el VFI cruza bajo CRIT
    d["PANIC_VFI"] = ((d["VFI"] < crit) & (d["VFI"].shift(1) >= crit)
                      & (d["MA_VFI_10"] < d["MA_VFI_10"].shift(1)))
    d["PANIC_EXIT"] = d["PANIC_VIX"] | d["PANIC_VFI"]
    return d


# ============================================================================
# 4. MOTOR DE BACKTEST
# ============================================================================
def run_backtest(d: pd.DataFrame, capital: float, pct_equity: int,
                 comision: float) -> tuple[pd.Series, pd.DataFrame, dict | None, float]:
    """Posición única larga (% del equity por defecto, entero por número de acciones):
    · Compra: ventana estacional + VOLCONDITION del cierre anterior → apertura del día.
    · Venta estacional: primer día hábil de SELLMONTH a la apertura.
    · Venta por pánico (VIX / VFI): condición al cierre → apertura del día siguiente.
    Devuelve (equity diario serie, tabla operaciones, posición abierta, exposición %).
    """
    opens = d["Open"].to_numpy(float)
    closes = d["Close"].to_numpy(float)
    months_v = d["Month"].to_numpy(int)
    buy = d["BUY"].to_numpy(bool)
    panic = d["PANIC_EXIT"].to_numpy(bool)
    panic_vix = d["PANIC_VIX"].to_numpy(bool)
    fechas = d.index
    n = len(d)
    first_day_of_month = np.r_[True, months_v[1:] != months_v[:-1]]

    sh_series = np.zeros(n, dtype=float)   # acciones al cierre de cada barra
    cash_series = np.zeros(n)
    equity_series = np.empty(n)

    sh, cash = 0, float(capital)
    entry = None
    pending = None
    trades: list[dict] = []

    def _cerrar(i: int, o: float, motivo: str):
        nonlocal sh, cash, entry, pending
        cash_out = sh * o - sh * comision
        trades[-1].update({
            "Fecha salida": fechas[i].date(),
            "Precio salida": round(o, 4),
            "Motivo salida": motivo,
            "P&L ($)": round(cash_out - entry["inversion"], 2),
            "Rentabilidad (%)": round((o / entry["precio"] - 1) * 100, 2),
            "Comisión total ($)": round(sh * comision * 2, 2),
            "Días en posición": int((fechas[i] - entry["fecha"]).days)})
        cash += cash_out
        sh, entry, pending = 0, None, None

    for i in range(n):
        o = opens[i]

        # 1) salida pendiente por pánico → apertura de hoy
        if sh > 0 and pending is not None:
            _cerrar(i, o, pending)
        # 2) venta estacional: primer día hábil del mes de salida, a la apertura
        elif sh > 0 and first_day_of_month[i] and months_v[i] == int(sell_month):
            _cerrar(i, o, "SELL SEASONAL")

        # 3) pánico al cierre de hoy → venta mañana a la apertura
        if sh > 0 and panic[i]:
            pending = "SELL VOLAT" if panic_vix[i] else "SELL MF"

        # 4) compra a la apertura de hoy (usa VOLCONDITION del cierre anterior)
        if sh == 0 and buy[i]:
            budget = cash * pct_equity / 100
            n_sh = int(budget // (o + comision)) if (o + comision) > 0 else 0
            if n_sh >= 1:
                cost = n_sh * o + n_sh * comision
                cash -= cost
                sh = n_sh
                entry = {"fecha": fechas[i], "precio": o, "inversion": cost}
                trades.append({"Fecha entrada": fechas[i].date(),
                               "Precio entrada": round(o, 4),
                               "Acciones": n_sh,
                               "Inversión ($)": round(cost, 2)})

        equity_series[i] = cash + sh * closes[i]
        cash_series[i], sh_series[i] = cash, sh

    eq = pd.Series(equity_series, index=d.index, name="Equity")

    open_pos = None
    if trades and sh > 0 and entry is not None:
        last_c = closes[-1]
        mv = sh * last_c - sh * comision
        open_pos = {"Fecha entrada": trades[-1]["Fecha entrada"],
                    "Precio entrada": trades[-1]["Precio entrada"],
                    "Acciones": int(sh),
                    "Inversión ($)": trades[-1]["Inversión ($)"],
                    "Fecha salida": "(pos. abierta)",
                    "Precio salida": round(last_c, 4),
                    "Motivo salida": "FIN DE DATOS",
                    "P&L ($)": round(mv - entry["inversion"], 2),
                    "Rentabilidad (%)": round((last_c / entry["precio"] - 1) * 100, 2),
                    "Comisión total ($)": round(sh * comision * 2, 2),
                    "Días en posición": int((fechas[-1] - entry["fecha"]).days)}

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        tdf = pd.DataFrame(columns=[
            "Fecha entrada", "Precio entrada", "Acciones", "Inversión ($)",
            "Fecha salida", "Precio salida", "Motivo salida", "P&L ($)",
            "Rentabilidad (%)", "Días en posición", "Comisión total ($)",
        ])
    # la última operación sin cerrar se representa únicamente como pos. abierta
    if open_pos is not None and not tdf.empty:
        tdf = tdf.iloc[:-1].copy()
    if not tdf.empty:
        tdf = tdf[["Fecha entrada", "Precio entrada", "Acciones", "Inversión ($)",
                   "Fecha salida", "Precio salida", "Motivo salida",
                   "P&L ($)", "Rentabilidad (%)", "Días en posición",
                   "Comisión total ($)"]]
    exposure = float((sh_series > 0).mean() * 100)
    return eq, tdf, open_pos, exposure


# ============================================================================
# 5. MÉTRICAS
# ============================================================================
def metricas(eq: pd.Series, capital: float, tdf: pd.DataFrame) -> dict:
    ret = eq.pct_change(fill_method=None).dropna()
    años = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    car = (eq.iloc[-1] / capital) ** (1 / años) - 1
    dd = (eq / eq.cummax() - 1) * 100
    ulcer = float(np.sqrt((dd ** 2).mean()))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    cerradas = tdf.loc[~tdf["Fecha salida"].astype(str).str.contains("abierta")] \
        if not tdf.empty else tdf
    gan = (cerradas["P&L ($)"] > 0).mean() * 100 if not cerradas.empty else np.nan
    wins = cerradas.loc[cerradas["P&L ($)"] > 0, "P&L ($)"].sum() if not cerradas.empty else 0.0
    losses = cerradas.loc[cerradas["P&L ($)"] < 0, "P&L ($)"].sum() if not cerradas.empty else 0.0
    pf = wins / abs(losses) if abs(losses) > 0 else np.nan
    return {
        "Beneficio neto ($)": round(float(eq.iloc[-1] - capital), 0),
        "CAR (rent. anual comp.)": round(car * 100, 2),
        "Sharpe (anualizado)": round(sharpe, 2),
        "Max. DD (%)": round(dd.min(), 1),
        "Ulcer Index": round(ulcer, 2),
        "Nº de operaciones": len(tdf),
        "% Operaciones ganadoras": gan,
        "Profit Factor": pf,
    }


# ============================================================================
# 6. GRÁFICO: velas + paneles de indicadores
# ============================================================================
def vfi_colors(v: pd.Series) -> list:
    out = []
    for x in v:
        if pd.isna(x):
            out.append("rgba(0,0,0,0)")
        elif x > 0:
            out.append("#00b26b")
        elif x >= -20:
            out.append("#ff9f1c")
        else:
            out.append("#e63946")
    return out


def plot_principal(d, tdf, sel_ind, crit, vix_up_max, k_mult, marcas, sma_velas,
                   vix_velas, sym) -> go.Figure:
    n_pan = len(sel_ind)
    fig = make_subplots(
        rows=1 + n_pan, cols=1, shared_xaxes=True,
        row_heights=[0.55] + [0.45 / n_pan] * n_pan if n_pan else [1.0],
        vertical_spacing=0.04,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * n_pan)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=sym, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False), row=1, col=1, secondary_y=False)

    if sma_velas:
        for col_sma, nom, c in [("SMA50", "SMA 50", "#4472c4"),
                                ("SMA200", "SMA 200", "#ed7d31")]:
            fig.add_trace(go.Scatter(x=d.index, y=d[col_sma], name=nom,
                                     line=dict(color=c, width=1.2),
                                     showlegend=False), row=1, col=1, secondary_y=False)
    if vix_velas:
        fig.add_trace(go.Scatter(x=d.index, y=d["VIX"], name="VIX",
                                 line=dict(color="#000", width=1),
                                 showlegend=False), row=1, col=1, secondary_y=True)

    # ---- marcas BUY / SELL ------------------------------------------------------
    if marcas and not tdf.empty:
        entradas = pd.to_datetime(tdf["Fecha entrada"])
        salidas = pd.to_datetime(
            tdf.loc[tdf["Fecha salida"].astype(str) != "(pos. abierta)", "Fecha salida"],
            errors="coerce").dropna()
        e_px = d.loc[d.index.isin(entradas), "Low"] * 0.985
        s_px = d.loc[d.index.isin(salidas), "High"] * 1.015
        fig.add_trace(go.Scatter(x=e_px.index, y=e_px, mode="markers",
                                 marker=dict(symbol="triangle-up", size=13,
                                             color="#1565c0"),
                                 name="Entrada", showlegend=False), row=1, col=1,
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=s_px.index, y=s_px, mode="markers",
                                 marker=dict(symbol="triangle-down", size=13,
                                             color="#c62828"),
                                 name="Salida", showlegend=False), row=1, col=1,
                      secondary_y=False)

    # ---- paneles de indicadores ------------------------------------------------
    r = 2
    for ind in sel_ind:
        if ind == "VFI (130) + MA(VFI,10)":
            fig.add_trace(go.Bar(x=d.index, y=d["VFI"], name="VFI",
                                 marker_color=vfi_colors(d["VFI"]),
                                 marker_line_width=0, showlegend=False), row=r, col=1)
            fig.add_trace(go.Scatter(x=d.index, y=d["MA_VFI_10"], name="MA(VFI,10)",
                                     line=dict(color="#1f77b4", width=1.2),
                                     showlegend=False), row=r, col=1)
            fig.add_hline(y=crit, line=dict(color="red", dash="dash", width=1),
                          row=r, col=1)
        elif ind == "VIX":
            fig.add_trace(go.Scatter(x=d.index, y=d["VIX"], name="VIX",
                                     line=dict(color="#222", width=1),
                                     showlegend=False), row=r, col=1)
        elif ind == "VIXUP % (subida VIX 25d)":
            fig.add_trace(go.Scatter(x=d.index, y=d["VIXUP"], name="VIXUP %",
                                     line=dict(color="#9467bd", width=1),
                                     showlegend=False), row=r, col=1)
            fig.add_hline(y=vix_up_max, line=dict(color="orange", dash="dot", width=1),
                          row=r, col=1)
            fig.add_hline(y=2 * vix_up_max, line=dict(color="red", dash="dash", width=1),
                          row=r, col=1)
        elif ind == "ATR (15)":
            fig.add_trace(go.Scatter(x=d.index, y=d["ATR"], name="ATR 15",
                                     line=dict(color="#2ca02c", width=1),
                                     showlegend=False), row=r, col=1)
        elif ind == "ATRUP % (subida ATR 25d)":
            fig.add_trace(go.Scatter(x=d.index, y=d["ATRUP"], name="ATRUP %",
                                     line=dict(color="#8c564b", width=1),
                                     showlegend=False), row=r, col=1)
            fig.add_hline(y=k_mult * vix_up_max,
                          line=dict(color="red", dash="dash", width=1), row=r, col=1)
        elif ind == "RSI (14)":
            fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                                     line=dict(color="#e377c2", width=1),
                                     showlegend=False), row=r, col=1)
            for lvl, c in [(70, "red"), (30, "green")]:
                fig.add_hline(y=lvl, line=dict(color=c, dash="dot", width=0.8),
                              row=r, col=1)
        elif ind == "Volumen":
            fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volumen",
                                 marker_color="#9ec9e8", showlegend=False), row=r, col=1)
        fig.update_yaxes(title_text=ind, title_font_size=10, row=r, col=1)
        r += 1

    fig.update_layout(title="Registros diarios: velas + señal + indicadores",
                      height=500 + 190 * n_pan,
                      margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified", template="plotly_white")
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig


# ============================================================================
# 7. MAIN
# ============================================================================
st.title("📈 S&P500 Hybrid Seasonal System")
st.caption("Markos Katsanos · *Stocks & Commodities* V.40:04 (2022) — código AFL "
           "implementado sobre datos diarios de Yahoo Finance")

if fecha_fin <= fecha_ini:
    st.error("La fecha final debe ser posterior a la inicial.")
    st.stop()

try:
    df_raw = fetch(ticker.strip(), vix_ticker.strip(), fecha_ini, fecha_fin,
                   bool(usar_ajustados))
except Exception as e:
    st.error(f"Error descargando datos: {e}")
    st.stop()

df = construir_senales(compute_indicadores(df_raw), int(sell_month),
                       float(vix_up_max), float(k_mult), float(crit_vfi))

start_idx = df["VFI"].first_valid_index()
if start_idx is None or df.loc[start_idx:].shape[0] < 30:
    st.error("Rango demasiado corto para calcular los indicadores "
             "(se necesita ~130 días de calentamiento).")
    st.stop()
d = df.loc[start_idx:]

eq, tdf, open_pos, exposure = run_backtest(d, float(capital), int(pct_equity),
                                           float(comision))

# --- Buy & Hold en el mismo rango -------------------------------------------------
bh_sh = int(float(capital) // (d["Open"].iloc[0] + comision))
bh_rest = float(capital) - bh_sh * d["Open"].iloc[0] - bh_sh * comision
eq_bh = pd.Series(bh_sh * d["Close"].to_numpy() + bh_rest, index=d.index,
                  name="Buy & Hold")

uras = {"Beneficio neto ($)": "${:,.0f}", "CAR (rent. anual comp.)": "{:.2f} %",
        "Sharpe (anualizado)": "{:.2f}", "Max. DD (%)": "{:.1f} %",
        "Ulcer Index": "{:.2f}", "Nº de operaciones": "{:d}",
        "% Operaciones ganadoras": "{:.1f} %", "Profit Factor": "{:.2f}"}

m_sys = metricas(eq, float(capital), tdf)
m_bh = metricas(eq_bh, float(capital), pd.DataFrame(
    columns=["Fecha salida", "P&L ($)", "Rentabilidad (%)"]))


def fmt(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        f = uras[k]
        out[k] = "—" if pd.isna(v) else f.format(v)
    return out


# ============================================================================
# PANEL CENTRAL: velas + indicadores
# ============================================================================
disp = d
if rango_grafico != "Máximo":
    años_g = {"20 años": 20, "10 años": 10, "5 años": 5, "2 años": 2}[rango_grafico]
    disp = d.loc[d.index[-1] - pd.DateOffset(years=años_g):]

if not sel_ind:
    st.info("Añade indicadores en la barra lateral para mostrarlos bajo las velas.")
show_plotly(
    plot_principal(disp, tdf, sel_ind, float(crit_vfi), float(vix_up_max),
                   float(k_mult), mostrar_marcas, sma_velas, vix_velas, ticker))

# ============================================================================
# COMPARATIVA SISTEMA vs BUY & HOLD
# ============================================================================
st.subheader("🏁 Sistema vs Buy & Hold SPY (mismo rango de fechas)")

fs, fb = fmt(m_sys), fmt(m_bh)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("##### 🟦 Estrategia")
    for k, v in fs.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown(f"**Exposición media:** {exposure:.0f} % del tiempo en el mercado")
with col_b:
    st.markdown("##### ⬜ Buy & Hold SPY")
    for k, v in fb.items():
        st.markdown(f"**{k}:** {v}")
with col_c:
    st.markdown("##### 📊 Ventaja del sistema")
    dif = m_sys["CAR (rent. anual comp.)"] - m_bh["CAR (rent. anual comp.)"]
    st.metric("CAGR sistema − B&H", f"{dif:+.2f} pp")
    dd_s = m_sys["Max. DD (%)"]
    dd_b = m_bh["Max. DD (%)"]
    st.metric("Max DD sistema / B&H", f"{dd_s:.1f} % / {dd_b:.1f} %")
    st.metric("Ulcer Index Sys / B&H",
              f"{m_sys['Ulcer Index']:.1f} / {m_bh['Ulcer Index']:.1f}")

# --- curvas de capital y drawdown ---------------------------------------------------
fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                       vertical_spacing=0.05)
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq, name="Sistema",
                            line=dict(color="#1565c0", width=2)), row=1, col=1)
fig_eq.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh, name="Buy & Hold SPY",
                            line=dict(color="#9e9e9e", width=1.4)), row=1, col=1)
fig_eq.add_trace(go.Scatter(x=eq.index, y=(eq / eq.cummax() - 1) * 100,
                            name="DD Sistema", line=dict(color="#c62828", width=1),
                            fill="tozeroy", fillcolor="rgba(198,40,40,.15)",
                            showlegend=False), row=2, col=1)
fig_eq.add_trace(go.Scatter(x=eq_bh.index, y=(eq_bh / eq_bh.cummax() - 1) * 100,
                            name="DD B&H", line=dict(color="#757575", width=1),
                            showlegend=False), row=2, col=1)
fig_eq.update_layout(height=430, template="plotly_white", hovermode="x unified",
                     legend=dict(orientation="h", y=1.06, x=0),
                     margin=dict(l=10, r=10, t=20, b=10))
fig_eq.update_yaxes(title_text="Equity ($)", row=1, col=1)
fig_eq.update_yaxes(title_text="Drawdown %", row=2, col=1)
show_plotly(fig_eq)

# ============================================================================
# TABLA DE OPERACIONES
# ============================================================================
st.subheader("📋 Operaciones (entradas y salidas)")
trades_show = tdf.copy()
if open_pos:
    trades_show = pd.concat([trades_show, pd.DataFrame([open_pos])],
                            ignore_index=True)
# fechas como texto (homogéneo para st.dataframe / CSV / XLSX)
for col_f in ("Fecha entrada", "Fecha salida"):
    trades_show[col_f] = trades_show[col_f].astype(str)
if trades_show.empty:
    st.info("El sistema no registra ninguna operación con estos parámetros y rango.")
else:
    trades_show.insert(0, "#", range(1, len(trades_show) + 1))
    show_dataframe(trades_show, hide_index=True)

    csv = trades_show.to_csv(index=False).encode("utf-8-sig")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        trades_show.to_excel(xw, index=False, sheet_name="Operaciones")
        df_sig_export = d[["Open", "High", "Low", "Close", "Volume", "VIX", "VIXUP",
                           "VIXDN", "ATR", "ATRUP", "ATRDN", "VFI", "MA_VFI_10",
                           "VENTANA", "VOLCONDITION", "BUY", "SELL_SEASONAL",
                           "PANIC_VIX", "PANIC_VFI"]].round(3)
        df_sig_export.to_excel(xw, sheet_name="Serie diaria")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("⬇️ Descargar CSV", csv, f"operaciones_{ticker}.csv", "text/csv")
    c2.download_button("⬇️ Descargar XLSX", buf.getvalue(),
                       f"operaciones_{ticker}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============================================================================
# ESTADO ACTUAL
# ============================================================================
u = df.iloc[-1]
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VFI (130)", f"{u['VFI']:.1f}")
c2.metric("VIX", f"{u['VIX']:.1f}")
c3.metric("VIXUP %", f"{u['VIXUP']:.1f} %")
c4.metric("ATRUP %", f"{u['ATRUP']:.1f} %")
c5.metric("Señal de entrada hoy", "SÍ ✅" if bool(u["BUY"]) else "NO ❌")
st.caption(
    f"**{df.index[-1].date()}** · Ventana estacional (oct–{MESES[sell_month]}): "
    f"**{'SÍ' if bool(u['VENTANA']) else 'NO'}** · Volatilidad y VFI en condiciones: "
    f"**{'SÍ' if bool(u['VOLCONDITION']) else 'NO'}** · Pánico activo (salida): "
    f"**{'SÍ ⚠️' if bool(u['PANIC_EXIT']) else 'NO'}**")

st.markdown("""
---
**NOTAS** · Precios **sin ajustar por dividendo** por defecto, idéntico al AFL original
(puedes activarlos con el interruptor de la barra lateral para incluir dividendos) ·
Comisión por acción ($0.01 por defecto) en cada lado de la operación ·
La salida estacional se ejecuta a la apertura del **primer día hábil de SELLMONTH**; las
salidas por pánico (VIXUP > 2×VIXUPMAX, o cruce del VFI bajo CRIT con MA(VFI,10) descendente)
se ejecutan a la apertura del **día siguiente** a la señal, tal y como indica el artículo.
""")
