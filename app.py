# app.py (remplace intégralement ton fichier existant)
import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_echarts import st_echarts
import yfinance as yf
from datetime import datetime
import requests

# ===========================
# CONFIG
# ===========================
st.set_page_config(page_title="AppEnzo V33", layout="wide", page_icon="🐰")

DATA_FILE = "appenzo_v33.json"

# ===========================
# CSS / STYLE
# ===========================
SVG_LOGO = """
<svg width="64" height="64" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="gold" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#FCD34D"/><stop offset="100%" style="stop-color:#B45309"/></linearGradient></defs>
  <circle cx="50" cy="50" r="45" fill="#FFFFFF" stroke="url(#gold)" stroke-width="3"/>
  <path d="M35 55 Q30 20 40 15 Q50 25 45 55" fill="#E2E8F0" stroke="#334155" stroke-width="1.5"/>
  <path d="M65 55 Q70 20 60 15 Q50 25 55 55" fill="#E2E8F0" stroke="#334155" stroke-width="1.5"/>
  <circle cx="50" cy="65" r="22" fill="#F8FAFC" stroke="#334155" stroke-width="1.5"/>
  <path d="M35 40 L35 25 L42 35 L50 20 L58 35 L65 25 L65 40 Z" fill="url(#gold)" stroke="#B45309" stroke-width="1"/>
  <circle cx="43" cy="62" r="2" fill="#334155"/>
  <circle cx="57" cy="62" r="2" fill="#334155"/>
  <path d="M46 70 Q50 75 54 70" fill="none" stroke="#334155" stroke-width="1.5"/>
</svg>
"""

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Libre+Baskerville:wght@700&display=swap');

  .stApp {{ background-color: #FFFFFF; font-family: 'Inter', sans-serif; color: #0F172A; }}

  /* HEADER */
  .main-header {{
    display:flex; align-items:center; gap:16px;
    padding:14px 20px; margin-bottom:18px; border-radius:0 0 14px 14px;
    background:#F8FAFC; border-bottom:3px solid #D4AF37;
    box-shadow:0 4px 10px rgba(0,0,0,0.04);
  }}
  .header-title {{
    font-family: 'Libre Baskerville', serif; font-size:34px; font-weight:800;
    color: #0F172A; margin:0;
    text-transform:uppercase; letter-spacing:1px;
    text-shadow: 0 1px 0 rgba(255,255,255,0.6);
  }}

  /* Top tabs area spacing */
  .top-tabs-space {{ margin-top: 8px; margin-left:6px; }}

  /* Inputs */
  .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {{
    background-color: #F8FAFC !important; border: 1px solid #CBD5E1 !important;
    color: #0F172A !important; border-radius:6px;
  }}

  /* Blue box for user analysis */
  .blue-zone {{
    background-color: #EFF6FF; border: 2px solid #3B82F6; border-radius:12px; padding:16px;
  }}
  .blue-zone h5 {{ color:#1E40AF; font-weight:800; margin:0 0 8px 0; }}

  /* Yield big display */
  .yield-display {{
    font-size:88px; font-weight:900; text-align:center; margin:6px 0; line-height:0.9;
    font-family: 'Arial', sans-serif;
  }}
  .col-green {{ color: #10B981; }}
  .col-orange {{ color: #F59E0B; }}
  .col-red {{ color: #EF4444; }}

  /* Classement */
  .rank-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 16px; border-radius:10px; margin-bottom:10px; background:#fff; box-shadow:0 4px 10px rgba(2,6,23,0.03);
    border-left:6px solid transparent;
  }}
  .rank-left {{ display:flex; align-items:center; gap:12px; }}
  .rank-name {{ font-weight:700; font-size:16px; color:#0F172A; }}
  .score-badge {{
    min-width:64px; text-align:center; padding:8px 10px; border-radius:10px; font-weight:800; color:white;
    box-shadow: 0 3px 8px rgba(2,6,23,0.08);
  }}
  .badge-green {{ background:#10B981; }}
  .badge-greenlight {{ background:#84CC16; }}
  .badge-orange {{ background:#F59E0B; }}
  .badge-red {{ background:#EF4444; }}
</style>
""", unsafe_allow_html=True)

# ===========================
# DB persistence
# ===========================
def load_db():
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_db(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if 'db' not in st.session_state:
    st.session_state.db = load_db()

# ===========================
# UTILITAIRES
# ===========================
def get_logo_html(domain, size=40):
    if not domain: return ""
    return f'<img src="https://www.google.com/s2/favicons?domain={domain}&sz=128" style="width:{size}px; height:{size}px; border-radius:8px;">'

def format_euro(n):
    try:
        n = float(n)
    except:
        return f"{n}"
    if n>=1e9: return f"{n/1e9:.2f} Md€"
    if n>=1e6: return f"{n/1e6:.2f} M€"
    return f"{n:,.0f} €".replace(",", " ")

# ===========================
# FETCH / YFINANCE (robuste)
# ===========================
def get_yahoo_url(ticker):
    return f"https://finance.yahoo.com/quote/{ticker}"

def fetch_data(ticker):
    tk = ticker.strip().upper()
    if not tk:
        return None
    try:
        stock = yf.Ticker(tk)
        # try info
        info = {}
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        # try fast_info fallback
        if not info:
            try:
                fi = getattr(stock, "fast_info", None)
                if fi and isinstance(fi, dict):
                    info = {"shortName": tk, "website": "", "sector":"Indéfini","industry":"Indéfini",
                            "currentPrice": fi.get("last_price"), "trailingEps": None, "trailingPE": None,
                            "dividendYield": None, "revenueGrowth": None, "earningsGrowth": None,
                            "operatingMargins": None, "profitMargins": None, "returnOnEquity": None,
                            "debtToEquity": None, "forwardPE": None}
            except Exception:
                info = {}
        # try history to confirm existence
        hist = None
        try:
            hist = stock.history(period="1mo")
        except Exception:
            hist = None
        if (not info or info == {}) and (hist is None or hist.empty):
            return None
        # price fallback
        price = None
        try:
            price = info.get('currentPrice') or info.get('regularMarketPrice')
        except:
            price = None
        if not price:
            try:
                if hist is not None and not hist.empty:
                    price = float(hist['Close'].iloc[-1])
            except:
                price = 0.0
        data = {
            'symbol': tk,
            'name': info.get('shortName') or tk,
            'domain': (info.get('website') or '').replace('https://www.', '').split('/')[0] if info.get('website') else '',
            'sector': info.get('sector', 'Indéfini'),
            'industry': info.get('industry', 'Indéfini'),
            'price': float(price or 0.0),
            'eps_ttm': info.get('trailingEps') or 0.0,
            'pe_now': info.get('trailingPE') or 0.0,
            'div_yield': (info.get('dividendYield', 0) or 0)*100,
            'rev_cagr_hist': (info.get('revenueGrowth', 0) or 0)*100,
            'rev_cagr_fut': (info.get('earningsGrowth', 0) or 0)*100,
            'eps_cagr_hist': 0.0,
            'eps_cagr_fut': (info.get('earningsGrowth', 0) or 0)*100,
            'op_margin': (info.get('operatingMargins', 0) or 0)*100,
            'net_margin': (info.get('profitMargins', 0) or 0)*100,
            'roic': (info.get('returnOnEquity', 0) or 0)*100,
            'net_debt_ebitda': (info.get('debtToEquity', 0) or 0)/100 if info.get('debtToEquity') is not None else 0,
            # user defaults
            'cash_conv': 20.0, 'moat_score': 5.0,
            'pe_target': info.get('forwardPE') or 20.0,
            'shares_owned': 0.0, 'sc_fair_value': 0.0
        }
        for k,v in data.items():
            if v is None: data[k] = 0.0
        return data
    except Exception as e:
        return None

# ===========================
# SCORE & CALCULS
# ===========================
def get_score(d):
    p = 0
    for k in ['rev_cagr_hist','rev_cagr_fut','eps_cagr_hist','eps_cagr_fut']:
        v = d.get(k,0)
        if v > 13: p += 2
        elif 10 <= v <= 12.99: p += 1.5
        elif 8 <= v < 10: p += 1
    if d.get('op_margin',0) >= 25: p += 1
    if d.get('net_margin',0) >= 20: p += 1
    if d.get('roic',0) >= 20 and d.get('net_debt_ebitda',0) <= 1 and d.get('cash_conv',0) >= 20: p += 1
    p += d.get('moat_score',0)
    return min(p, 20)

def calc_cagr_net(d, sim_g=None, sim_pe=None, tax_rate=0.30):
    p0 = d.get('price', 100); eps0 = d.get('eps_ttm', 1)
    if p0 <= 0: return 0
    pe_now = d.get('pe_now', 20); pe_tgt = d.get('pe_target', 20)
    g = (d.get('eps_cagr_fut', 10)/100) if sim_g is None else sim_g/100
    # project 5 ans
    eps5 = eps0 * ((1+g)**5)
    pe5 = pe_tgt
    price5 = eps5 * pe5
    divs = 0
    cd = p0 * (d.get('div_yield',0)/100)
    for _ in range(5):
        cd *= (1+g)
        divs += cd
    gain = (price5 + divs) - p0
    net = gain * (1 - tax_rate)
    total = p0 + net
    cagr = ((total/p0)**(1/5)-1)
    return cagr * 100

# ===========================
# HEADER + TOP TABS
# ===========================
st.markdown(f"""
<div class="main-header">
    {SVG_LOGO}
    <div>
      <h1 class="header-title">APP ENZO</h1>
      <div style="font-size:12px;color:#64748B;margin-top:4px;">Gestion & analyse rapide</div>
    </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["PORTEFEUILLE", "ANALYSE GRAPHIQUE", "CLASSEMENT", "CONSEILS"])

# small sidebar info (kept)
with st.sidebar:
    st.markdown("### 🧭 MENU")
    if st.session_state.db:
        tot = sum([d['price']*d.get('shares_owned',0) for d in st.session_state.db.values()])
        st.metric("Capital Total", format_euro(tot))
    st.markdown("---")
    st.markdown("Made with Streamlit — App Enzo")

# ===========================
# PAGE: PORTEFEUILLE
# ===========================
with tabs[0]:
    if st.session_state.db:
        df = pd.DataFrame(st.session_state.db.values())
        df['total'] = df['price'] * df.get('shares_owned',0)
        if df['total'].sum() > 0:
            c_v, c_c = st.columns([1, 4])
            with c_v:
                st.markdown("**Vue :**")
                view = st.radio("", ["Positions", "Secteur", "Industrie"], label_visibility="collapsed")
                col_map = {"Positions":"name","Secteur":"sector","Industrie":"industry"}
            with c_c:
                pie_df = df.groupby(col_map[view])['total'].sum().reset_index()
                pdata = [{"value": float(r['total']), "name": r[col_map[view]]} for i, r in pie_df.iterrows()]
                opt = {
                    "tooltip": {"trigger": "item", "formatter": "{b}: {c}€ ({d}%)"},
                    "legend": {"bottom": 0},
                    "series": [{"type":"pie", "radius":[30,110], "center":["50%","50%"], "roseType":"area", "itemStyle":{"borderRadius":5}, "avoidLabelOverlap":True, "data":pdata}]
                }
                st_echarts(options=opt, height="300px")

    st.markdown("---")

    # FORM ADD ACTION
    with st.expander("➕ AJOUTER ACTION (Yahoo)", expanded=False):
        with st.form("add"):
            c1, c2 = st.columns([3,1])
            tk = c1.text_input("Ticker", placeholder="NVDA")
            if c2.form_submit_button("CHERCHER"):
                if not tk.strip():
                    st.error("Saisis un ticker.")
                else:
                    nd = fetch_data(tk.upper())
                    yahoo_url = get_yahoo_url(tk.upper())
                    url_ok = False
                    try:
                        r = requests.get(yahoo_url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
                        url_ok = (r.status_code == 200)
                    except:
                        url_ok = False

                    if nd:
                        key = nd.get('symbol', tk.upper())
                        st.session_state.db[key] = nd
                        save_db(st.session_state.db)
                        st.success(f"{key} ajouté·e.")
                        st.markdown(f"[Voir sur Yahoo Finance]({yahoo_url})")
                        st.experimental_rerun()
                    else:
                        if url_ok:
                            st.error(f"Ticker trouvé sur Yahoo (page ok) mais yfinance ne renvoie pas d'infos complets.")
                            st.markdown(f"[Ouvrir la page Yahoo]({yahoo_url})")
                        else:
                            st.error(f"Ticker {tk.upper()} introuvable. Vérifie le symbole.")
                            st.markdown(f"[Tester manuellement sur Yahoo]({yahoo_url})")

    st.markdown("---")
    if st.session_state.db:
        sel = st.selectbox("Modifier une ligne :", ["-- Sélectionner --"] + list(st.session_state.db.keys()))
        if sel != "-- Sélectionner --":
            d = st.session_state.db[sel]
            c_logo, c_nom = st.columns([1, 10])
            with c_logo: st.markdown(get_logo_html(d.get('domain'),60), unsafe_allow_html=True)
            with c_nom:
                new_n = st.text_input("Nom", sel, label_visibility="collapsed")
                if new_n != sel:
                    st.session_state.db[new_n] = st.session_state.db.pop(sel)
                    save_db(st.session_state.db)
                    st.experimental_rerun()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### 🌍 MARCHÉ (Yahoo)")
                d['price'] = st.number_input("Prix", value=float(d.get('price',0)), step=0.5)
                d['pe_now'] = st.number_input("PER Actuel", value=float(d.get('pe_now',0)), step=0.5)
                d['eps_cagr_fut'] = st.number_input("Croiss. BPA Est. %", value=float(d.get('eps_cagr_fut',10)), step=0.5)
                d['shares_owned'] = st.number_input("Qté Détenue", value=float(d.get('shares_owned',0)), step=1.0)
            with c2:
                st.markdown("##### 📈 FONDAMENTAUX")
                d['op_margin'] = st.number_input("Marge Op %", value=float(d.get('op_margin',0)), step=0.5)
                d['net_margin'] = st.number_input("Marge Nette %", value=float(d.get('net_margin',0)), step=0.5)
                d['roic'] = st.number_input("ROIC %", value=float(d.get('roic',0)), step=0.5)
                d['net_debt_ebitda'] = st.number_input("Dette/EBITDA", value=float(d.get('net_debt_ebitda',0)), step=0.1)
                st.info("Autres")
                d['rev_cagr_hist'] = st.number_input("Crois. CA Hist. %", value=float(d.get('rev_cagr_hist',0)), step=0.5)
            with c3:
                st.markdown('<div class="blue-zone">', unsafe_allow_html=True)
                st.markdown("<h5>💎 VOTRE ANALYSE</h5>", unsafe_allow_html=True)
                d['moat_score'] = st.number_input("Score Moat (0-9)", value=float(d.get('moat_score',5)), min_value=0.0, max_value=9.0, step=1.0)
                d['pe_target'] = st.number_input("PER Cible (Sortie)", value=float(d.get('pe_target',20)), step=1.0)
                d['sc_fair_value'] = st.number_input("Fair Value (Juste Valeur)", value=float(d.get('sc_fair_value',0)), step=1.0)
                st.markdown('</div>', unsafe_allow_html=True)
            if st.button("💾 ENREGISTRER"):
                st.session_state.db[sel] = d
                save_db(st.session_state.db)
                st.toast("Sauvegardé")
            if st.button("🗑️"):
                del st.session_state.db[sel]; save_db(st.session_state.db); st.experimental_rerun()

# ===========================
# PAGE: ANALYSE GRAPHIQUE
# ===========================
with tabs[1]:
    if not st.session_state.db:
        st.warning("Ajoute des actions d'abord.")
    else:
        c_sel, c_tax = st.columns([2,1])
        target = c_sel.selectbox("Action :", list(st.session_state.db.keys()))
        tax = c_tax.selectbox("Fiscalité :", ["Compte-Titres (30%)", "PEA (17.2%)", "Brut (0%)"])
        dat = st.session_state.db[target]

        # sliders init
        if 'last_t' not in st.session_state or st.session_state.last_t != target:
            st.session_state.sim_g = float(dat.get('eps_cagr_fut', 10.0))
            st.session_state.sim_pe = float(dat.get('pe_target', 20.0))
            st.session_state.last_t = target

        cc1, cc2 = st.columns(2)
        with cc1: st.session_state.sim_g = st.slider("Croissance Future (%)", 0.0, 50.0, value=st.session_state.sim_g, step=0.5, key="sg")
        with cc2: st.session_state.sim_pe = st.slider("PER Cible", 5.0, 100.0, value=st.session_state.sim_pe, step=0.5, key="spe")

        # Fair value gauge
        fv = dat.get('sc_fair_value', 0); curr = dat.get('price', 0)
        if fv > 0 and curr > 0:
            diff = (fv - curr) / curr
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = curr,
                domain = {'x':[0,1], 'y':[0,1]},
                title = {'text': f"Prix ({curr}€) vs Fair Value ({fv}€)", 'font': {'size':14, 'family':'Inter'}},
                delta = {'reference': fv, 'relative': True, 'valueformat': '.1%', 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                gauge = {'axis':{'range':[0,max(fv,curr)*1.4]}, 'bar':{'color': "#0F172A"}, 'threshold':{'line':{'color':"blue",'width':3}, 'thickness':0.8, 'value':fv}}
            ))
            fig_g.update_layout(height=180, margin=dict(t=30,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        # projections
        p0 = dat.get('price', 100); eps0 = dat.get('eps_ttm', 5); hist_g = dat.get('eps_cagr_hist', 10.0)/100
        y_h = [2020,2021,2022,2023,2024]; y_f = [2025,2026,2027,2028,2029,2030]
        ph, eh = [], []; cp, ce = p0, eps0
        for _ in range(5):
            cp /= (1+hist_g); ce /= (1+hist_g); ph.append(cp); eh.append(ce)
        ph.reverse(); eh.reverse()
        pf, ef = [p0], [eps0]; pe0 = dat.get('pe_now', 20)
        for i in range(1,6):
            ne = eps0 * ((1 + st.session_state.sim_g/100) ** i)
            npe = pe0 + (st.session_state.sim_pe - pe0) * (i/5)
            pf.append(ne * npe); ef.append(ne)
        divs = 0; cd = p0 * (dat.get('div_yield',0)/100)
        for i in range(5):
            cd *= (1 + st.session_state.sim_g/100)
            divs += cd

        brut = (pf[-1] + divs) - p0
        tx = 0.30 if "30%" in tax else (0.172 if "17.2%" in tax else 0.0)
        net = brut * (1 - tx)
        final = p0 + net
        cagr = ((final/p0)**(1/5) - 1) if p0>0 else 0

        # color thresholds requested
        cls = "col-red"
        if cagr > 0.12:
            cls = "col-green"
        elif 0.10 <= cagr <= 0.1199:
            cls = "col-orange"
        else:
            cls = "col-red"

        st.markdown(f"<div style='text-align:center; font-weight:700; margin-top:10px; color:#64748B;'>RENDEMENT NET ANNUEL ESTIMÉ</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='yield-display {cls}'>{cagr:.1%} / an</div>", unsafe_allow_html=True)

        # Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=y_h+[2025], y=ph+[p0], name="Prix Hist", line=dict(color='#9CA3AF', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=y_f, y=pf, name="Prix Proj", line=dict(color='#10B981', width=4), fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=y_h+y_f, y=eh+ef, name="EPS", line=dict(color='#2563EB', width=2, dash='dot')), secondary_y=True)
        fig.add_vline(x=2025, line_width=2, line_dash="solid", line_color="#333")
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', height=520, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# PAGE: CLASSEMENT
# ===========================
with tabs[2]:
    st.markdown("## Classement des actions")
    if not st.session_state.db:
        st.info("Aucune donnée.")
    else:
        ranked = []
        for n,d in st.session_state.db.items():
            ranked.append((n, get_score(d), d))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for n,s,d in ranked:
            # choose badge color
            if s >= 16:
                badge_class = "badge-green"
            elif 13 <= s <= 15:
                badge_class = "badge-greenlight"
            elif 10 <= s <= 12:
                badge_class = "badge-orange"
            else:
                badge_class = "badge-red"
            left_html = f'{get_logo_html(d.get("domain"),40)}<div class="rank-name">{n}</div>'
            st.markdown(f"""
            <div class="rank-row" style="border-left:6px solid {('#10B981' if s>=16 else ('#84CC16' if s>=13 else ('#F59E0B' if s>=10 else '#EF4444')))};">
                <div class="rank-left">
                    {get_logo_html(d.get('domain'),40)}
                    <div>
                        <div style="font-size:13px; color:#64748B;">{d.get('sector','')}</div>
                        <div class="rank-name">{n}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="font-size:12px; color:#64748B; text-align:right;">
                        PER {d.get('pe_now',0):.1f}x<br>
                        Croiss. {d.get('eps_cagr_fut',0):.1f}%
                    </div>
                    <div class="score-badge {badge_class}">{s:.0f}/20</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ===========================
# PAGE: CONSEILS
# ===========================
with tabs[3]:
    st.markdown("### 🧭 STRATÉGIE AUTOMATIQUE")
    if not st.session_state.db:
        st.info("Ajoute des actions pour obtenir des conseils.")
    else:
        buy, hold, sell = [], [], []
        for n,d in st.session_state.db.items():
            s = get_score(d)
            c = calc_cagr_net(d)
            if s >= 15 or c >= 15:
                buy.append((n,s,c))
            elif s >= 12 or c >= 10:
                hold.append((n,s,c))
            else:
                sell.append((n,s,c))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div style='background:#DCFCE7; padding:10px; border-radius:8px; color:#166534; font-weight:bold; text-align:center; margin-bottom:10px;'>💎 ACHAT FORT ({len(buy)})</div>", unsafe_allow_html=True)
            for n,s,c in buy:
                st.markdown(f"<div style='background:white; border-left:5px solid #10B981; padding:10px; border-radius:8px; margin-bottom:8px;'><b>{n}</b><br><span style='font-size:12px'>Note: {s}/20 • TRI: {c:.1f}%</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='background:#FEF9C3; padding:10px; border-radius:8px; color:#854D0E; font-weight:bold; text-align:center; margin-bottom:10px;'>🛡️ CONSERVER ({len(hold)})</div>", unsafe_allow_html=True)
            for n,s,c in hold:
                st.markdown(f"<div style='background:white; border-left:5px solid #F59E0B; padding:10px; border-radius:8px; margin-bottom:8px;'><b>{n}</b><br><span style='font-size:12px'>Note: {s}/20 • TRI: {c:.1f}%</span></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='background:#FEE2E2; padding:10px; border-radius:8px; color:#991B1B; font-weight:bold; text-align:center; margin-bottom:10px;'>⚠️ VENDRE ({len(sell)})</div>", unsafe_allow_html=True)
            for n,s,c in sell:
                st.markdown(f"<div style='background:white; border-left:5px solid #EF4444; padding:10px; border-radius:8px; margin-bottom:8px;'><b>{n}</b><br><span style='font-size:12px'>Note: {s}/20 • TRI: {c:.1f}%</span></div>", unsafe_allow_html=True)
