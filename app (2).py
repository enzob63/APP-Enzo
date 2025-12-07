from pwa import inject_pwa
inject_pwa()

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Valuator Pro", layout="wide", page_icon="📈")

# --- GESTION DES DONNÉES (PERSISTANCE JSON) ---
DATA_FILE = "portfolio_data.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

data = load_data()

# --- LOGIQUE MÉTIER (ALGORITHMES) ---

def calculate_g_prudent(g_brut):
    """
    Applique le filtre de sécurité sur la croissance.
    """
    g_brut_percent = g_brut * 100
    
    if g_brut_percent <= 15:
        coeff = 0.90
    elif g_brut_percent > 60:
        coeff = 0.50
    else:
        # Zone de freinage : baisse de 0.025 tous les 2.5% au-dessus de 15%
        steps = (g_brut_percent - 15) / 2.5
        coeff = 0.90 - (int(steps) * 0.025)
        # Sécurité pour ne pas descendre sous 0.50 si le calcul mathématique le faisait
        coeff = max(0.50, coeff)
        
    return g_brut * coeff, coeff

def get_pe_coefficient(pe_base):
    """
    Retourne le coefficient réducteur selon la grille de gravité.
    """
    if pe_base < 15: return 1.0
    if 15 <= pe_base < 25: return 0.90
    if 25 <= pe_base < 30: return 0.85
    if 30 <= pe_base < 35: return 0.825
    if 35 <= pe_base < 40: return 0.80
    if 40 <= pe_base < 45: return 0.775
    if 45 <= pe_base < 50: return 0.75
    if 50 <= pe_base < 55: return 0.725
    if 55 <= pe_base < 60: return 0.70
    if 60 <= pe_base < 65: return 0.65
    if pe_base >= 65: return 0.60
    return 1.0

def run_simulation(vals, horizon, tax_mode):
    """
    Exécute les 5 étapes de l'algorithme.
    """
    results = {}
    logs = []

    # 1. Mécanismes de Sécurité - Croissance
    g_est = vals.get('growth_est', 0) / 100
    g_prudent, g_coeff = calculate_g_prudent(g_est)
    results['g_prudent'] = g_prudent
    logs.append(f"🛡️ **Croissance** : Estimée {g_est:.1%} → Ajustée **{g_prudent:.1%}** (Coeff {g_coeff:.3f})")

    # 2. Mécanismes de Sécurité - PER Cible (Double Verrou)
    pe_10 = vals.get('pe_10y', 15)
    pe_5 = vals.get('pe_5y', 15)
    pe_base = min(pe_10, pe_5)
    pe_coeff = get_pe_coefficient(pe_base)
    pe_target = pe_base * pe_coeff
    results['pe_target'] = pe_target
    logs.append(f"🔒 **PER Cible** : Base {pe_base:.1f} (min 5y/10y) x Coeff {pe_coeff} = **{pe_target:.1f}**")

    # 3. Projection EPS et Prix
    eps_current = vals.get('eps_ttm', 1)
    eps_final = eps_current * ((1 + g_prudent) ** horizon)
    price_final = eps_final * pe_target
    current_price = vals.get('price', 100)
    
    # CAGR Brut (Capital Gain uniquement)
    if current_price > 0:
        cagr_brut_price = (price_final / current_price) ** (1 / horizon) - 1
    else:
        cagr_brut_price = 0
        
    logs.append(f"📈 **Projection Prix** : EPS {eps_current:.2f} → {eps_final:.2f}. Prix {current_price} → **{price_final:.2f}**")

    # 4. Dividendes (Approximation simplifiée : cumulés et réinvestis implicitement ou simple somme pour le rendement total)
    # Note : Pour un calcul de CAGR Total précis, on additionne souvent la valeur finale du prix + dividendes cumulés.
    div_yield = vals.get('div_yield', 0) / 100
    div_growth = vals.get('div_growth', 0) / 100
    
    current_div = current_price * div_yield
    total_dividends = 0
    # Simulation année par année pour les dividendes
    for i in range(1, horizon + 1):
        # Le dividende croit au rythme g_div
        d = current_div * ((1 + div_growth) ** i)
        total_dividends += d
    
    value_final_gross = price_final + total_dividends
    total_gain_gross = value_final_gross - current_price

    # 5. Fiscalité
    tax_coeff = 0.828 if tax_mode == "PEA" else 0.70
    
    # On applique la taxe sur la plus-value totale (PV Prix + Dividendes)
    # Note: Dans la réalité c'est plus complexe (div taxés au fil de l'eau en CTO), 
    # mais ici on suit la logique "Rendement final intègre l'impôt" en fin de course.
    if total_gain_gross > 0:
        net_gain = total_gain_gross * tax_coeff
    else:
        net_gain = total_gain_gross # Pas d'impôt sur les moins-values, on garde la perte brute
        
    value_final_net = current_price + net_gain
    
    if current_price > 0 and value_final_net > 0:
        cagr_net = (value_final_net / current_price) ** (1 / horizon) - 1
    else:
        cagr_net = -1 # Perte totale ou erreur

    results['cagr_net'] = cagr_net
    results['value_final_net'] = value_final_net
    results['logs'] = logs
    
    return results

# --- INTERFACE UTILISATEUR ---

st.title("🛡️ Analyseur de Valeur & Sécurité")

# SIDEBAR : SÉLECTION ENTREPRISE
st.sidebar.header("Portefeuille")
company_names = list(data.keys())
selected_company = st.sidebar.selectbox("Choisir une entreprise", ["Nouvelle Entreprise"] + company_names)

if selected_company == "Nouvelle Entreprise":
    ticker_input = st.sidebar.text_input("Nom / Ticker de l'entreprise")
    if st.sidebar.button("Créer") and ticker_input:
        data[ticker_input] = {} # Init empty
        save_data(data)
        st.rerun()
    active_company_key = None
else:
    active_company_key = selected_company

# MAIN CONTENT
if active_company_key:
    comp_data = data[active_company_key]
    
    st.header(f"Analyse : {active_company_key}")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Données de Marché & Historiques")
        with st.container(border=True):
            new_price = st.number_input("Prix Actuel (€/$)", value=comp_data.get('price', 100.0))
            new_eps = st.number_input("EPS TTM (Bénéfice par action)", value=comp_data.get('eps_ttm', 5.0))
            
            c1, c2 = st.columns(2)
            new_pe_5y = c1.number_input("PER Moyen 5 ans", value=comp_data.get('pe_5y', 20.0))
            new_pe_10y = c2.number_input("PER Médian 10 ans", value=comp_data.get('pe_10y', 18.0))
            
            st.divider()
            st.caption("Estimations Analystes")
            new_growth_est = st.number_input("Croissance EPS estimée (CAGR %)", value=comp_data.get('growth_est', 10.0), help="g brut")
            
            c3, c4 = st.columns(2)
            new_div_yield = c3.number_input("Rendement Div Actuel (%)", value=comp_data.get('div_yield', 2.0))
            new_div_growth = c4.number_input("Croissance Div estimée (%)", value=comp_data.get('div_growth', 5.0))

    with col2:
        st.subheader("2. Scoring Achat (Indicateurs)")
        with st.container(border=True):
            # Partie A du prompt
            sc_per_ttm = st.number_input("PER TTM", value=comp_data.get('sc_per_ttm', 0.0))
            sc_per_avg4 = st.number_input("PER Moyen / 4 ans", value=comp_data.get('sc_per_avg4', 0.0))
            st.divider()
            sc_peg_ttm = st.number_input("PEG TTM", value=comp_data.get('sc_peg_ttm', 0.0))
            sc_peg_avg4 = st.number_input("PEG Moyen / 4 ans", value=comp_data.get('sc_peg_avg4', 0.0))
            st.divider()
            sc_fair_value = st.number_input("Fair Value (GF/Finbox)", value=comp_data.get('sc_fair_value', 0.0))
            
            # Calcul automatique de la marge de sécu sur fair value
            if sc_fair_value > 0 and new_price > 0:
                upside = (sc_fair_value - new_price) / new_price
                color = "green" if upside > 0 else "red"
                st.markdown(f"Marge de sécurité (Fair Value) : :{color}[**{upside:.1%}**]")

    # Sauvegarde automatique des entrées
    if st.button("💾 Sauvegarder les données"):
        data[active_company_key] = {
            'price': new_price, 'eps_ttm': new_eps,
            'pe_5y': new_pe_5y, 'pe_10y': new_pe_10y,
            'growth_est': new_growth_est,
            'div_yield': new_div_yield, 'div_growth': new_div_growth,
            'sc_per_ttm': sc_per_ttm, 'sc_per_avg4': sc_per_avg4,
            'sc_peg_ttm': sc_peg_ttm, 'sc_peg_avg4': sc_peg_avg4,
            'sc_fair_value': sc_fair_value
        }
        save_data(data)
        st.success("Données mises à jour !")

    st.markdown("---")
    
    # SECTION SIMULATION
    st.header("3. Simulateur de Rendement (Algorithme Sécurisé)")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.markdown("#### Paramètres")
        horizon = st.slider("Horizon (années)", 3, 10, 5)
        tax_mode = st.radio("Enveloppe Fiscale", ["PEA (17.2%)", "CTO (30%)"])
        tax_code = "PEA" if "PEA" in tax_mode else "CTO"
        
        if st.button("🚀 Lancer la Simulation", type="primary"):
            # Prepare data dict for sim
            sim_inputs = {
                'growth_est': new_growth_est,
                'pe_10y': new_pe_10y, 'pe_5y': new_pe_5y,
                'eps_ttm': new_eps, 'price': new_price,
                'div_yield': new_div_yield, 'div_growth': new_div_growth
            }
            
            res = run_simulation(sim_inputs, horizon, tax_code)
            
            with sim_col2:
                st.markdown("#### Résultats")
                
                # Affichage du CAGR NET FINAL
                cagr = res['cagr_net']
                color_metric = "normal"
                if cagr > 0.15: color_metric = "off" # Streamlit trick for green usually, or use custom HTML
                
                st.metric(label="CAGR TOTAL NET (après impôts & sécurité)", value=f"{cagr:.2%}", delta=f"Horizon {horizon} ans")
                
                if cagr >= 0.15:
                    st.success("🎯 Objectif > 15% atteint ! Opportunité potentielle.")
                elif cagr >= 0.10:
                    st.warning("⚠️ Rendement correct mais sous 15%.")
                else:
                    st.error("❌ Rendement insuffisant selon les critères de sécurité.")

                with st.expander("🔍 Voir le détail du calcul (Les 5 Étapes)"):
                    for log in res['logs']:
                        st.markdown(f"- {log}")
                    st.markdown(f"- 💰 **Dividendes & Fiscalité** : Rendement net calculé via {tax_code}.")
                    st.markdown(f"- **Valeur Finale Nette** estimée : {res['value_final_net']:.2f}")

else:
    st.info("👈 Sélectionnez une entreprise dans le menu ou créez-en une nouvelle pour commencer.")

    # Vue tableau global
    if data:
        st.markdown("### Vue d'ensemble")
        df = pd.DataFrame.from_dict(data, orient='index')
        # On sélectionne quelques colonnes clés pour l'affichage tableau
        cols_to_show = ['price', 'sc_fair_value', 'growth_est', 'pe_5y']
        existing_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[existing_cols])