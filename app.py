"""
Brazil Address Risk Bot
=======================
Analyzes Brazilian addresses for:
  - Socioeconomic Status (IBGE + neighborhood heuristics)
  - Logistics Risk (Área de Risco detection)
  - Fraud Risk (composite score)
  - Approval Recommendation

Data Sources:
  - ViaCEP (https://viacep.com.br)        — free, no key needed
  - IBGE API (https://servicodados.ibge.gov.br) — free, no key needed
  - Nominatim / OpenStreetMap             — free, no key needed
"""

import re
import time
import json
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List


def normalize(text: str) -> str:
    """Remove accents and lowercase — so 'Alemão' == 'alemao'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()

import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🇧🇷 Brazil Address Risk Bot",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# RISK DATA
# ============================================================

# All keyword lists are pre-normalized (no accents, lowercase)
# so matching works regardless of how the user types the address.
HIGH_RISK_KEYWORDS: List[str] = [normalize(k) for k in [
    "favela", "comunidade", "complexo do", "morro do", "morro da", "morro de",
    "area de risco", "loteamento irregular", "invasao",
    # RJ
    "rocinha", "alemao", "manguinhos", "mare", "jacarezinho", "jacare",
    "acari", "mandela", "cidade de deus", "cidade alta", "vigario geral",
    "parada de lucas", "serrinha", "dendê", "grotao",
    "caramujo", "fallet", "fogueteiro", "santa marta favela", "tabajara",
    "cantagalo favela", "pavaozinho", "vila kennedy", "vila alianca",
    "nova brasilia rj", "parque uniao rj", "nova holanda",
    "ramos favela", "barreira do vasco", "chatuba",
    # SP
    "heliopolis", "paraisopolis", "jardim angela", "capao redondo",
    "mboi mirim", "brasilandia", "cidade tiradentes", "jardim campo limpo",
]]

RESTRICTED_DELIVERY_BAIRROS: List[str] = [normalize(k) for k in [
    # RJ
    "pavuna", "acari", "anchieta", "bangu", "realengo", "paciencia",
    "santa cruz rj", "cosmos rj", "senador vasconcelos", "inhoaiba",
    "pedra de guaratiba", "sepetiba", "madureira", "oswaldo cruz rj",
    "quintino bocaiuva", "bento ribeiro", "campinho rj", "cascadura",
    "honorio gurgel", "iraja", "piedade rj", "pilares rj",
    "vicente de carvalho rj", "vila da penha", "turiacu", "vaz lobo",
    "costa barros", "cordovil rj", "penha rj",
    # SP
    "jardim angela", "capao redondo", "cidade tiradentes", "guaianazes",
    "lajeado sp", "sapopemba", "jardim helena sp", "itaim paulista",
    "parelheiros", "grajau sp",
]]

PRESTIGIOUS_KEYWORDS: List[str] = [normalize(k) for k in [
    # RJ
    "ipanema", "leblon", "gavea", "lagoa rodrigo de freitas",
    "barra da tijuca", "botafogo", "urca", "sao conrado",
    "joatinga", "itanhanga", "recreio dos bandeirantes",
    # SP
    "jardins sp", "jardim paulista", "jardim europa",
    "higienopolis", "itaim bibi", "vila nova conceicao",
    "moema", "brooklin", "alphaville sp",
    # BH
    "savassi", "lourdes bh", "belvedere bh", "mangabeiras bh",
    # Curitiba
    "batel", "agua verde curitiba",
]]

# State baseline logistics/fraud risk
STATE_RISK: Dict[str, str] = {
    "RJ": "high", "ES": "high",
    "BA": "medium", "PE": "medium", "CE": "medium",
    "PA": "medium", "AM": "medium", "MA": "medium",
    "AL": "medium", "SE": "medium",
    "SP": "medium_low", "MG": "medium_low",
    "GO": "low", "DF": "low", "RS": "low",
    "SC": "low", "PR": "low", "MT": "low", "MS": "low",
}

SOCIOECONOMIC_META: Dict[str, Dict] = {
    "A":   {"color": "#27ae60", "label": "Class A",     "desc": "High Income"},
    "B1":  {"color": "#2ecc71", "label": "Class B1",    "desc": "Upper-Middle Income"},
    "B2":  {"color": "#f1c40f", "label": "Class B2",    "desc": "Middle Income"},
    "C1":  {"color": "#e67e22", "label": "Class C1/C2", "desc": "Lower-Middle Income"},
    "C2":  {"color": "#e67e22", "label": "Class C1/C2", "desc": "Lower-Middle Income"},
    "D/E": {"color": "#e74c3c", "label": "Class D/E",   "desc": "Low Income"},
}

RISK_COLORS: Dict[str, str] = {
    "Low":         "#27ae60",
    "Medium":      "#f1c40f",
    "Medium-High": "#e67e22",
    "High":        "#e74c3c",
}

# ============================================================
# API LAYER
# ============================================================

def extract_cep(text: str) -> Optional[str]:
    """Extract an 8-digit CEP from arbitrary text."""
    match = re.search(r"\b(\d{5})-?(\d{3})\b", text)
    return match.group(1) + match.group(2) if match else None


def lookup_cep(cep: str) -> Optional[Dict]:
    """
    Fetch address data from ViaCEP.
    Returns a dict with keys: cep, logradouro, bairro, localidade, uf, ibge
    """
    cep_clean = re.sub(r"\D", "", cep)
    if len(cep_clean) != 8:
        return None
    try:
        r = requests.get(
            f"https://viacep.com.br/ws/{cep_clean}/json/",
            timeout=10,
        )
        data = r.json()
        return None if "erro" in data else data
    except Exception:
        return None


def geocode_address(address_str: str) -> Optional[Tuple[float, float]]:
    """
    Geocode a Brazilian address using Nominatim (OpenStreetMap).
    Returns (lat, lon) or None.
    """
    if "brazil" not in address_str.lower() and "brasil" not in address_str.lower():
        address_str += ", Brazil"

    headers = {"User-Agent": "BrazilAddressRiskBot/1.0 (riskified.com)"}
    params = {
        "q": address_str,
        "format": "json",
        "limit": 1,
        "countrycodes": "br",
    }
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=15,
        )
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        return None
    except Exception:
        return None


def get_municipal_pib(ibge_code: str) -> Optional[float]:
    """
    Fetch PIB per capita (R$) for a municipality via IBGE Sidra API.
    Table 5938, variable 6575 (PIB per capita), period 2021.
    """
    if not ibge_code or len(ibge_code) < 6:
        return None
    try:
        url = (
            "https://servicodados.ibge.gov.br/api/v3/agregados"
            "/5938/periodos/2021/variaveis/6575"
        )
        r = requests.get(
            url,
            params={"localidades": f"N6[{ibge_code}]"},
            timeout=12,
        )
        data = r.json()
        value_str = (
            data[0]["resultados"][0]["series"][0]["serie"].get("2021")
        )
        if value_str and value_str not in ("-", "..."):
            return float(value_str)
        return None
    except Exception:
        return None


# ============================================================
# RISK ENGINE
# ============================================================

def classify_socioeconomic(
    pib_per_capita: Optional[float],
    bairro: str,
    cidade: str,
    uf: str,
) -> Tuple[str, str]:
    """
    Returns (class_label, explanation_string).
    Labels: A | B1 | B2 | C1 | C2 | D/E
    """
    text = normalize(f"{bairro} {cidade}")

    # Prestigious neighborhood → bump up
    for kw in PRESTIGIOUS_KEYWORDS:
        if kw in text:
            cls = "A" if (pib_per_capita and pib_per_capita >= 40000) else "B1"
            return cls, f"Prestigious neighbourhood ({bairro}) — upper-income area"

    # Informal settlement → pull down
    for kw in HIGH_RISK_KEYWORDS[:25]:
        if kw in text:
            return "D/E", f"Informal settlement indicator in address ({kw})"

    # PIB per capita thresholds (IBGE, in R$)
    if pib_per_capita:
        if pib_per_capita >= 65000:
            return "A",   f"High PIB per capita: R${pib_per_capita:,.0f}/yr"
        if pib_per_capita >= 45000:
            return "B1",  f"Upper-middle PIB per capita: R${pib_per_capita:,.0f}/yr"
        if pib_per_capita >= 28000:
            return "B2",  f"Middle PIB per capita: R${pib_per_capita:,.0f}/yr"
        if pib_per_capita >= 18000:
            return "C1",  f"Lower-middle PIB per capita: R${pib_per_capita:,.0f}/yr"
        if pib_per_capita >= 12000:
            return "C2",  f"Lower PIB per capita: R${pib_per_capita:,.0f}/yr"
        return "D/E", f"Low PIB per capita: R${pib_per_capita:,.0f}/yr"

    # City fallback (all keys pre-normalized)
    cidade_map = {
        "sao paulo": "B2", "rio de janeiro": "C1",
        "brasilia": "B2", "curitiba": "B2", "porto alegre": "B2",
        "florianopolis": "B2", "belo horizonte": "C1",
        "vitoria": "B2", "goiania": "C1", "campinas": "B2",
        "manaus": "C2", "belem": "C2", "recife": "C2",
        "salvador": "C2", "fortaleza": "C2", "natal": "C2",
        "joao pessoa": "C2", "maceio": "C2",
        "teresina": "D/E", "sao luis": "D/E",
    }
    for city, cls in cidade_map.items():
        if city in text:
            return cls, f"Estimated from city baseline ({cidade})"

    # State fallback
    state_map = {
        "SP": "B2", "RJ": "C1", "DF": "B1", "SC": "B2", "RS": "C1",
        "PR": "B2", "MG": "C1", "GO": "C1", "ES": "C1", "MT": "C1",
        "MS": "C1", "RO": "C2", "AM": "C2", "PA": "D/E", "MA": "D/E",
        "CE": "C2", "PE": "C2", "BA": "C2", "PB": "C2", "RN": "C2",
        "AL": "D/E", "SE": "C2", "PI": "D/E", "AC": "D/E",
        "AP": "D/E", "RR": "D/E", "TO": "D/E",
    }
    cls = state_map.get(uf.upper() if uf else "", "C2")
    return cls, f"Estimated from state baseline ({uf})"


def assess_logistics_risk(
    bairro: str,
    cidade: str,
    uf: str,
    logradouro: str,
) -> Tuple[str, int, str]:
    """
    Returns (level, score_0_100, reason_string).
    Level: Low | Medium | Medium-High | High
    """
    score = 0
    reasons: List[str] = []

    text = normalize(f"{bairro} {cidade} {logradouro}")

    # Prestigious area → strong negative signal
    for kw in PRESTIGIOUS_KEYWORDS:
        if kw in text:
            score = max(0, score - 20)
            reasons.append(f"Prestigious area ({bairro}) — low delivery friction")
            break

    # High-risk keywords (favelas, informal settlements)
    matched = [kw for kw in HIGH_RISK_KEYWORDS if kw in text]
    if matched:
        score += 55
        reasons.append(f'Informal/high-risk area indicator: "{matched[0]}"')

    # Known restricted-delivery peripheral bairros
    if not matched:
        restricted = [kw for kw in RESTRICTED_DELIVERY_BAIRROS if kw in text]
        if restricted:
            score += 35
            reasons.append(f'Known restricted-delivery area: "{restricted[0]}"')

    # State modifier
    state_risk = STATE_RISK.get(uf.upper() if uf else "", "medium_low")
    if state_risk == "high":
        score += 20
        reasons.append(f"High-risk logistics state: {uf}")
    elif state_risk == "medium":
        score += 10
        reasons.append(f"Moderate-risk state: {uf}")
    elif state_risk == "low":
        score = max(0, score - 5)

    # Address completeness signals
    if not logradouro or logradouro.strip() in ("", "-"):
        score += 15
        reasons.append("No street name (possibly rural/informal)")

    if not bairro or bairro.strip() in ("", "-"):
        score += 10
        reasons.append("Missing neighbourhood")

    if re.search(r"\bs/?n\b", text, re.IGNORECASE):
        score += 5
        reasons.append("Address without number (S/N)")

    score = min(100, max(0, score))

    if score <= 25:
        level = "Low"
    elif score <= 50:
        level = "Medium"
    elif score <= 70:
        level = "Medium-High"
    else:
        level = "High"

    reason = " | ".join(reasons) if reasons else "No specific risk indicators found"
    return level, score, reason


def assess_fraud_risk(
    logistics_level: str,
    ses_class: str,
    address_complete: bool,
    uf: str,
) -> Tuple[str, str]:
    """Returns (fraud_level, explanation)."""
    logistics_score = {"Low": 0, "Medium": 25, "Medium-High": 50, "High": 75}
    ses_score = {"A": 0, "B1": 5, "B2": 15, "C1": 25, "C2": 35, "D/E": 55}
    completeness_penalty = 0 if address_complete else 20

    base = (
        logistics_score.get(logistics_level, 25) * 0.45
        + ses_score.get(ses_class, 30) * 0.35
        + completeness_penalty * 0.20
    )

    # State adjustment
    if uf and uf.upper() in ("RJ", "ES"):
        base += 10
    elif uf and uf.upper() in ("BA", "PE", "CE", "AL", "MA"):
        base += 5

    base = min(100, max(0, base))

    if base <= 20:
        return "Low", "Address profile is consistent with legitimate orders"
    if base <= 40:
        return "Medium", "Moderate risk — verify details on high-value orders"
    if base <= 60:
        return "Medium-High", "Elevated risk — logistics + socioeconomic signals combined"
    return "High", "High-risk profile — multiple negative indicators detected"


def generate_recommendation(
    fraud_level: str,
    logistics_level: str,
) -> Tuple[str, str, str]:
    """Returns (icon, short_label, detail)."""
    highs = {"High", "Medium-High"}

    if logistics_level == "High" and fraud_level == "High":
        return (
            "🚫", "Not worth the risk",
            "Critical risk on both logistics and fraud dimensions. "
            "Recommend decline unless additional strong verification is available.",
        )
    if logistics_level == "High" and fraud_level in highs:
        return (
            "🚫", "Not worth the risk",
            "High logistics risk (potential Área de Risco / restricted delivery) "
            "combined with elevated fraud profile. Recommend decline.",
        )
    if fraud_level == "High":
        return (
            "⚠️", "Risky approval",
            "High fraud risk detected. If approving, ensure the order value is low "
            "and require additional identity verification.",
        )
    if fraud_level == "Medium-High" and logistics_level in highs:
        return (
            "⚠️", "Depends on the amount",
            "Moderate-high risk on both dimensions. "
            "Approve low-value orders; decline or manually review orders above R$300.",
        )
    if fraud_level == "Medium-High" or logistics_level == "Medium-High":
        return (
            "⚠️", "Depends on the amount",
            "Elevated risk profile. Consider order value: "
            "routine approvals for small amounts, flag larger ones for review.",
        )
    if fraud_level == "Medium" and logistics_level == "Medium":
        return (
            "🟡", "Approve with standard checks",
            "Moderate risk — within normal parameters. Standard fraud checks apply.",
        )
    return (
        "✅", "You can feel confident approving",
        "Address profile is within acceptable risk parameters. No major red flags detected.",
    )


# ============================================================
# ORCHESTRATION
# ============================================================

def analyze_address(raw: str) -> Dict[str, Any]:
    """Run full analysis pipeline on a single raw address string."""
    result: Dict[str, Any] = {
        "input": raw,
        "validated_address": raw,
        "cep": None,
        "logradouro": "",
        "bairro": "",
        "cidade": "",
        "uf": "",
        "ibge_code": None,
        "pib_per_capita": None,
        "coordinates": None,
        "ses_class": None,
        "ses_explanation": None,
        "logistics_level": None,
        "logistics_score": None,
        "logistics_reason": None,
        "fraud_level": None,
        "fraud_explanation": None,
        "rec_icon": None,
        "rec_label": None,
        "rec_detail": None,
        "api_sources": [],
        "errors": [],
    }

    # ---- Step 1: CEP lookup ----
    cep = extract_cep(raw)
    cep_data = None
    if cep:
        cep_data = lookup_cep(cep)
        if cep_data:
            result["cep"] = cep_data.get("cep", "")
            result["logradouro"] = cep_data.get("logradouro", "")
            result["bairro"] = cep_data.get("bairro", "")
            result["cidade"] = cep_data.get("localidade", "")
            result["uf"] = cep_data.get("uf", "")
            result["ibge_code"] = cep_data.get("ibge", "")
            result["validated_address"] = (
                f"{result['logradouro']}, {result['bairro']}, "
                f"{result['cidade']}, {result['uf']}"
            )
            result["api_sources"].append("ViaCEP")

    # ---- Step 2: Parse if no CEP data ----
    if not cep_data:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 2:
            result["cidade"] = parts[-2] if len(parts) >= 2 else ""
        state_match = re.search(r"\b([A-Z]{2})\b", raw)
        if state_match:
            result["uf"] = state_match.group(1)
        if len(parts) >= 3:
            result["bairro"] = parts[-3]

    # ---- Step 3: Geocode ----
    coords = geocode_address(result["validated_address"])
    if coords:
        result["coordinates"] = coords
        result["api_sources"].append("OpenStreetMap")
    else:
        result["errors"].append("Geocoding failed — map unavailable")

    # ---- Step 4: IBGE PIB ----
    if result["ibge_code"]:
        pib = get_municipal_pib(result["ibge_code"])
        if pib:
            result["pib_per_capita"] = pib
            result["api_sources"].append("IBGE")

    # ---- Step 5: Socioeconomic ----
    ses_class, ses_exp = classify_socioeconomic(
        result["pib_per_capita"],
        result["bairro"],
        result["cidade"],
        result["uf"],
    )
    result["ses_class"] = ses_class
    result["ses_explanation"] = ses_exp

    # ---- Step 6: Logistics ----
    logi_level, logi_score, logi_reason = assess_logistics_risk(
        result["bairro"],
        result["cidade"],
        result["uf"],
        result["logradouro"] or raw,
    )
    result["logistics_level"] = logi_level
    result["logistics_score"] = logi_score
    result["logistics_reason"] = logi_reason

    # ---- Step 7: Fraud ----
    complete = bool(result["bairro"] and result["cidade"] and result["uf"])
    fraud_level, fraud_exp = assess_fraud_risk(
        logi_level, ses_class, complete, result["uf"]
    )
    result["fraud_level"] = fraud_level
    result["fraud_explanation"] = fraud_exp

    # ---- Step 8: Recommendation ----
    icon, label, detail = generate_recommendation(fraud_level, logi_level)
    result["rec_icon"] = icon
    result["rec_label"] = label
    result["rec_detail"] = detail

    return result


# ============================================================
# UI HELPERS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #16213e;
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    border-top: 4px solid;
    height: 100%;
}
.metric-label { font-size: 0.7em; color: #888; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { font-size: 1.35em; font-weight: 700; margin-bottom: 2px; }
.metric-sub   { font-size: 0.75em; color: #aaa; }

.address-header {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    border-left: 4px solid #4a90d9;
}

.rec-box {
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 18px;
    border: 2px solid;
}

.source-tag {
    display: inline-block;
    background: #2a2a3e;
    color: #aaa;
    border-radius: 10px;
    padding: 1px 9px;
    font-size: 0.72em;
    margin: 1px 2px;
}
</style>
"""


def rc(level: str) -> str:
    return RISK_COLORS.get(level, "#888")


def render_result(result: Dict) -> None:
    ses  = result["ses_class"] or "?"
    ses_meta = SOCIOECONOMIC_META.get(ses, {"color": "#888", "label": ses, "desc": ""})
    logi = result["logistics_level"] or "?"
    frau = result["fraud_level"] or "?"

    icon  = result["rec_icon"] or ""
    label = result["rec_label"] or ""
    detail= result["rec_detail"] or ""

    rec_palette = {
        "✅": ("#27ae60", "rgba(39,174,96,0.12)"),
        "🟡": ("#f1c40f", "rgba(241,196,15,0.12)"),
        "⚠️": ("#e67e22", "rgba(230,126,34,0.12)"),
        "🚫": ("#e74c3c", "rgba(231,76,60,0.12)"),
    }
    rb, rbg = rec_palette.get(icon, ("#888", "rgba(136,136,136,0.1)"))

    sources_html = "".join(
        f'<span class="source-tag">{s}</span>' for s in result.get("api_sources", [])
    )
    cep_str = f"CEP {result['cep']}  ·  " if result.get("cep") else ""
    city_str = f"{result['cidade']}, {result['uf']}" if result.get("cidade") else ""

    st.markdown(f"""
    <div class="address-header">
        <div style="font-size:1.05em; font-weight:600; color:#eee;">
            📍 {result['validated_address']}
        </div>
        <div style="font-size:0.8em; color:#666; margin-top:4px;">
            {cep_str}{city_str} &nbsp;&nbsp; {sources_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{ses_meta['color']}">
            <div class="metric-label">Socio-Economic</div>
            <div class="metric-value" style="color:{ses_meta['color']}">{ses_meta['label']}</div>
            <div class="metric-sub">{ses_meta['desc']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        logi_color = rc(logi)
        st.markdown(f"""
        <div class="metric-card" style="border-color:{logi_color}">
            <div class="metric-label">Logistics Risk</div>
            <div class="metric-value" style="color:{logi_color}">{logi}</div>
            <div class="metric-sub">Score {result.get('logistics_score', 0)}/100</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        frau_color = rc(frau)
        frau_sub = (result.get("fraud_explanation") or "")[:40] + "…"
        st.markdown(f"""
        <div class="metric-card" style="border-color:{frau_color}">
            <div class="metric-label">Fraud Risk</div>
            <div class="metric-value" style="color:{frau_color}">{frau}</div>
            <div class="metric-sub">{frau_sub}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{rb}">
            <div class="metric-label">Recommendation</div>
            <div class="metric-value" style="color:{rb}; font-size:1em;">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("📊 Risk breakdown"):
        st.markdown(f"**Socioeconomic:** `{ses}` — {result.get('ses_explanation','')}")
        pib = result.get("pib_per_capita")
        if pib:
            st.markdown(f"**Municipal PIB per capita (IBGE 2021):** R$ {pib:,.0f}/year")
        st.markdown(f"**Logistics risk reason:** {result.get('logistics_reason','')}")
        st.markdown(f"**Fraud assessment:** {result.get('fraud_explanation','')}")
        if result.get("errors"):
            st.warning("⚠️ " + " | ".join(result["errors"]))

    st.markdown(f"""
    <div class="rec-box" style="border-color:{rb}; background:{rbg}">
        <strong style="color:{rb}; font-size:1.05em;">{icon} {label}</strong>
        <p style="color:#ccc; margin:6px 0 0; font-size:0.88em;">{detail}</p>
    </div>
    """, unsafe_allow_html=True)

    if result.get("coordinates"):
        st.markdown("#### 🗺️ Location")
        lat, lon = result["coordinates"]
        m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
        folium.Marker(
            [lat, lon],
            popup=result["validated_address"],
            tooltip=result["validated_address"],
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        st_folium(m, width="100%", height=360)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🇧🇷 Brazil Address Risk Bot")
        st.markdown("---")
        st.markdown("""
**Data Sources (all free, no API key):**

| Source | Data |
|---|---|
| [ViaCEP](https://viacep.com.br) | CEP → address |
| [IBGE Sidra](https://servicodados.ibge.gov.br) | Municipal PIB per capita |
| [Nominatim](https://nominatim.openstreetmap.org) | Geocoding / map |
        """)
        st.markdown("---")
        st.markdown("""
**Risk scoring logic:**

- **Socio-Economic** — IBGE PIB per capita + neighbourhood keyword matching
- **Logistics Risk** — "Área de risco" keywords, state baseline, address completeness
- **Fraud Risk** — weighted composite of all signals + state factor

---
**To run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
        """)
        st.markdown("---")
        st.caption("⚠️ Risk scores are statistical signals, not definitive verdicts. Always combine with other fraud signals.")


# ============================================================
# MAIN APP
# ============================================================

EXAMPLES = [
    "Rua Ataíde Ferreira, Pavuna, Rio de Janeiro, RJ 21510-370",
    "01310-100",  # Av. Paulista CEP
    "Rua Marquês de São Vicente, 209, Gávea, Rio de Janeiro, RJ",
    "Rua das Rosas, Morro do Alemão, Rio de Janeiro, RJ",
    "Av. Brigadeiro Faria Lima, Itaim Bibi, São Paulo, SP",
]


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:4px;'>🇧🇷 Brazil Address Risk Bot</h1>"
        "<p style='text-align:center; color:#888; margin-bottom:28px;'>"
        "Socioeconomic classification · Logistics risk · Fraud assessment</p>",
        unsafe_allow_html=True,
    )

    tab_single, tab_bulk = st.tabs(["🔍 Single Address", "📁 Bulk CSV Upload"])

    # ── TAB 1 ──────────────────────────────────────────────
    with tab_single:
        # Pre-fill box when an example is clicked
        if "addr_val" not in st.session_state:
            st.session_state.addr_val = ""

        st.markdown("**Quick examples:**")
        cols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            with cols[i]:
                label = ex[:28] + "…" if len(ex) > 28 else ex
                if st.button(label, key=f"ex{i}", use_container_width=True, help=ex):
                    st.session_state.addr_val = ex

        # Form ensures input + button are always submitted together
        with st.form("address_form", clear_on_submit=False):
            addr_input = st.text_input(
                "Address",
                value=st.session_state.addr_val,
                placeholder="e.g. Rua Ataíde Ferreira, Pavuna, Rio de Janeiro, RJ  or just the CEP",
                label_visibility="collapsed",
            )
            run = st.form_submit_button("Analyze →", type="primary", use_container_width=False)

        if run and addr_input:
            st.session_state.addr_val = addr_input  # persist typed value
            with st.spinner("Fetching data and scoring…"):
                res = analyze_address(addr_input)
            st.session_state.last_result = res

        if st.session_state.get("last_result"):
            st.divider()
            render_result(st.session_state.last_result)

    # ── TAB 2 ──────────────────────────────────────────────
    with tab_bulk:
        st.markdown("#### Upload a CSV with an `address` column")
        st.markdown(
            "Accepted column names: `address`, `endereco`, `endereço`, `addr`  \n"
            "Any extra columns (order ID, amount, etc.) are preserved in the output."
        )

        # Template download
        tmpl = pd.DataFrame({
            "address": EXAMPLES,
            "order_id": [f"ORD-{i+1:03d}" for i in range(len(EXAMPLES))],
            "amount": [299.99, 1499.0, 89.90, 450.0, 2200.0],
        })
        st.download_button(
            "📥 Download CSV template",
            data=tmpl.to_csv(index=False),
            file_name="address_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            addr_col = next(
                (c for c in df.columns if c.lower().strip() in ("address", "endereco", "endereço", "addr")),
                None,
            )
            if not addr_col:
                st.error("❌ No recognised address column found. Rename it to `address`.")
                st.dataframe(df.head())
            else:
                st.success(f"✅ {len(df)} rows  ·  address column: **{addr_col}**")

                if st.button("🚀 Run bulk analysis", type="primary"):
                    prog = st.progress(0)
                    status = st.empty()
                    rows = []

                    for i, row in df.iterrows():
                        addr = str(row[addr_col])
                        status.text(f"Analyzing {i+1}/{len(df)}: {addr[:70]}…")
                        res = analyze_address(addr)
                        rows.append({
                            "address_input":      addr,
                            "validated_address":  res.get("validated_address", ""),
                            "cep":                res.get("cep", ""),
                            "bairro":             res.get("bairro", ""),
                            "cidade":             res.get("cidade", ""),
                            "uf":                 res.get("uf", ""),
                            "socioeconomic":      res.get("ses_class", ""),
                            "logistics_risk":     res.get("logistics_level", ""),
                            "logistics_score":    res.get("logistics_score", 0),
                            "fraud_risk":         res.get("fraud_level", ""),
                            "recommendation":     res.get("rec_label", ""),
                        })
                        prog.progress((i + 1) / len(df))
                        # Respect Nominatim's 1 req/sec rate limit
                        if i < len(df) - 1:
                            time.sleep(1.1)

                    status.text("✅ Done!")
                    out = pd.DataFrame(rows)
                    extra_cols = [c for c in df.columns if c != addr_col]
                    if extra_cols:
                        out = pd.concat(
                            [out, df[extra_cols].reset_index(drop=True)], axis=1
                        )

                    st.dataframe(out, use_container_width=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        "📥 Download results CSV",
                        data=out.to_csv(index=False),
                        file_name=f"brazil_risk_analysis_{ts}.csv",
                        mime="text/csv",
                    )

    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#444; font-size:0.78em;'>"
        "Sources: ViaCEP · IBGE Sidra · OpenStreetMap/Nominatim  ·  "
        "Risk scores are statistical signals — use as one input among many."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
