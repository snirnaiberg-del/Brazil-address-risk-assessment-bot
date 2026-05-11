"""
Brazil Address Risk Bot
=======================
Analyzes Brazilian addresses for:
  - Socioeconomic Status (IBGE + neighbourhood heuristics)
  - Logistics Risk (Área de Risco detection)
  - Fraud Risk (composite score)
  - Approval Recommendation

Data Sources (all free, no API key needed):
  - ViaCEP            https://viacep.com.br
  - IBGE Sidra API    https://servicodados.ibge.gov.br
  - Nominatim / OSM   https://nominatim.openstreetmap.org
  - Map tiles         OpenStreetMap (via Leaflet.js)
  - Street View       Google Maps (link out)
"""

import re
import time
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import quote_plus

import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🇧🇷 Brazil Address Risk Bot",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HELPERS
# ============================================================

def normalize(text: str) -> str:
    """Strip accents + lowercase. 'Alemão' → 'alemao'."""
    return (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


# ============================================================
# RISK DATA  (all keywords pre-normalised)
# ============================================================

HIGH_RISK_KEYWORDS: List[str] = [normalize(k) for k in [
    # Generic
    "favela", "comunidade", "complexo do", "morro do", "morro da", "morro de",
    "area de risco", "loteamento irregular", "invasao", "ocupacao irregular",
    # RJ
    "rocinha", "alemao", "manguinhos", "mare", "jacarezinho", "jacare favela",
    "acari", "mandela", "cidade de deus", "cidade alta", "vigario geral",
    "parada de lucas", "serrinha", "dendê", "grotao",
    "caramujo", "fallet", "fogueteiro", "tabajara", "cantagalo favela",
    "pavaozinho", "vila kennedy", "vila alianca rj",
    "nova brasilia rj", "parque uniao rj", "nova holanda rj",
    "chatuba", "barreira do vasco",
    # SP
    "heliopolis", "paraisopolis", "jardim angela", "capao redondo",
    "mboi mirim", "brasilandia", "cidade tiradentes", "jardim campo limpo",
    # General
    "sem logradouro", "s/n favela",
]]

RESTRICTED_BAIRROS: List[str] = [normalize(k) for k in [
    # RJ peripheral / restricted delivery
    "pavuna", "anchieta rj", "bangu", "realengo", "paciencia rj",
    "senador vasconcelos", "inhoaiba", "pedra de guaratiba",
    "sepetiba", "madureira", "oswaldo cruz rj", "quintino bocaiuva",
    "bento ribeiro", "campinho rj", "cascadura",
    "honorio gurgel", "iraja", "piedade rj", "pilares rj",
    "vicente de carvalho rj", "vila da penha rj", "turiacu",
    "vaz lobo", "costa barros", "cordovil rj", "penha rj",
    "acari", "coelho neto rj", "barros filho",
    # SP
    "guaianazes", "lajeado sp", "sapopemba", "itaim paulista",
    "parelheiros", "grajau sp", "jardim helena sp",
]]

PRESTIGIOUS_KEYWORDS: List[str] = [normalize(k) for k in [
    # RJ
    "ipanema", "leblon", "gavea", "lagoa rodrigo de freitas",
    "barra da tijuca", "botafogo rj", "urca rj", "sao conrado rj",
    "joatinga", "itanhanga rj", "recreio dos bandeirantes",
    # SP
    "jardins sp", "jardim paulista sp", "jardim europa sp",
    "higienopolis sp", "itaim bibi", "vila nova conceicao",
    "moema sp", "brooklin sp", "alphaville sp",
    # BH
    "savassi", "lourdes bh", "belvedere bh", "mangabeiras bh",
    # Curitiba
    "batel cwb", "agua verde cwb",
]]

STATE_RISK: Dict[str, str] = {
    "RJ": "high",  "ES": "high",
    "BA": "medium", "PE": "medium", "CE": "medium",
    "PA": "medium", "AM": "medium", "MA": "medium",
    "AL": "medium", "SE": "medium", "PI": "medium",
    "SP": "medium_low", "MG": "medium_low", "RN": "medium_low",
    "GO": "low", "DF": "low", "RS": "low",
    "SC": "low",  "PR": "low",  "MT": "low", "MS": "low",
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
    m = re.search(r"\b(\d{5})-?(\d{3})\b", text)
    return m.group(1) + m.group(2) if m else None


def lookup_cep(cep: str) -> Optional[Dict]:
    try:
        r = requests.get(f"https://viacep.com.br/ws/{cep}/json/", timeout=10)
        d = r.json()
        return None if "erro" in d else d
    except Exception:
        return None


def geocode_address(address_str: str) -> Optional[Tuple[float, float]]:
    if "brazil" not in address_str.lower() and "brasil" not in address_str.lower():
        address_str += ", Brazil"
    headers = {"User-Agent": "BrazilAddressRiskBot/2.0 (riskified.com)"}
    params = {"q": address_str, "format": "json", "limit": 1, "countrycodes": "br"}
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params, headers=headers, timeout=15,
        )
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def get_municipal_pib(ibge_code: str) -> Optional[float]:
    if not ibge_code or len(ibge_code) < 6:
        return None
    try:
        r = requests.get(
            "https://servicodados.ibge.gov.br/api/v3/agregados"
            "/5938/periodos/2021/variaveis/6575",
            params={"localidades": f"N6[{ibge_code}]"},
            timeout=12,
        )
        val = r.json()[0]["resultados"][0]["series"][0]["serie"].get("2021")
        return float(val) if val and val not in ("-", "...") else None
    except Exception:
        return None


# ============================================================
# RISK ENGINE
# ============================================================

def classify_socioeconomic(
    pib: Optional[float], bairro: str, cidade: str, uf: str
) -> Tuple[str, str]:
    txt = normalize(f"{bairro} {cidade}")

    for kw in PRESTIGIOUS_KEYWORDS:
        if kw in txt:
            cls = "A" if (pib and pib >= 40000) else "B1"
            return cls, f"Prestigious neighbourhood ({bairro})"

    for kw in HIGH_RISK_KEYWORDS[:20]:
        if kw in txt:
            return "D/E", f"Informal settlement indicator: '{kw}'"

    for kw in RESTRICTED_BAIRROS:
        if kw in txt:
            return "D/E", f"Low-income restricted-delivery area: '{kw}'"

    if pib:
        if pib >= 65000: return "A",   f"High municipal PIB per capita: R${pib:,.0f}"
        if pib >= 45000: return "B1",  f"Upper-middle PIB per capita: R${pib:,.0f}"
        if pib >= 28000: return "B2",  f"Middle PIB per capita: R${pib:,.0f}"
        if pib >= 18000: return "C1",  f"Lower-middle PIB per capita: R${pib:,.0f}"
        if pib >= 12000: return "C2",  f"Low PIB per capita: R${pib:,.0f}"
        return "D/E", f"Very low PIB per capita: R${pib:,.0f}"

    city_map = {
        "sao paulo": "B2", "rio de janeiro": "C1", "brasilia": "B2",
        "curitiba": "B2", "porto alegre": "B2", "florianopolis": "B2",
        "belo horizonte": "C1", "vitoria": "B2", "goiania": "C1",
        "campinas": "B2", "manaus": "C2", "belem": "C2",
        "recife": "C2", "salvador": "C2", "fortaleza": "C2",
        "natal": "C2", "joao pessoa": "C2", "maceio": "C2",
        "teresina": "D/E", "sao luis": "D/E",
    }
    for city, cls in city_map.items():
        if city in txt:
            return cls, f"Estimated from city baseline ({cidade})"

    state_map = {
        "SP": "B2", "RJ": "C1", "DF": "B1", "SC": "B2", "RS": "C1",
        "PR": "B2", "MG": "C1", "GO": "C1", "ES": "C1",
        "CE": "C2", "PE": "C2", "BA": "C2", "PB": "C2", "RN": "C2",
        "AM": "C2", "PA": "D/E", "MA": "D/E", "AL": "D/E",
        "SE": "C2", "PI": "D/E", "AC": "D/E", "AP": "D/E", "TO": "D/E",
    }
    cls = state_map.get((uf or "").upper(), "C2")
    return cls, f"Estimated from state baseline ({uf})"


def assess_logistics_risk(
    bairro: str, cidade: str, uf: str, logradouro: str
) -> Tuple[str, int, str]:
    score = 0
    reasons: List[str] = []
    txt = normalize(f"{bairro} {cidade} {logradouro}")

    # Prestigious → reduce risk
    for kw in PRESTIGIOUS_KEYWORDS:
        if kw in txt:
            score = max(0, score - 20)
            reasons.append(f"Prestigious area ({bairro})")
            break

    # Favela / informal settlement
    matched_high = [kw for kw in HIGH_RISK_KEYWORDS if kw in txt]
    if matched_high:
        score += 60
        reasons.append(f'Informal/high-risk area: "{matched_high[0]}"')

    # Restricted-delivery peripheral bairro
    if not matched_high:
        matched_rest = [kw for kw in RESTRICTED_BAIRROS if kw in txt]
        if matched_rest:
            score += 40
            reasons.append(f'Restricted-delivery area: "{matched_rest[0]}"')

    # State risk
    state_risk = STATE_RISK.get((uf or "").upper(), "medium_low")
    if state_risk == "high":
        score += 25; reasons.append(f"High-risk state ({uf})")
    elif state_risk == "medium":
        score += 15; reasons.append(f"Moderate-risk state ({uf})")
    elif state_risk == "medium_low":
        score += 5
    # low → no change

    # Address completeness
    if not logradouro or normalize(logradouro) in ("", "-"):
        score += 15; reasons.append("No street name")
    if not bairro or normalize(bairro) in ("", "-"):
        score += 10; reasons.append("Missing neighbourhood")

    score = min(100, max(0, score))
    if score <= 20:   level = "Low"
    elif score <= 45: level = "Medium"
    elif score <= 65: level = "Medium-High"
    else:             level = "High"

    return level, score, (" | ".join(reasons) if reasons else "No specific risk flags")


def assess_fraud_risk(
    logistics_level: str, ses_class: str, complete: bool, uf: str
) -> Tuple[str, str]:
    l = {"Low": 0, "Medium": 25, "Medium-High": 50, "High": 75}
    s = {"A": 0, "B1": 5, "B2": 15, "C1": 25, "C2": 35, "D/E": 55}
    c = 0 if complete else 20
    base = l.get(logistics_level, 25) * 0.45 + s.get(ses_class, 30) * 0.35 + c * 0.20
    if (uf or "").upper() in ("RJ", "ES"):     base += 10
    elif (uf or "").upper() in ("BA", "PE", "CE", "AL", "MA"): base += 5
    base = min(100, max(0, base))

    if base <= 20: return "Low",         "Address profile consistent with legitimate orders"
    if base <= 40: return "Medium",      "Moderate risk — verify details on high-value orders"
    if base <= 60: return "Medium-High", "Elevated risk — multiple negative signals combined"
    return         "High",               "High-risk profile — multiple negative indicators"


def generate_recommendation(fraud: str, logistics: str) -> Tuple[str, str, str]:
    highs = {"High", "Medium-High"}
    if logistics == "High" and fraud == "High":
        return "🚫", "Not worth the risk", \
            "Critical risk on both dimensions. Recommend decline unless strong additional verification."
    if logistics == "High" and fraud in highs:
        return "🚫", "Not worth the risk", \
            "High logistics risk (Área de Risco / restricted delivery) + elevated fraud. Recommend decline."
    if fraud == "High":
        return "⚠️", "Risky approval", \
            "High fraud risk. If approving, keep order value low and require identity verification."
    if fraud == "Medium-High" and logistics in highs:
        return "⚠️", "Depends on the amount", \
            "Moderate-high risk on both dimensions. Approve small orders; decline or review R$300+."
    if fraud == "Medium-High" or logistics == "Medium-High":
        return "⚠️", "Depends on the amount", \
            "Elevated risk profile. Routine approvals for small amounts; flag larger ones."
    if fraud == "Medium" and logistics == "Medium":
        return "🟡", "Approve with standard checks", \
            "Within normal risk parameters. Standard fraud checks apply."
    return "✅", "You can feel confident approving", \
        "Address profile is within acceptable risk parameters. No major red flags."


# ============================================================
# ORCHESTRATION
# ============================================================

def analyze_address(raw: str) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "input": raw, "validated_address": raw,
        "cep": None, "logradouro": "", "bairro": "", "cidade": "", "uf": "",
        "ibge_code": None, "pib_per_capita": None, "coordinates": None,
        "ses_class": None, "ses_explanation": None,
        "logistics_level": None, "logistics_score": None, "logistics_reason": None,
        "fraud_level": None, "fraud_explanation": None,
        "rec_icon": None, "rec_label": None, "rec_detail": None,
        "api_sources": [], "errors": [],
    }

    # 1 — CEP lookup
    cep = extract_cep(raw)
    cep_data = None
    if cep:
        cep_data = lookup_cep(cep)
        if cep_data:
            r["cep"]        = cep_data.get("cep", "")
            r["logradouro"] = cep_data.get("logradouro", "")
            r["bairro"]     = cep_data.get("bairro", "")
            r["cidade"]     = cep_data.get("localidade", "")
            r["uf"]         = cep_data.get("uf", "")
            r["ibge_code"]  = cep_data.get("ibge", "")
            r["validated_address"] = (
                f"{r['logradouro']}, {r['bairro']}, {r['cidade']}, {r['uf']}"
            )
            r["api_sources"].append("ViaCEP")

    # 2 — Parse raw if no CEP
    if not cep_data:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 3:
            r["bairro"] = parts[-3]
        if len(parts) >= 2:
            r["cidade"] = parts[-2]
        m = re.search(r"\b([A-Z]{2})\b", raw)
        if m:
            r["uf"] = m.group(1)

    # 3 — Geocode
    coords = geocode_address(r["validated_address"])
    if coords:
        r["coordinates"] = coords
        r["api_sources"].append("OpenStreetMap")
    else:
        r["errors"].append("Geocoding failed")

    # 4 — IBGE PIB
    if r["ibge_code"]:
        pib = get_municipal_pib(r["ibge_code"])
        if pib:
            r["pib_per_capita"] = pib
            r["api_sources"].append("IBGE")

    # 5 — Socioeconomic
    r["ses_class"], r["ses_explanation"] = classify_socioeconomic(
        r["pib_per_capita"], r["bairro"], r["cidade"], r["uf"]
    )

    # 6 — Logistics
    r["logistics_level"], r["logistics_score"], r["logistics_reason"] = assess_logistics_risk(
        r["bairro"], r["cidade"], r["uf"], r["logradouro"] or raw
    )

    # 7 — Fraud
    complete = bool(r["bairro"] and r["cidade"] and r["uf"])
    r["fraud_level"], r["fraud_explanation"] = assess_fraud_risk(
        r["logistics_level"], r["ses_class"], complete, r["uf"]
    )

    # 8 — Recommendation
    r["rec_icon"], r["rec_label"], r["rec_detail"] = generate_recommendation(
        r["fraud_level"], r["logistics_level"]
    )

    return r


# ============================================================
# MAP  (Leaflet.js — no extra package needed)
# ============================================================

def render_map(lat: float, lon: float, address: str) -> None:
    escaped = address.replace("'", "\\'")
    html = f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <div id="map" style="height:380px; border-radius:10px; overflow:hidden;"></div>
    <script>
      var map = L.map('map').setView([{lat}, {lon}], 16);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
        {{attribution:'© OpenStreetMap'}}).addTo(map);
      L.marker([{lat}, {lon}]).addTo(map)
        .bindPopup('{escaped}').openPopup();
    </script>
    """
    components.html(html, height=395)


# ============================================================
# RESULT CARD
# ============================================================

CSS = """<style>
.metric-card {
    background: #16213e; border-radius: 10px; padding: 18px 16px;
    text-align: center; border-top: 4px solid; height: 100%;
}
.metric-label { font-size:.7em; color:#888; text-transform:uppercase;
    letter-spacing:.08em; margin-bottom:6px; }
.metric-value { font-size:1.35em; font-weight:700; margin-bottom:2px; }
.metric-sub   { font-size:.75em; color:#aaa; }
.addr-box {
    background:#1a1a2e; border-radius:10px; padding:16px 20px;
    margin-bottom:16px; border-left:4px solid #4a90d9;
}
.rec-box { border-radius:10px; padding:16px 20px; margin-top:18px; border:2px solid; }
.src-tag { display:inline-block; background:#2a2a3e; color:#aaa;
    border-radius:10px; padding:1px 9px; font-size:.72em; margin:1px 2px; }
</style>"""


def rc(level: str) -> str:
    return RISK_COLORS.get(level, "#888")


def render_result(res: Dict) -> None:
    ses      = res.get("ses_class") or "?"
    ses_meta = SOCIOECONOMIC_META.get(ses, {"color": "#888", "label": ses, "desc": ""})
    logi     = res.get("logistics_level") or "?"
    frau     = res.get("fraud_level") or "?"
    icon     = res.get("rec_icon", "")
    label    = res.get("rec_label", "")
    detail   = res.get("rec_detail", "")

    rec_pal = {
        "✅": ("#27ae60", "rgba(39,174,96,.12)"),
        "🟡": ("#f1c40f", "rgba(241,196,15,.12)"),
        "⚠️": ("#e67e22", "rgba(230,126,34,.12)"),
        "🚫": ("#e74c3c", "rgba(231,76,60,.12)"),
    }
    rb, rbg = rec_pal.get(icon, ("#888", "rgba(136,136,136,.1)"))

    srcs = "".join(f'<span class="src-tag">{s}</span>' for s in res.get("api_sources", []))
    cep_s  = f"CEP {res['cep']}  ·  " if res.get("cep") else ""
    city_s = f"{res['cidade']}, {res['uf']}" if res.get("cidade") else ""

    st.markdown(f"""
    <div class="addr-box">
        <div style="font-size:1.05em;font-weight:600;color:#eee;">
            📍 {res['validated_address']}
        </div>
        <div style="font-size:.8em;color:#666;margin-top:4px;">
            {cep_s}{city_s}&nbsp;&nbsp;{srcs}
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-color:{ses_meta['color']}">
            <div class="metric-label">Socio-Economic</div>
            <div class="metric-value" style="color:{ses_meta['color']}">{ses_meta['label']}</div>
            <div class="metric-sub">{ses_meta['desc']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        lc = rc(logi)
        st.markdown(f"""<div class="metric-card" style="border-color:{lc}">
            <div class="metric-label">Logistics Risk</div>
            <div class="metric-value" style="color:{lc}">{logi}</div>
            <div class="metric-sub">Score {res.get('logistics_score',0)}/100</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        fc = rc(frau)
        fsub = (res.get("fraud_explanation") or "")[:40] + "…"
        st.markdown(f"""<div class="metric-card" style="border-color:{fc}">
            <div class="metric-label">Fraud Risk</div>
            <div class="metric-value" style="color:{fc}">{frau}</div>
            <div class="metric-sub">{fsub}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-color:{rb}">
            <div class="metric-label">Recommendation</div>
            <div class="metric-value" style="color:{rb};font-size:1em;">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("📊 Risk breakdown"):
        st.markdown(f"**Socioeconomic:** `{ses}` — {res.get('ses_explanation','')}")
        if res.get("pib_per_capita"):
            st.markdown(f"**Municipal PIB per capita (IBGE 2021):** R$ {res['pib_per_capita']:,.0f}/year")
        st.markdown(f"**Logistics reason:** {res.get('logistics_reason','')}")
        st.markdown(f"**Fraud assessment:** {res.get('fraud_explanation','')}")
        if res.get("errors"):
            st.warning("⚠️ " + " | ".join(res["errors"]))

    st.markdown(f"""
    <div class="rec-box" style="border-color:{rb};background:{rbg}">
        <strong style="color:{rb};font-size:1.05em;">{icon} {label}</strong>
        <p style="color:#ccc;margin:6px 0 0;font-size:.88em;">{detail}</p>
    </div>""", unsafe_allow_html=True)

    # Map + Street View
    coords = res.get("coordinates")
    if coords:
        lat, lon = coords
        st.markdown("#### 🗺️ Location")
        col_map, col_sv = st.columns([4, 1])
        with col_map:
            render_map(lat, lon, res["validated_address"])
        with col_sv:
            sv_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
            gmaps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            st.markdown(f"""
            <div style="padding:12px 0">
                <a href="{sv_url}" target="_blank"
                   style="display:block;background:#e67e22;color:white;padding:10px 16px;
                   border-radius:8px;text-decoration:none;text-align:center;margin-bottom:10px;font-weight:600;">
                   📸 Street View
                </a>
                <a href="{gmaps_url}" target="_blank"
                   style="display:block;background:#4a90d9;color:white;padding:10px 16px;
                   border-radius:8px;text-decoration:none;text-align:center;font-weight:600;">
                   🗺️ Open in Maps
                </a>
            </div>""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🇧🇷 Brazil Address Risk Bot")
        st.divider()
        st.markdown("""
**Data Sources** (free, no API key)

| Source | Data |
|---|---|
| [ViaCEP](https://viacep.com.br) | CEP → address |
| [IBGE](https://servicodados.ibge.gov.br) | Municipal PIB |
| [Nominatim](https://nominatim.openstreetmap.org) | Geocoding |
| OpenStreetMap | Map tiles |
| Google Maps | Street View link |
""")
        st.divider()
        st.markdown("""
**Scoring logic**

- **Socio-Economic** — IBGE PIB + keyword matching
- **Logistics** — Área de Risco keywords, restricted-delivery bairros, state baseline
- **Fraud** — Composite: 45% logistics + 35% SES + 20% completeness + state factor
""")
        st.divider()
        st.caption("⚠️ Risk scores are statistical signals. Always combine with other fraud signals.")


# ============================================================
# MAIN
# ============================================================

EXAMPLES = [
    "Rua Ataíde Ferreira, Pavuna, Rio de Janeiro, RJ 21510-370",
    "01310-100",
    "Rua Marquês de São Vicente, 209, Gávea, Rio de Janeiro, RJ",
    "Morro do Alemão, Rio de Janeiro, RJ",
    "Av. Brigadeiro Faria Lima, Itaim Bibi, São Paulo, SP",
]


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    st.markdown(
        "<h1 style='text-align:center;margin-bottom:4px;'>🇧🇷 Brazil Address Risk Bot</h1>"
        "<p style='text-align:center;color:#888;margin-bottom:28px;'>"
        "Socioeconomic · Logistics Risk · Fraud Assessment</p>",
        unsafe_allow_html=True,
    )

    tab_single, tab_bulk = st.tabs(["🔍 Single Address", "📁 Bulk CSV"])

    # ── SINGLE ADDRESS ──────────────────────────────────────
    with tab_single:
        # Session state init
        if "addr_input" not in st.session_state:
            st.session_state.addr_input = ""
        if "last_result" not in st.session_state:
            st.session_state.last_result = None

        # Quick examples
        st.markdown("**Quick examples:**")
        ecols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            lbl = ex[:28] + "…" if len(ex) > 28 else ex
            with ecols[i]:
                if st.button(lbl, key=f"ex{i}", use_container_width=True, help=ex):
                    st.session_state.addr_input = ex
                    st.rerun()

        # Input + button — key binds text box to session state
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            st.text_input(
                "Address",
                key="addr_input",
                placeholder="e.g. Rua Ataíde Ferreira, Pavuna, Rio de Janeiro, RJ  or just the CEP",
                label_visibility="collapsed",
            )
        with col_btn:
            run = st.button("Analyze →", type="primary", use_container_width=True)

        # Analyze on button click
        if run:
            current = st.session_state.addr_input.strip()
            if current:
                with st.spinner("Fetching data and scoring…"):
                    res = analyze_address(current)
                st.session_state.last_result = res
            else:
                st.warning("Please enter an address first.")

        # Always render last result
        if st.session_state.last_result:
            st.divider()
            render_result(st.session_state.last_result)

    # ── BULK CSV ────────────────────────────────────────────
    with tab_bulk:
        st.markdown("#### Upload a CSV with an `address` column")
        st.markdown(
            "Accepted column names: `address`, `endereco`, `addr`  \n"
            "Extra columns are preserved in the output."
        )

        tmpl = pd.DataFrame({
            "address": EXAMPLES,
            "order_id": [f"ORD-{i+1:03d}" for i in range(len(EXAMPLES))],
            "amount":   [299.99, 1499.0, 89.90, 450.0, 2200.0],
        })
        st.download_button("📥 Download CSV template", tmpl.to_csv(index=False),
                           "address_template.csv", "text/csv")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            addr_col = next(
                (c for c in df.columns if c.lower().strip() in ("address","endereco","endereço","addr")),
                None,
            )
            if not addr_col:
                st.error("❌ No recognised address column found. Rename it to `address`.")
                st.dataframe(df.head())
            else:
                st.success(f"✅ {len(df)} rows  ·  column: **{addr_col}**")
                if st.button("🚀 Run bulk analysis", type="primary"):
                    prog   = st.progress(0)
                    status = st.empty()
                    rows   = []
                    for i, row in df.iterrows():
                        addr = str(row[addr_col])
                        status.text(f"Analyzing {i+1}/{len(df)}: {addr[:70]}…")
                        res = analyze_address(addr)
                        rows.append({
                            "address_input":     addr,
                            "validated_address": res.get("validated_address",""),
                            "cep":               res.get("cep",""),
                            "bairro":            res.get("bairro",""),
                            "cidade":            res.get("cidade",""),
                            "uf":                res.get("uf",""),
                            "socioeconomic":     res.get("ses_class",""),
                            "logistics_risk":    res.get("logistics_level",""),
                            "logistics_score":   res.get("logistics_score",0),
                            "fraud_risk":        res.get("fraud_level",""),
                            "recommendation":    res.get("rec_label",""),
                        })
                        prog.progress((i+1)/len(df))
                        if i < len(df)-1:
                            time.sleep(1.1)  # Nominatim rate limit

                    status.text("✅ Done!")
                    out = pd.DataFrame(rows)
                    extra = [c for c in df.columns if c != addr_col]
                    if extra:
                        out = pd.concat([out, df[extra].reset_index(drop=True)], axis=1)
                    st.dataframe(out, use_container_width=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        "📥 Download results", out.to_csv(index=False),
                        f"brazil_risk_{ts}.csv", "text/csv",
                    )

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#444;font-size:.78em;'>"
        "Sources: ViaCEP · IBGE · OpenStreetMap  ·  "
        "Risk scores are statistical signals — use as one input among many."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
