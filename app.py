# ============================================================
# ADAPTIVE XAI IDS DASHBOARD
# PROJECT 1 — X-AI
# ------------------------------------------------------------
# UI EDITION: themed / futuristic interface
#
# Only the presentation layer was redesigned. Project paths,
# dataset configuration, model loading, prediction, SHAP, LIME,
# XAI JSON building, the Gemini prompt and all saved output
# files are the same as the original app.
# ============================================================

import os
import json
import re
import html                      # standard library (HTML escaping)
from datetime import datetime    # standard library (report timestamp)
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptive Explainable IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# UI THEME ENGINE  (presentation only — no model / data logic)
# ============================================================

THEMES = {

    "🌌 Cyber Neon": {
        "scheme": "dark",
        "bg": "#050914", "bg2": "#0b1329",
        "card": "rgba(13,22,46,0.72)",
        "text": "#e8f1ff", "muted": "#8da2c7",
        "accent": "#00e5ff", "accent2": "#7c4dff",
        "border": "rgba(0,229,255,0.24)",
        "glow1": "rgba(0,229,255,0.13)", "glow2": "rgba(124,77,255,0.17)",
        "grid": "rgba(0,229,255,0.04)",
        "input_bg": "rgba(4,10,26,0.92)", "btn_text": "#03111a",
        "danger": "#ff3d71", "safe": "#00e676", "warn": "#ffb300",
        "shadow": "0 10px 40px rgba(0,0,0,0.45)",
        "track": "rgba(141,162,199,0.18)",
    },

    "🟢 Matrix Green": {
        "scheme": "dark",
        "bg": "#030a06", "bg2": "#07150d",
        "card": "rgba(8,24,14,0.74)",
        "text": "#dcffe9", "muted": "#7fb596",
        "accent": "#2bff88", "accent2": "#00b3a4",
        "border": "rgba(43,255,136,0.24)",
        "glow1": "rgba(43,255,136,0.11)", "glow2": "rgba(0,179,164,0.14)",
        "grid": "rgba(43,255,136,0.04)",
        "input_bg": "rgba(2,10,6,0.92)", "btn_text": "#02130a",
        "danger": "#ff4d6d", "safe": "#2bff88", "warn": "#ffd23f",
        "shadow": "0 10px 40px rgba(0,0,0,0.5)",
        "track": "rgba(127,181,150,0.18)",
    },

    "🔮 Quantum Violet": {
        "scheme": "dark",
        "bg": "#0a0613", "bg2": "#150c27",
        "card": "rgba(28,16,52,0.72)",
        "text": "#f1e9ff", "muted": "#a994cf",
        "accent": "#c77dff", "accent2": "#ff4fd8",
        "border": "rgba(199,125,255,0.26)",
        "glow1": "rgba(199,125,255,0.15)", "glow2": "rgba(255,79,216,0.12)",
        "grid": "rgba(199,125,255,0.045)",
        "input_bg": "rgba(10,5,22,0.92)", "btn_text": "#1a0830",
        "danger": "#ff3d71", "safe": "#3ddc97", "warn": "#ffb703",
        "shadow": "0 10px 40px rgba(0,0,0,0.5)",
        "track": "rgba(169,148,207,0.2)",
    },

    "🔥 Solar Flare": {
        "scheme": "dark",
        "bg": "#0f0805", "bg2": "#1d0f08",
        "card": "rgba(40,20,10,0.72)",
        "text": "#fff0e4", "muted": "#c9a58b",
        "accent": "#ffab40", "accent2": "#ff5252",
        "border": "rgba(255,171,64,0.26)",
        "glow1": "rgba(255,171,64,0.13)", "glow2": "rgba(255,82,82,0.12)",
        "grid": "rgba(255,171,64,0.04)",
        "input_bg": "rgba(18,8,4,0.92)", "btn_text": "#1f0d00",
        "danger": "#ff4d4d", "safe": "#46d37b", "warn": "#ffd54f",
        "shadow": "0 10px 40px rgba(0,0,0,0.5)",
        "track": "rgba(201,165,139,0.2)",
    },

    "❄️ Arctic Light": {
        "scheme": "light",
        "bg": "#eef3fb", "bg2": "#ffffff",
        "card": "rgba(255,255,255,0.86)",
        "text": "#10213a", "muted": "#5b6f8f",
        "accent": "#0077ff", "accent2": "#00c2d1",
        "border": "rgba(0,119,255,0.24)",
        "glow1": "rgba(0,119,255,0.12)", "glow2": "rgba(0,194,209,0.15)",
        "grid": "rgba(0,119,255,0.05)",
        "input_bg": "#ffffff", "btn_text": "#ffffff",
        "danger": "#e5195a", "safe": "#0f9d58", "warn": "#d98200",
        "shadow": "0 10px 30px rgba(16,33,58,0.10)",
        "track": "rgba(16,33,58,0.10)",
    },
}


FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Orbitron:wght@500;700;800;900&"
    "family=Inter:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;600;700&display=swap');"
)


BASE_CSS = r"""
html,body,.stApp{font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif}
.stApp{color:var(--text);color-scheme:var(--scheme);
background:linear-gradient(var(--grid) 1px,transparent 1px) 0 0/46px 46px,
linear-gradient(90deg,var(--grid) 1px,transparent 1px) 0 0/46px 46px,
radial-gradient(900px 520px at 6% -8%,var(--glow1),transparent 62%),
radial-gradient(820px 480px at 100% 0%,var(--glow2),transparent 58%),
var(--bg) !important;background-attachment:fixed !important}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:transparent !important}
[data-testid="stHeader"]{background:transparent !important}
[data-testid="stHeader"] button,[data-testid="stHeader"] svg{color:var(--text) !important}
[data-testid="stDecoration"]{display:none}
footer{visibility:hidden}
.element-container:has(style),[data-testid="stElementContainer"]:has(style){display:none}
.block-container{padding-top:1.4rem !important;padding-bottom:3rem !important;max-width:1360px}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:8px}
::-webkit-scrollbar-track{background:transparent}
hr{border:none !important;height:1px !important;background:linear-gradient(90deg,transparent,var(--accent),transparent) !important;opacity:.45;margin:1.6rem 0 !important}

/* ---------- typography ---------- */
h1,h2,h3,h4,h5,h6{color:var(--text) !important}
h2{font-family:'Orbitron',sans-serif !important;font-size:1.15rem !important;font-weight:700 !important;letter-spacing:2px;text-transform:uppercase;border-left:3px solid var(--accent);padding-left:.75rem !important}
h3{font-size:1.02rem !important;font-weight:600 !important;letter-spacing:.3px}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:var(--text)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--muted) !important}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--bg2) 0%,var(--bg) 100%) !important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] > div,[data-testid="stSidebarContent"]{background:transparent !important}
[data-testid="stSidebar"] *{color:var(--text)}

/* ---------- widgets ---------- */
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{color:var(--muted) !important;font-size:.8rem}
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p{font-family:'JetBrains Mono',monospace;font-size:.74rem}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"] > div,[data-baseweb="textarea"]{background:var(--input-bg) !important;border-color:var(--border) !important;border-radius:10px !important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"] > div:focus-within{border-color:var(--accent) !important;box-shadow:0 0 0 1px var(--accent),0 0 18px color-mix(in srgb,var(--accent) 35%,transparent) !important}
input,textarea{color:var(--text) !important;-webkit-text-fill-color:var(--text) !important;font-family:'JetBrains Mono',monospace !important}
[data-baseweb="select"] *{color:var(--text) !important}
[data-baseweb="select"] svg{fill:var(--accent) !important}
[data-testid="stNumberInput"] button{background:var(--input-bg) !important;color:var(--text) !important;border-color:var(--border) !important}
[data-baseweb="popover"] > div,[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--bg2) !important;border:1px solid var(--border);border-radius:12px}
[data-baseweb="popover"] li,[data-baseweb="menu"] li,[role="option"]{background:transparent !important;color:var(--text) !important}
[data-baseweb="popover"] li:hover,[role="option"]:hover,[role="option"][aria-selected="true"]{background:color-mix(in srgb,var(--accent) 18%,transparent) !important}
[data-testid="stCheckbox"] *,[data-testid="stRadio"] label,[data-testid="stRadio"] label *{color:var(--text) !important}
label[data-baseweb="checkbox"]:has(input:checked) > span:first-child{background-color:var(--accent) !important;border-color:var(--accent) !important}

/* ---------- buttons ---------- */
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{
background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;color:var(--btn-text) !important;border:none !important;border-radius:12px !important;
font-family:'Orbitron',sans-serif !important;font-weight:700 !important;letter-spacing:1.5px;text-transform:uppercase;padding:.65rem 1.6rem !important;
box-shadow:0 0 22px color-mix(in srgb,var(--accent) 45%,transparent);transition:transform .15s ease,box-shadow .15s ease,filter .15s ease}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{
transform:translateY(-2px);filter:brightness(1.1);box-shadow:0 0 36px color-mix(in srgb,var(--accent) 65%,transparent)}
button[kind="primary"] *,button[kind="primaryFormSubmit"] *,[data-testid="stBaseButton-primary"] *,[data-testid="stBaseButton-primaryFormSubmit"] *{color:inherit !important;font-family:inherit !important}
button[kind="secondary"],[data-testid="stBaseButton-secondary"]{background:transparent !important;color:var(--accent) !important;border:1px solid var(--border) !important;border-radius:12px !important}
button[kind="secondary"]:hover,[data-testid="stBaseButton-secondary"]:hover{border-color:var(--accent) !important;background:color-mix(in srgb,var(--accent) 12%,transparent) !important}
button[kind="secondary"] *,[data-testid="stBaseButton-secondary"] *{color:var(--accent) !important}

/* ---------- full-width action buttons ---------- */
.stButton > button,[data-testid="stFormSubmitButton"] > button{width:100%}

/* ---------- containers ---------- */
[data-testid="stForm"]{background:var(--card);border:1px solid var(--border) !important;border-radius:20px !important;padding:1.4rem 1.4rem 1rem !important;box-shadow:var(--shadow);backdrop-filter:blur(10px)}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--border) !important;border-radius:14px !important;overflow:hidden}
[data-testid="stExpander"] details{border:none !important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--text) !important}
[data-testid="stExpander"] summary:hover,[data-testid="stExpander"] summary:hover *{color:var(--accent) !important}
button[data-baseweb="tab"]{color:var(--muted) !important;font-family:'Orbitron',sans-serif;font-size:.78rem;letter-spacing:1.2px;text-transform:uppercase}
button[data-baseweb="tab"] *{color:inherit !important}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent) !important}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--accent),var(--accent2)) !important;height:3px !important}
[data-baseweb="tab-border"]{background:var(--border) !important}
[data-testid="stAlert"]{border-radius:14px !important;border:1px solid var(--border) !important;backdrop-filter:blur(6px)}
[data-testid="stAlert"] *{color:var(--text) !important}
[data-testid="stSpinner"] *{color:var(--muted) !important}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:.8rem 1rem}
[data-testid="stMetricValue"]{font-family:'Orbitron',sans-serif;color:var(--accent) !important}

/* ---------- hero ---------- */
.ids-hero{position:relative;padding:1.6rem 1.8rem;border:1px solid var(--border);border-radius:22px;overflow:hidden;margin-bottom:1rem;box-shadow:var(--shadow);
background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 10%,var(--card)),var(--card) 55%,color-mix(in srgb,var(--accent2) 12%,var(--card)))}
.ids-hero::after{content:"";position:absolute;inset:0;pointer-events:none;transform:translateX(-100%);animation:ids-sweep 7s ease-in-out infinite;
background:linear-gradient(115deg,transparent 40%,color-mix(in srgb,var(--accent) 10%,transparent) 50%,transparent 60%)}
@keyframes ids-sweep{0%,55%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.ids-badge{display:inline-flex;align-items:center;gap:.5rem;font:600 .68rem 'JetBrains Mono',monospace;letter-spacing:2px;color:var(--accent);border:1px solid var(--border);padding:.28rem .75rem;border-radius:999px;background:color-mix(in srgb,var(--accent) 8%,transparent)}
.ids-dot{width:8px;height:8px;border-radius:50%;background:var(--safe);box-shadow:0 0 10px var(--safe);animation:ids-pulse 1.8s infinite}
@keyframes ids-pulse{50%{opacity:.3}}
.ids-title{font-family:'Orbitron',sans-serif;font-weight:900;font-size:clamp(1.4rem,3vw,2.3rem);line-height:1.18;margin:.85rem 0 .45rem;color:var(--text)}
.ids-title b{background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.ids-sub{color:var(--muted);max-width:920px;line-height:1.6;font-size:.95rem}

/* ---------- pipeline tracker ---------- */
.ids-steps{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem;margin:.3rem 0 1rem}
.ids-step{display:inline-flex;align-items:center;gap:.4rem;padding:.4rem .85rem;border-radius:999px;border:1px solid var(--border);background:var(--card);font:600 .74rem 'Inter',sans-serif;color:var(--muted)}
.ids-step.done{color:var(--text);border-color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,transparent)}
.ids-step.active{color:var(--btn-text);background:linear-gradient(135deg,var(--accent),var(--accent2));border-color:transparent;animation:ids-glow 1.3s infinite}
.ids-step.skip{opacity:.45;text-decoration:line-through}
.ids-step.fail{color:var(--danger);border-color:var(--danger);background:color-mix(in srgb,var(--danger) 12%,transparent)}
.ids-arrow{color:var(--muted);opacity:.6}
@keyframes ids-glow{50%{box-shadow:0 0 18px color-mix(in srgb,var(--accent) 65%,transparent)}}

/* ---------- generic cards ---------- */
.ids-card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:1.1rem 1.3rem;box-shadow:var(--shadow);backdrop-filter:blur(10px);margin-top:.9rem}
.ids-sec{display:flex;align-items:center;gap:.7rem;font:700 .8rem 'Orbitron',sans-serif;letter-spacing:2.4px;text-transform:uppercase;color:var(--accent);margin:0 0 .85rem}
.ids-sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
.ids-empty{color:var(--muted);font-size:.86rem;font-style:italic}

/* ---------- verdict ---------- */
.ids-verdict{display:flex;gap:1.6rem;align-items:center;flex-wrap:wrap;padding:1.4rem 1.6rem;border-radius:22px;border:1px solid var(--vc);
background:linear-gradient(135deg,color-mix(in srgb,var(--vc) 14%,var(--card)),var(--card) 70%);
box-shadow:var(--shadow),0 0 40px color-mix(in srgb,var(--vc) 16%,transparent)}
.ids-verdict.bad{--vc:var(--danger)}
.ids-verdict.ok{--vc:var(--safe)}
.ids-verdict.unk{--vc:var(--accent)}
.ids-ring{--size:128px;width:var(--size);height:var(--size);border-radius:50%;display:grid;place-items:center;flex:none;
background:conic-gradient(var(--vc) calc(var(--pct) * 1%),var(--track) 0);box-shadow:0 0 30px color-mix(in srgb,var(--vc) 35%,transparent)}
.ids-ring-in{width:calc(var(--size) - 24px);height:calc(var(--size) - 24px);border-radius:50%;background:var(--bg2);display:flex;flex-direction:column;align-items:center;justify-content:center}
.ids-ring-in b{font:800 1.4rem 'Orbitron',sans-serif;color:var(--text)}
.ids-ring-in span{font:600 .6rem 'JetBrains Mono',monospace;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase}
.ids-vmain{flex:1;min-width:240px}
.ids-eyebrow{font:600 .68rem 'JetBrains Mono',monospace;letter-spacing:3px;color:var(--vc);text-transform:uppercase}
.ids-vlabel{font:900 clamp(1.4rem,2.6vw,2.1rem) 'Orbitron',sans-serif;color:var(--vc);margin:.25rem 0}
.ids-vsub{color:var(--muted);font-size:.92rem;line-height:1.55;max-width:660px}
.ids-vsub b{color:var(--text)}
.ids-chips{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.85rem}
.ids-chip{font:500 .74rem 'JetBrains Mono',monospace;padding:.28rem .7rem;border-radius:8px;border:1px solid var(--border);background:color-mix(in srgb,var(--accent) 7%,transparent);color:var(--text)}
.ids-chip i{font-style:normal;color:var(--muted);margin-right:.4rem}

/* ---------- probabilities ---------- */
.ids-prob{display:grid;gap:.75rem}
.ids-prow{display:grid;grid-template-columns:90px 1fr 100px;align-items:center;gap:.8rem;font:500 .8rem 'JetBrains Mono',monospace;color:var(--muted)}
.ids-pbar{height:12px;border-radius:99px;background:var(--track);overflow:hidden}
.ids-pbar i{display:block;height:100%;border-radius:99px}
.ids-prow b{color:var(--text);text-align:right}

/* ---------- LLM threat report ---------- */
.ids-report{border:1px solid var(--border);border-radius:22px;background:var(--card);box-shadow:var(--shadow);overflow:hidden;backdrop-filter:blur(10px)}
.ids-rhead{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.6rem;padding:.9rem 1.5rem;border-bottom:1px solid var(--border);
background:linear-gradient(90deg,color-mix(in srgb,var(--accent) 13%,transparent),transparent)}
.ids-rhead .t{font:800 .85rem 'Orbitron',sans-serif;letter-spacing:3px;color:var(--text)}
.ids-rhead .m{font:500 .72rem 'JetBrains Mono',monospace;color:var(--muted)}
.ids-rbody{padding:1.4rem 1.5rem;display:grid;gap:1.7rem}
.ids-grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
.ids-evcol{--tone:var(--accent);border:1px solid var(--border);border-radius:16px;padding:1rem;background:color-mix(in srgb,var(--bg) 35%,transparent)}
.ids-evcol.for{--tone:var(--accent)}
.ids-evcol.against{--tone:var(--warn)}
.ids-evcol.lime{--tone:var(--accent2)}
.ids-evtitle{display:flex;align-items:center;gap:.5rem;font:700 .8rem 'Inter',sans-serif;margin-bottom:.8rem;color:var(--tone)}
.ids-count{margin-left:auto;font:600 .68rem 'JetBrains Mono',monospace;padding:.05rem .5rem;border-radius:99px;background:color-mix(in srgb,var(--tone) 16%,transparent)}
.ids-ev{padding:.7rem .85rem;border-radius:12px;border:1px solid var(--border);background:var(--card);margin-bottom:.6rem}
.ids-ev:last-child{margin-bottom:0}
.ids-evtop{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;flex-wrap:wrap}
.ids-evname{font:600 .84rem 'JetBrains Mono',monospace;color:var(--text);word-break:break-word}
.ids-evval{font:500 .74rem 'JetBrains Mono',monospace;color:var(--muted)}
.ids-evbar{height:7px;border-radius:99px;background:var(--track);margin:.5rem 0 .4rem;overflow:hidden}
.ids-evbar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--tone),color-mix(in srgb,var(--tone) 55%,transparent))}
.ids-evnums{display:flex;gap:.4rem;flex-wrap:wrap}
.ids-num{font:600 .72rem 'JetBrains Mono',monospace;padding:.12rem .5rem;border-radius:6px;background:color-mix(in srgb,var(--tone) 14%,transparent);color:var(--tone)}
.ids-num.ids-dir{background:transparent;color:var(--muted);border:1px solid var(--border);font-weight:500}
.ids-evnote{margin-top:.45rem;font-size:.83rem;color:var(--muted);line-height:1.5}
.ids-prose .ids-para{line-height:1.75;margin:0 0 .8rem;color:var(--text);font-size:.97rem}
.ids-prose ul{margin:.2rem 0 .8rem 1.1rem;padding:0}
.ids-prose li{margin:.3rem 0;line-height:1.6;color:var(--text) !important}
.ids-code{font:500 .85em 'JetBrains Mono',monospace;padding:.08rem .38rem;border-radius:6px;background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)}
.ids-action{display:flex;gap:1rem;padding:1.1rem 1.3rem;border-radius:16px;border:1px solid var(--warn);background:linear-gradient(135deg,color-mix(in srgb,var(--warn) 13%,transparent),transparent 70%)}
.ids-action .ic{font-size:1.6rem;line-height:1.2}
.ids-actionbody{flex:1}
.ids-todo{display:grid;gap:.65rem}
.ids-todoitem{display:flex;gap:.75rem;align-items:flex-start;line-height:1.6;color:var(--text)}
.ids-todoitem .n{flex:none;width:24px;height:24px;border-radius:50%;display:grid;place-items:center;font:700 .72rem 'JetBrains Mono',monospace;background:var(--warn);color:#1b1200}
.ids-flag{padding:.75rem 1rem;border-radius:12px;border:1px dashed var(--warn);color:var(--text);background:color-mix(in srgb,var(--warn) 10%,transparent);font-size:.87rem;line-height:1.5}
.ids-foot{font-size:.76rem;color:var(--muted);border-top:1px dashed var(--border);padding-top:.9rem;line-height:1.5}

/* ---------- SHAP / LIME bars ---------- */
.ids-legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.76rem;color:var(--muted);margin-bottom:.9rem}
.ids-legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:.4rem;background:var(--lc)}
.ids-dv{display:grid;gap:.4rem}
.ids-dvrow{display:grid;grid-template-columns:minmax(130px,1.3fr) minmax(70px,.6fr) 2.2fr 84px;gap:.8rem;align-items:center;padding:.4rem .2rem;border-bottom:1px dashed color-mix(in srgb,var(--border) 70%,transparent)}
.ids-dvrow.nov{grid-template-columns:minmax(150px,1.7fr) 2.2fr 84px}
.ids-dvname{font:500 .8rem 'JetBrains Mono',monospace;word-break:break-word;color:var(--text)}
.ids-dvval{font:500 .74rem 'JetBrains Mono',monospace;color:var(--muted);text-align:right}
.ids-dvtrack{position:relative;height:12px;border-radius:99px;background:var(--track)}
.ids-dvtrack::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--muted);opacity:.55}
.ids-dvtrack i{position:absolute;top:0;bottom:0;border-radius:99px}
.ids-dvnum{font:700 .78rem 'JetBrains Mono',monospace;text-align:right}
@media (max-width:700px){.ids-dvrow,.ids-dvrow.nov{grid-template-columns:1fr 1fr}.ids-dvtrack{grid-column:1 / -1}}

/* ---------- tables / json ---------- */
.ids-tablewrap{overflow:auto;border:1px solid var(--border);border-radius:14px}
.ids-table{width:100%;border-collapse:collapse;font:500 .8rem 'JetBrains Mono',monospace}
.ids-table th{position:sticky;top:0;background:var(--bg2);color:var(--accent);text-align:left;padding:.55rem .8rem;font-size:.7rem;letter-spacing:1.5px;text-transform:uppercase}
.ids-table td{padding:.45rem .8rem;border-top:1px solid color-mix(in srgb,var(--border) 60%,transparent);color:var(--text);word-break:break-word}
.ids-table tr:hover td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
.ids-pre{max-height:420px;overflow:auto;white-space:pre;font:400 .78rem/1.55 'JetBrains Mono',monospace;color:var(--text);background:var(--input-bg);border:1px solid var(--border);border-radius:14px;padding:1rem 1.2rem}

/* ---------- sidebar bits ---------- */
.ids-brand{display:flex;align-items:center;gap:.75rem;padding:.3rem 0 .6rem}
.ids-logo{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;font-size:1.3rem;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 0 22px color-mix(in srgb,var(--accent) 50%,transparent)}
.ids-brand .n{font:800 1rem 'Orbitron',sans-serif;letter-spacing:1px}
.ids-brand .s{font:500 .64rem 'JetBrains Mono',monospace;color:var(--muted);letter-spacing:1.5px}
.ids-sidelabel{font:700 .66rem 'JetBrains Mono',monospace;letter-spacing:2.5px;color:var(--muted) !important;text-transform:uppercase;margin:1rem 0 .45rem}
.ids-pill{display:flex;justify-content:space-between;align-items:center;gap:.5rem;padding:.45rem .7rem;border-radius:10px;border:1px solid var(--border);background:var(--card);font-size:.78rem;margin-bottom:.4rem}
.ids-pill b{font:700 .66rem 'JetBrains Mono',monospace;letter-spacing:1px;padding:.1rem .45rem;border-radius:6px;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ids-pill b.ok{color:var(--safe) !important;background:color-mix(in srgb,var(--safe) 15%,transparent)}
.ids-pill b.off{color:var(--danger) !important;background:color-mix(in srgb,var(--danger) 15%,transparent)}
.ids-pill b.info{color:var(--accent) !important;background:color-mix(in srgb,var(--accent) 15%,transparent)}
.ids-kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.6rem}
.ids-kvi{padding:.6rem .8rem;border:1px solid var(--border);border-radius:12px;background:var(--card)}
.ids-kvi span{display:block;font:600 .64rem 'JetBrains Mono',monospace;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:.2rem}
.ids-kvi div{font:500 .82rem 'JetBrains Mono',monospace;word-break:break-all;color:var(--text)}
.ids-footer{text-align:center;color:var(--muted);font:500 .72rem 'JetBrains Mono',monospace;letter-spacing:2px;padding:.4rem 0 1rem}
"""


def theme_css(theme):
    """Return a <style> block for the chosen colour theme."""

    variables = "".join(
        "--" + key.replace("_", "-") + ":" + value + ";"
        for key, value in theme.items()
    )

    return (
        "<style>"
        + FONT_IMPORT
        + ":root{" + variables + "}"
        + BASE_CSS
        + "</style>"
    )


# ============================================================
# HTML HELPERS
# ============================================================

def _html(markup):
    """Strip indentation / blank lines so Markdown never turns the
    HTML into a code block."""
    return "\n".join(
        line.strip()
        for line in str(markup).splitlines()
        if line.strip()
    )


def ui(markup):
    st.markdown(_html(markup), unsafe_allow_html=True)


def side_ui(markup):
    st.sidebar.markdown(_html(markup), unsafe_allow_html=True)


def _e(value):
    """HTML-escape any value (LLM text is never trusted)."""
    return html.escape(str(value), quote=True)


def _md_inline(text):
    """Escape text, then allow **bold**, `code` and line breaks."""
    out = _e(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`([^`]+)`", r'<span class="ids-code">\1</span>', out)
    return out.replace("\n", "<br>")


def _to_float(value, default=None):
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value, signed=False):
    number = _to_float(value)

    if number is None:
        return str(value)

    if signed:
        return f"{number:+.4f}"

    return format(number, ".10g")


def _confidence_pct(value):
    """Accept 0.93, 93, '93%' ... and return a 0-100 percentage."""
    number = _to_float(value, 0.0)

    if number <= 1.0:
        number = number * 100.0

    return max(0.0, min(100.0, number))


def _flat(value):
    if value is None:
        return ""

    if isinstance(value, dict):
        return "; ".join(
            f"{k}: {_flat(v)}" for k, v in value.items()
        )

    if isinstance(value, (list, tuple)):
        return ", ".join(_flat(v) for v in value)

    return str(value)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


_NAME_KEYS = ("feature", "feature_name", "name", "condition", "rule")
_VALUE_KEYS = ("feature_value", "value", "observed_value", "input_value")
_NUM_KEYS = (
    "shap_value", "shap", "lime_value", "lime",
    "contribution", "contribution_value", "weight",
    "importance", "impact"
)
_TEXT_KEYS = (
    "explanation", "description", "reason", "note",
    "comment", "summary", "interpretation", "meaning"
)


def _first_key(item, keys):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return key
    return None


def _num_label(key):
    key = key.lower()

    if "shap" in key:
        return "SHAP"

    if "lime" in key:
        return "LIME"

    return "Contribution"


def _as_list(value):
    """The LLM should return lists, but be tolerant of anything else."""

    if value is None or value == "":
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):

        if _first_key(value, _NAME_KEYS):
            return [value]

        items = []

        for key, val in value.items():

            if _to_float(val) is not None and not isinstance(val, bool):
                items.append({"feature": key, "contribution": val})

            else:
                items.append({"feature": key, "explanation": _flat(val)})

        return items

    return [value]


def _primary_number(item):
    if not isinstance(item, dict):
        return None

    for key in _NUM_KEYS:
        if key in item:
            number = _to_float(item[key])

            if number is not None:
                return number

    return None


def _group_scale(items):
    values = [
        abs(n)
        for n in (_primary_number(i) for i in items)
        if n is not None
    ]

    return max(values) if values else 0.0


# ------------------------------------------------------------
# Pipeline tracker
# ------------------------------------------------------------

PIPELINE_STEPS = [
    ("📥", "Input"),
    ("🧠", "Model"),
    ("🎯", "Prediction"),
    ("🔎", "SHAP"),
    ("🔬", "LIME"),
    ("🤖", "LLM Report"),
]


def pipeline_html(done=None, active=None, skipped=(), failed=None):
    """done=None renders the neutral 'how it works' strip."""

    parts = []

    for i, (icon, label) in enumerate(PIPELINE_STEPS):

        state = ""

        if done is not None:

            if failed == i:
                state = "fail"

            elif i in skipped:
                state = "skip"

            elif i < done:
                state = "done"

            elif i == active:
                state = "active"

        mark = "✓ " if state == "done" else ""

        parts.append(
            f'<span class="ids-step {state}">{icon} {mark}{label}</span>'
        )

        if i < len(PIPELINE_STEPS) - 1:
            parts.append('<span class="ids-arrow">›</span>')

    return '<div class="ids-steps">' + "".join(parts) + "</div>"


def show_pipeline(slot, done, active=None, skipped=(), failed=None):
    slot.markdown(
        _html(pipeline_html(done, active, skipped, failed)),
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Hero / brand / small pieces
# ------------------------------------------------------------

def hero_html():
    return (
        '<div class="ids-hero">'
        '<span class="ids-badge"><span class="ids-dot"></span>'
        'SYSTEM ONLINE · X-AI IDS</span>'
        '<div class="ids-title">Adaptive <b>Explainable</b> '
        'Intrusion Detection System</div>'
        '<div class="ids-sub">Live network-flow classification with SHAP '
        '&amp; LIME explainability and an LLM-written threat assessment. '
        'Choose a dataset, tune the flow features and let the pipeline '
        'explain every verdict.</div>'
        '</div>'
    )


def brand_html():
    return (
        '<div class="ids-brand"><div class="ids-logo">🛡️</div>'
        '<div><div class="n">Adaptive IDS</div>'
        '<div class="s">EXPLAINABLE · LIVE</div></div></div>'
    )


def status_pill(label, text, kind):
    return (
        f'<div class="ids-pill"><span>{_e(label)}</span>'
        f'<b class="{kind}">{_e(text)}</b></div>'
    )


def kv_html(pairs):
    items = "".join(
        f'<div class="ids-kvi"><span>{_e(k)}</span><div>{_e(v)}</div></div>'
        for k, v in pairs
    )

    return f'<div class="ids-kv">{items}</div>'


def _cell(value):
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".10g")

    return str(value)


def table_html(df, max_height=380):
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)

    rows = "".join(
        "<tr>"
        + "".join(f"<td>{_e(_cell(v))}</td>" for v in row)
        + "</tr>"
        for row in df.itertuples(index=False, name=None)
    )

    return (
        f'<div class="ids-tablewrap" style="max-height:{int(max_height)}px">'
        f'<table class="ids-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def json_pre_html(obj):
    text = html.escape(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    )

    return '<div class="ids-pre">' + text.replace("\n", "&#10;") + "</div>"


# ------------------------------------------------------------
# Live prediction cards
# ------------------------------------------------------------

def verdict_html(tone, label, pct, eyebrow, sub, chips):
    chips_html = "".join(
        f'<span class="ids-chip"><i>{_e(k)}</i>{_e(v)}</span>'
        for k, v in chips
    )

    return (
        f'<div class="ids-verdict {tone}">'
        f'<div class="ids-ring" style="--pct:{pct:.2f}">'
        f'<div class="ids-ring-in"><b>{pct:.1f}%</b>'
        f'<span>confidence</span></div></div>'
        f'<div class="ids-vmain">'
        f'<div class="ids-eyebrow">{_e(eyebrow)}</div>'
        f'<div class="ids-vlabel">{_e(label)}</div>'
        f'<div class="ids-vsub">{sub}</div>'
        f'<div class="ids-chips">{chips_html}</div>'
        f'</div></div>'
    )


def prob_html(p0, p1):

    def row(name, p, color):
        p = max(0.0, min(1.0, float(p)))
        return (
            f'<div class="ids-prow"><span>{name}</span>'
            f'<div class="ids-pbar"><i style="width:{p * 100:.2f}%;'
            f'background:{color}"></i></div>'
            f'<b>{p * 100:.4f}%</b></div>'
        )

    return (
        '<div class="ids-card"><div class="ids-sec">Class probabilities</div>'
        '<div class="ids-prob">'
        + row("Normal", p0, "var(--safe)")
        + row("Malicious", p1, "var(--danger)")
        + "</div></div>"
    )


def diverging_html(rows, pos_color, neg_color, pos_text, neg_text,
                   show_value=True):
    """rows = [(name, value_or_None, contribution)]"""

    if not rows:
        return '<div class="ids-empty">No contributions available.</div>'

    max_abs = max(abs(n) for _, _, n in rows) or 1.0

    legend = (
        f'<div class="ids-legend">'
        f'<span style="--lc:{pos_color}">{_e(pos_text)}</span>'
        f'<span style="--lc:{neg_color}">{_e(neg_text)}</span></div>'
    )

    row_class = "ids-dvrow" if show_value else "ids-dvrow nov"

    out = []

    for name, value, number in rows:

        width = abs(number) / max_abs * 50

        if number >= 0:
            bar = (
                f'<i style="left:50%;width:{width:.2f}%;'
                f'background:{pos_color}"></i>'
            )
            color = pos_color

        else:
            bar = (
                f'<i style="right:50%;width:{width:.2f}%;'
                f'background:{neg_color}"></i>'
            )
            color = neg_color

        value_cell = (
            f'<div class="ids-dvval">{_e(_fmt(value))}</div>'
            if show_value else ""
        )

        out.append(
            f'<div class="{row_class}">'
            f'<div class="ids-dvname">{_e(name)}</div>{value_cell}'
            f'<div class="ids-dvtrack">{bar}</div>'
            f'<div class="ids-dvnum" style="color:{color}">'
            f'{number:+.4f}</div></div>'
        )

    return legend + '<div class="ids-dv">' + "".join(out) + "</div>"


# ------------------------------------------------------------
# LLM threat-assessment report (readable format)
# ------------------------------------------------------------

def _paragraphs(text):
    text = str(text).strip()

    if not text:
        return []

    if "\n" in text:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    if len(text) <= 380:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z`\"'(])", text)

    paragraphs, current = [], ""

    for sentence in sentences:

        current = (current + " " + sentence).strip()

        if len(current) > 280:
            paragraphs.append(current)
            current = ""

    if current:
        paragraphs.append(current)

    return paragraphs


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def _as_steps(value):
    """Return a list of steps when the value is list-like, else None."""

    if isinstance(value, (list, tuple)):
        return [_flat(v) for v in value if _flat(v)]

    if isinstance(value, str) and "\n" in value:

        lines = [ln for ln in value.splitlines() if ln.strip()]
        matches = [_BULLET.match(ln) for ln in lines]

        if len(lines) >= 2 and all(matches):
            return [m.group(1) for m in matches]

    return None


def prose_html(value):

    if value is None or value == "" or value == [] or value == {}:
        return '<div class="ids-empty">Not provided.</div>'

    if isinstance(value, dict):

        rows = "".join(
            f'<div class="ids-para"><b>'
            f'{_e(str(k).replace("_", " ").title())}:</b> '
            f'{_md_inline(_flat(v))}</div>'
            for k, v in value.items()
        )

        return f'<div class="ids-prose">{rows}</div>'

    if isinstance(value, (list, tuple)):

        items = "".join(
            f"<li>{_md_inline(_flat(v))}</li>" for v in value
        )

        return f'<div class="ids-prose"><ul>{items}</ul></div>'

    parts, bullets = [], []

    def flush():
        if bullets:
            parts.append(
                "<ul>"
                + "".join(f"<li>{_md_inline(b)}</li>" for b in bullets)
                + "</ul>"
            )
            bullets.clear()

    for line in _paragraphs(value):

        match = _BULLET.match(line)

        if match:
            bullets.append(match.group(1))

        else:
            flush()
            parts.append(f'<div class="ids-para">{_md_inline(line)}</div>')

    flush()

    return '<div class="ids-prose">' + "".join(parts) + "</div>"


def action_html(value):

    steps = _as_steps(value)

    if steps:

        body = '<div class="ids-todo">' + "".join(
            f'<div class="ids-todoitem"><span class="n">{i}</span>'
            f'<div>{_md_inline(s)}</div></div>'
            for i, s in enumerate(steps, start=1)
        ) + "</div>"

    else:
        body = prose_html(value)

    return (
        '<div class="ids-action"><div class="ic">🛡️</div>'
        f'<div class="ids-actionbody">{body}</div></div>'
    )


def evidence_item_html(item, scale, by_sign=False):

    if not isinstance(item, dict):
        return (
            '<div class="ids-ev"><div class="ids-evnote" style="margin:0">'
            f'{_md_inline(_flat(item))}</div></div>'
        )

    name_key = _first_key(item, _NAME_KEYS)
    name = _flat(item[name_key]) if name_key else "Unnamed evidence"

    value_key = _first_key(item, _VALUE_KEYS)

    value_html = (
        f'<span class="ids-evval">= {_e(_fmt(item[value_key]))}</span>'
        if value_key else ""
    )

    numbers = []

    for key in _NUM_KEYS:

        if key in item:

            number = _to_float(item[key])

            if number is not None:
                numbers.append((_num_label(key), number))

    primary = numbers[0][1] if numbers else None

    style = ""

    if by_sign and primary is not None:
        tone = "var(--accent)" if primary >= 0 else "var(--warn)"
        style = f' style="--tone:{tone}"'

    bar = ""

    if primary is not None and scale > 0:
        width = max(3.0, min(100.0, abs(primary) / scale * 100))
        bar = (
            f'<div class="ids-evbar"><i style="width:{width:.1f}%"></i></div>'
        )

    chips = "".join(
        f'<span class="ids-num">{label} {number:+.4f}</span>'
        for label, number in numbers
    )

    direction = item.get("direction")

    if isinstance(direction, str) and direction:
        pretty = direction.replace("_", " ")
        chips += f'<span class="ids-num ids-dir">{_e(pretty)}</span>'

    notes = [
        _md_inline(_flat(item[k]))
        for k in _TEXT_KEYS
        if k in item and item[k] not in (None, "")
    ]

    note_html = (
        '<div class="ids-evnote">' + "<br>".join(notes) + "</div>"
        if notes else ""
    )

    return (
        f'<div class="ids-ev"{style}>'
        f'<div class="ids-evtop"><span class="ids-evname">{_e(name)}</span>'
        f'{value_html}</div>{bar}'
        f'<div class="ids-evnums">{chips}</div>{note_html}</div>'
    )


def evidence_col_html(title, icon, items, kind, scale, by_sign=False):

    if items:
        body = "".join(
            evidence_item_html(i, scale, by_sign) for i in items
        )

    else:
        body = '<div class="ids-empty">No evidence returned for this group.</div>'

    return (
        f'<div class="ids-evcol {kind}">'
        f'<div class="ids-evtitle">{icon} {_e(title)}'
        f'<span class="ids-count">{len(items)}</span></div>{body}</div>'
    )


def _section(title, inner):
    return f'<div><div class="ids-sec">{_e(title)}</div>{inner}</div>'


def llm_report_html(llm_result, reference_label=None):
    """Turn the Gemini JSON into a readable threat-assessment report."""

    threat = _as_dict(llm_result.get("threat_assessment"))
    model_info = _as_dict(llm_result.get("model_information"))
    evidence = _as_dict(llm_result.get("key_xai_evidence"))

    interpretation = llm_result.get("interpretation", "")
    recommended_action = llm_result.get("recommended_action", "")

    label = _flat(threat.get("prediction", "Unknown")) or "Unknown"
    lowered = label.lower()

    if "malicious" in lowered:
        tone, icon = "bad", "⚠"
        sub = (
            "The AI analyst assessed this flow as <b>malicious</b> based on "
            "the supplied SHAP and LIME evidence."
        )

    elif "normal" in lowered:
        tone, icon = "ok", "✓"
        sub = (
            "The AI analyst assessed this flow as <b>normal</b> based on "
            "the supplied SHAP and LIME evidence."
        )

    else:
        tone, icon = "unk", "?"
        sub = "The verdict returned by the LLM could not be interpreted."

    pct = _confidence_pct(threat.get("confidence", 0.0))

    chips = [
        ("dataset", _flat(model_info.get("dataset", "Unknown"))),
        ("model", _flat(model_info.get("selected_model", "Unknown"))),
    ]

    verdict = verdict_html(
        tone,
        f"{icon} {label.upper()}",
        pct,
        "Threat assessment",
        sub,
        chips
    )

    flag = ""

    if reference_label and reference_label.lower() not in lowered:
        flag = (
            '<div class="ids-flag">⚠ The LLM-reported verdict '
            f'(<b>{_e(label)}</b>) differs from the live model prediction '
            f'(<b>{_e(reference_label)}</b>). Treat the live model output '
            'as authoritative and review the raw JSON below.</div>'
        )

    towards = _as_list(evidence.get("towards_prediction"))
    against = _as_list(evidence.get("against_prediction"))
    lime_items = _as_list(evidence.get("lime"))

    main_scale = max(_group_scale(towards), _group_scale(against))

    columns = (
        evidence_col_html(
            "Evidence supporting the prediction", "➡️",
            towards, "for", main_scale
        )
        + evidence_col_html(
            "Evidence against the prediction", "⬅️",
            against, "against", main_scale
        )
    )

    lime_block = ""

    if lime_items:
        lime_block = (
            '<div style="margin-top:1rem">'
            + evidence_col_html(
                "LIME local evidence", "🔬",
                lime_items, "lime", _group_scale(lime_items), by_sign=True
            )
            + "</div>"
        )

    stamp = datetime.now().strftime("%d %b %Y · %H:%M")

    head = (
        '<div class="ids-rhead"><span class="t">THREAT ASSESSMENT REPORT</span>'
        f'<span class="m">Generated {stamp} · Analyst: Gemini '
        f'({_e(LLM_MODEL)})</span></div>'
    )

    body = (
        flag
        + verdict
        + _section(
            "🔎 Key XAI evidence",
            f'<div class="ids-grid2">{columns}</div>' + lime_block
        )
        + _section("🧠 Interpretation", prose_html(interpretation))
        + _section("🛡️ Recommended action", action_html(recommended_action))
        + '<div class="ids-foot">AI-generated from the supplied SHAP / LIME '
        'evidence. A model prediction is not proof of an attack — validate '
        'with additional telemetry before taking any action.</div>'
    )

    return f'<div class="ids-report">{head}<div class="ids-rbody">{body}</div></div>'


# ============================================================
# THEME SELECTOR + HERO HEADER
# ============================================================

side_ui(brand_html())

side_ui('<div class="ids-sidelabel">🎨 Interface theme</div>')

theme_name = st.sidebar.selectbox(

    "Interface theme",

    list(THEMES.keys()),

    index=0,

    key="ids_theme",

    label_visibility="collapsed"
)

ui(theme_css(THEMES[theme_name]))

ui(hero_html())



# ============================================================
# LOAD ENVIRONMENT VARIABLES AND STREAMLIT SECRETS
# ============================================================

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load local .env if it exists; do not overwrite existing variables.
load_dotenv(dotenv_path=ENV_PATH, override=False)


def get_setting(name, default=""):
    """Read Streamlit Secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if value:
        return str(value).strip()

    return str(os.getenv(name, default) or default).strip()


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
LLM_MODEL = get_setting("GEMINI_MODEL", "gemini-2.5-flash")


# Safe status indicator — never display the API key itself.
side_ui(
    '<div class="ids-sidelabel">System status</div>'
    + status_pill(
        "Gemini API key",
        "LOADED" if GEMINI_API_KEY else "MISSING",
        "ok" if GEMINI_API_KEY else "off"
    )
    + status_pill("LLM model", LLM_MODEL, "info")
)

if not GEMINI_API_KEY:
    st.warning(
        "Gemini LLM explanations are disabled because "
        "GEMINI_API_KEY was not found. Add it to "
        ".streamlit/secrets.toml or the local .env file."
    )


# ============================================================
# DEPLOYMENT PATHS (raw datasets are not required)
# ============================================================

PROJECT_ROOT = BASE_DIR
SAVED_MODELS_DIR = PROJECT_ROOT / "models"
ADAPTIVE_OUTPUT_DIR = PROJECT_ROOT / "adaptive_outputs"
XAI_OUTPUT_DIR = PROJECT_ROOT / "xai_outputs"
BACKGROUND_DIR = PROJECT_ROOT / "backgrounds"
SCHEMA_DIR = PROJECT_ROOT / "schemas"

for _directory in (XAI_OUTPUT_DIR, ADAPTIVE_OUTPUT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH = PROJECT_ROOT / "model_registry.json"
with REGISTRY_PATH.open("r", encoding="utf-8") as _f:
    MODEL_REGISTRY = json.load(_f)

DATASET_CONFIG = {
    "NSL-KDD": {
        "target": "binary_attack",
        "xgb_model": SAVED_MODELS_DIR / "NSL-KDD_XGBoost.joblib",
        "dnn_model": SAVED_MODELS_DIR / "NSL-KDD_DNN.keras",
    },
    "Farm-Flow": {
        "target": "is_attack",
        "xgb_model": SAVED_MODELS_DIR / "Farm-Flow_XGBoost.joblib",
        "dnn_model": SAVED_MODELS_DIR / "Farm-Flow_DNN.keras",
    },
    "IoMT-AI-IDS": {
        "target": "is_attack",
        "xgb_model": SAVED_MODELS_DIR / "IoMT_AI_IDS_XGBoost.joblib",
        "dnn_model": SAVED_MODELS_DIR / "IoMT_AI_IDS_DNN.keras",
    },
}

CLASS_MAPPING = {0: "Normal", 1: "Malicious"}

# ============================================================
# ADAPTIVE MODEL SELECTION
# ============================================================

SELECTION_PATH = ADAPTIVE_OUTPUT_DIR / "final_model_selection.json"


def load_adaptive_selection():

    if not SELECTION_PATH.exists():

        st.warning(
            "Adaptive model selection file was not found. "
            "You can manually select the model."
        )

        return {}

    try:

        with open(
            SELECTION_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        st.error(
            f"Could not read adaptive model selection: {e}"
        )

        return {}


selection_registry = load_adaptive_selection()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_xgb_model(model_path):

    import joblib

    return joblib.load(model_path)


@st.cache_resource
def load_dnn_model(model_path):

    import tensorflow as tf

    return tf.keras.models.load_model(
        model_path
    )


def load_model(dataset_name, model_name):

    config = DATASET_CONFIG[
        dataset_name
    ]

    if model_name == "XGBoost":

        model_path = config["xgb_model"]

        if not model_path.exists():

            raise FileNotFoundError(
                f"XGBoost model not found:\n{model_path}"
            )

        return load_xgb_model(
            str(model_path)
        )

    elif model_name == "DNN":

        model_path = config["dnn_model"]

        if not model_path.exists():

            raise FileNotFoundError(
                f"DNN model not found:\n{model_path}"
            )

        return load_dnn_model(
            str(model_path)
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ============================================================
# LOAD SMALL EXPLAINABILITY BACKGROUND (not raw dataset)
# ============================================================

@st.cache_data
def load_training_data(dataset_name):
    """Load a small feature-only background sample created locally from training data."""
    background_path = BACKGROUND_DIR / f"{dataset_name}_background.csv"
    if not background_path.exists():
        raise FileNotFoundError(
            f"Explainability background is missing: {background_path}. "
            "Run prepare_backgrounds.py in your original project before deployment."
        )
    df = pd.read_csv(background_path)
    expected = MODEL_REGISTRY[dataset_name]["features"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Background file for {dataset_name} is missing features: {missing}")
    return df[expected].apply(pd.to_numeric, errors="coerce").fillna(0.0)


@st.cache_data
def load_test_data(dataset_name):
    """Raw test sets are intentionally not shipped in the deployment."""
    raise FileNotFoundError("Raw test datasets are excluded from this deployment.")


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(
    model,
    model_name,
    input_df
):

    if model_name == "XGBoost":

        probabilities = (
            model.predict_proba(
                input_df
            )[0]
        )

        probability_class_0 = float(
            probabilities[0]
        )

        probability_class_1 = float(
            probabilities[1]
        )

    else:

        probability_class_1 = float(
            np.asarray(
                model.predict(
                    input_df,
                    verbose=0
                )
            ).reshape(-1)[0]
        )

        probability_class_0 = (
            1.0 -
            probability_class_1
        )

    predicted_class = int(
        np.argmax(
            [
                probability_class_0,
                probability_class_1
            ]
        )
    )

    confidence = max(
        probability_class_0,
        probability_class_1
    )

    return {

        "class":
            predicted_class,

        "label":
            CLASS_MAPPING[
                predicted_class
            ],

        "probability_class_0":
            probability_class_0,

        "probability_class_1":
            probability_class_1,

        "confidence":
            confidence
    }


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(
    model,
    model_name,
    input_df,
    train_df,
    feature_names
):

    import shap

    X_train = train_df[
        feature_names
    ]

        if model_name == "XGBoost":

        try:
            # Preferred: fast tree-based SHAP
            explainer = shap.TreeExplainer(model)
            explanation = explainer(input_df)

            values = np.asarray(
                explanation.values
            ).reshape(1, -1)[0]

            base_value = float(
                np.asarray(
                    explanation.base_values
                ).reshape(-1)[0]
            )

        except Exception as tree_error:
            # Fallback: avoids SHAP's XGBoost tree-model parser
            st.warning(
                "TreeSHAP is incompatible with this model format. "
                "Using model-agnostic SHAP instead."
            )

            background_size = min(30, len(X_train))
            background = X_train.sample(
                background_size,
                random_state=42
            ).astype(float)

            def xgb_predict_probability(X):
                X_df = pd.DataFrame(
                    X,
                    columns=feature_names
                )
                return model.predict_proba(X_df)[:, 1]

            explainer = shap.KernelExplainer(
                xgb_predict_probability,
                background.to_numpy()
            )

            shap_values = explainer.shap_values(
                input_df[feature_names].to_numpy(),
                nsamples=100
            )

            values = np.asarray(
                shap_values
            ).reshape(1, -1)[0]

            base_value = float(
                np.asarray(
                    explainer.expected_value
                ).reshape(-1)[0]
            )

        return {
            "base_value": base_value,
            "values": values
        }

    else:

        # ----------------------------------------------------
        # DNN background data
        # ----------------------------------------------------

        background_size = min(
            50,
            len(X_train)
        )

        background = X_train.sample(
            background_size,
            random_state=42
        )

        # KernelExplainer is slower but reliable
        # for arbitrary DNN models.

        def dnn_predict(X):

            p1 = (
                model.predict(
                    X,
                    verbose=0
                ).reshape(-1)
            )

            p0 = 1.0 - p1

            return np.column_stack(
                [
                    p0,
                    p1
                ]
            )

        explainer = shap.KernelExplainer(
            dnn_predict,
            background
        )

        shap_values = explainer.shap_values(
            input_df,
            nsamples=100
        )

        # Binary classification
        # We explain class 1.

        if isinstance(
            shap_values,
            list
        ):

            values = np.asarray(
                shap_values[1]
            )

        else:

            values = np.asarray(
                shap_values
            )

        values = values.reshape(
            1,
            -1
        )

        base_value = (
            float(
                np.asarray(
                    explainer.expected_value
                ).reshape(-1)[-1]
            )
        )

        return {

            "base_value":
                base_value,

            "values":
                values[0]
        }


    values = np.asarray(
        explanation.values
    )

    values = values.reshape(
        1,
        -1
    )[0]

    base_value = float(
        np.asarray(
            explanation.base_values
        ).reshape(-1)[0]
    )

    return {

        "base_value":
            base_value,

        "values":
            values
    }


# ============================================================
# LIME EXPLANATION
# ============================================================

def generate_lime_explanation(
    model,
    model_name,
    input_df,
    train_df,
    feature_names,
    predicted_class
):

    from lime.lime_tabular import (
        LimeTabularExplainer
    )

    X_train = train_df[
        feature_names
    ]

    if model_name == "DNN":

        def predict_function(X):

            p1 = (
                model.predict(
                    X,
                    verbose=0
                ).reshape(-1)
            )

            p0 = 1.0 - p1

            return np.column_stack(
                [
                    p0,
                    p1
                ]
            )

    else:

        def predict_function(X):

            return model.predict_proba(
                X
            )


    explainer = LimeTabularExplainer(

        training_data=
            X_train.values,

        feature_names=
            feature_names,

        class_names=[
            "Normal",
            "Malicious"
        ],

        mode="classification",

        random_state=42
    )


    explanation = (
        explainer.explain_instance(

            input_df.iloc[0].values,

            predict_function,

            num_features=10,

            top_labels=2
        )
    )


    return explanation.as_list(
        label=predicted_class
    )


# ============================================================
# BUILD STRUCTURED XAI OUTPUT
# ============================================================

def build_xai_json(
    dataset_name,
    model_name,
    prediction,
    input_df,
    shap_result,
    lime_result,
    feature_names
):

    predicted_class = prediction[
        "class"
    ]

    feature_values = (
        input_df.iloc[0].values
    )

    shap_values = (
        shap_result["values"]
    )

    shap_features = []

    for feature, value, shap_value in zip(
        feature_names,
        feature_values,
        shap_values
    ):

        shap_value = float(
            shap_value
        )

        if shap_value > 0:

            contribution_class = 1

        elif shap_value < 0:

            contribution_class = 0

        else:

            contribution_class = None


        if shap_value == 0:

            direction = (
                "no_contribution"
            )

        elif (
            contribution_class ==
            predicted_class
        ):

            direction = (
                "towards_predicted_class"
            )

        else:

            direction = (
                "away_from_predicted_class"
            )


        shap_features.append({

            "feature":
                feature,

            "feature_value":
                float(value),

            "shap_value":
                shap_value,

            "contribution_class":
                contribution_class,

            "contribution_label":
                (
                    CLASS_MAPPING[
                        contribution_class
                    ]
                    if contribution_class is not None
                    else "None"
                ),

            "direction":
                direction
        })


    shap_features.sort(
        key=lambda x:
            abs(x["shap_value"]),
        reverse=True
    )


    lime_features = []

    for feature, contribution in (
        lime_result
    ):

        contribution = float(
            contribution
        )

        if contribution > 0:

            direction = (
                "towards_predicted_class"
            )

        elif contribution < 0:

            direction = (
                "away_from_predicted_class"
            )

        else:

            direction = (
                "no_contribution"
            )


        lime_features.append({

            "feature":
                feature,

            "contribution":
                contribution,

            "direction":
                direction
        })


    return {

        "input_type":
            "custom_live_input",

        "dataset":
            dataset_name,

        "selected_model":
            model_name,

        "prediction":
            prediction,

        "actual_class":
            None,

        "actual_label":
            None,

        "class_mapping": {

            "0":
                "Normal",

            "1":
                "Malicious"
        },

        "explanations": {

            "SHAP": {

                "method":
                    "SHAP",

                "type":
                    "local",

                "base_value":
                    shap_result[
                        "base_value"
                    ],

                "feature_contributions":
                    shap_features
            },

            "LIME": {

                "method":
                    "LIME",

                "type":
                    "local",

                "explained_class":
                    predicted_class,

                "explained_label":
                    CLASS_MAPPING[
                        predicted_class
                    ],

                "feature_contributions":
                    lime_features
            }
        }
    }


# ============================================================
# SAVE XAI OUTPUT
# ============================================================

def save_xai_output(
    output
):

    dataset_name = (
        output["dataset"]
    )

    model_name = (
        output["selected_model"]
    )

    filename = (
        f"{dataset_name}_"
        f"{model_name}_"
        f"custom_live_explanation.json"
    )

    output_path = (
        XAI_OUTPUT_DIR /
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    return output_path


# ============================================================
# GEMINI LLM EXPLANATION
# ============================================================

def _extract_json_object(text):
    """Extract a JSON object from Gemini output.

    Gemini may return JSON directly, wrap it in ```json ... ``` fences,
    or occasionally add a short sentence before the JSON.
    """
    if text is None:
        raise ValueError("Gemini returned no text.")

    text = str(text).strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")

    # Remove Markdown code fences if present.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # First try the complete response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If Gemini added prose, isolate the outer JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Gemini response did not contain a JSON object. "
            f"Raw response: {text[:500]}"
        )

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned text that could not be parsed as JSON. "
            f"Raw response: {text[:1000]}"
        ) from exc


def generate_llm_explanation(xai_json):
    """Generate a structured Gemini explanation from the LIVE XAI JSON."""

    api_key = get_setting("GEMINI_API_KEY")

    if not api_key:
        return None, "GEMINI_API_KEY is not configured."

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an explainable AI cybersecurity assistant for an Intrusion Detection
System (IDS).

Analyze ONLY the supplied LIVE XAI JSON. It contains the prediction made for
the current custom network-flow input and its local SHAP and LIME explanations.

Your job is to convert that evidence into a clear, cautious cybersecurity
explanation for a human analyst.

- Do not infer the semantic meaning of encoded categorical feature values
  unless an explicit feature mapping is provided.
- For categorical features such as protocol_type, service, and flag,
  report the encoded value exactly as provided.
- Do not say that protocol_type=1 means TCP unless the input explicitly
  provides that mapping.
- Do not invent attack types, protocol meanings, or real-world causes
  that are not present in the XAI data.
- Preserve numerical feature values exactly as provided.
- Do not change SHAP or LIME values.

IMPORTANT RULES:
1. Do not claim that a model prediction proves an attack.
2. Explain which features pushed the prediction toward and away from the
   predicted class.
3. Compare SHAP and LIME where appropriate, including disagreements.
4. Keep the explanation tied to the supplied feature values and contributions.
5. Give a cautious recommended action.
6. Never recommend automatic blocking or isolation solely because of the
   model prediction.
7. Do not invent attack types, feature meanings, or evidence not present in
   the supplied JSON.
8. Return ONLY valid JSON. Do not use Markdown fences. Do not add text before
   or after the JSON.

Return exactly this JSON structure:
{{
  "threat_assessment": {{
    "prediction": "Normal or Malicious",
    "confidence": 0.0
  }},
  "model_information": {{
    "dataset": "...",
    "selected_model": "..."
  }},
  "key_xai_evidence": {{
    "towards_prediction": [],
    "against_prediction": [],
    "lime": []
  }},
  "interpretation": "...",
  "recommended_action": "..."
}}

LIVE XAI JSON:
{json.dumps(xai_json, indent=2, ensure_ascii=False)}
"""

        # Prefer JSON response mode. If the installed SDK/model rejects it,
        # fall back to a normal request and parse the response robustly below.
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )

        # Some versions of the SDK expose parsed JSON directly.
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, dict):
                return parsed, None
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump(), None

        text = getattr(response, "text", None)
        parsed = _extract_json_object(text)

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response was JSON, but not a JSON object.")

        return parsed, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ============================================================
# SIDEBAR
# ============================================================

side_ui('<div class="ids-sidelabel">⚙ Configuration</div>')

dataset_name = st.sidebar.selectbox(

    "Dataset",

    list(
        DATASET_CONFIG.keys()
    )
)


# ============================================================
# MODEL SELECTION
# ============================================================

adaptive_model = selection_registry.get(
    dataset_name
)

if adaptive_model in [
    "DNN",
    "XGBoost"
]:

    default_index = [
        "DNN",
        "XGBoost"
    ].index(
        adaptive_model
    )

else:

    default_index = 0


model_name = st.sidebar.selectbox(

    "Model",

    [
        "DNN",
        "XGBoost"
    ],

    index=default_index
)


if adaptive_model:

    st.sidebar.success(
        f"Adaptive selector: {adaptive_model}"
    )

    if model_name != adaptive_model:

        st.sidebar.warning(
            "Manual model differs from the "
            "adaptive selector."
        )


# ------------------------------------------------------------
# Sidebar extras (layout + quick guide)
# ------------------------------------------------------------

side_ui('<div class="ids-sidelabel">🧩 Layout</div>')

input_columns = st.sidebar.radio(

    "Input grid columns",

    [2, 3, 4],

    index=1,

    horizontal=True
)

with st.sidebar.expander("ℹ️ Quick guide"):

    st.markdown(
        "1. Pick a **dataset** and **model**.\n"
        "2. Adjust the flow features (defaults come from a small background sample).\n"
        "3. Tick **SHAP + LIME** for explanations.\n"
        "4. Tick **Gemini LLM** for a readable threat report.\n"
        "5. Press **Run Threat Analysis**."
    )


# ============================================================
# MAIN HEADER
# ============================================================

ui(
    '<div class="ids-sub" style="margin-bottom:.2rem">Enter a custom '
    'network-flow instance below. The system will run the full pipeline:</div>'
    + pipeline_html()
)

# ============================================================
# LOAD DATASET
# ============================================================

try:

    train_df = load_training_data(
        dataset_name
    )

except Exception as e:

    st.error(
        str(e)
    )

    st.stop()


target = DATASET_CONFIG[
    dataset_name
]["target"]


feature_names = list(MODEL_REGISTRY[dataset_name]["features"])


# ============================================================
# DISPLAY MODEL PATH
# ============================================================

with st.expander(
    "🔧 System Information"
):

    ui(
        kv_html([

            ("Project root", str(PROJECT_ROOT)),

            ("Saved models", str(SAVED_MODELS_DIR)),

            ("Selected dataset", dataset_name),

            ("Selected model", model_name),

            ("Number of features", len(feature_names)),

            (
                "Model path",
                str(
                    DATASET_CONFIG[
                        dataset_name
                    ][
                        "xgb_model"
                        if model_name == "XGBoost"
                        else "dnn_model"
                    ]
                )
            )
        ])
    )


# ============================================================
# INPUT FORM
# ============================================================

st.subheader(
    "📥 Custom Network Flow Input"
)

st.caption(
    "Enter values in the same feature representation "
    "used during model training."
)


@st.cache_data
def compute_normal_values(dataset_key):
    """Load a reference row; its true class is unknown."""
    df = load_training_data(dataset_key)
    if df.empty:
        raise ValueError(
            f"No reference rows available for {dataset_key}."
        )
    return df.iloc[0].astype(float).to_dict()


@st.cache_data
def compute_median_values(dataset_key):
    """Calculate medians from the available background sample."""
    df = load_training_data(dataset_key)
    if df.empty:
        raise ValueError(
            f"No feature rows available for {dataset_key}."
        )
    return df.median(numeric_only=True).astype(float).to_dict()


normal_values = compute_normal_values(dataset_name)
median_values = compute_median_values(dataset_name)

if "ids_input_mode" not in st.session_state:
    st.session_state["ids_input_mode"] = "normal"

if "ids_reset_nonce" not in st.session_state:
    st.session_state["ids_reset_nonce"] = 0


reset_normal_col, reset_median_col, hint_col = st.columns([1, 1, 2])

with reset_normal_col:
    if st.button(
        "🟢 Load Reference Sample",
        use_container_width=True,
        help="Load the first reference row from the small feature-only background sample; its class is unknown."
    ):
        st.session_state["ids_input_mode"] = "normal"
        st.session_state["ids_reset_nonce"] += 1
        st.rerun()

with reset_median_col:
    if st.button(
        "📊 Reset to Dataset Medians",
        use_container_width=True,
        help="Reset all inputs to the numeric medians of the training dataset."
    ):
        st.session_state["ids_input_mode"] = "median"
        st.session_state["ids_reset_nonce"] += 1
        st.rerun()

with hint_col:
    if st.session_state["ids_input_mode"] == "normal":
        st.caption(
            f"{len(feature_names)} features • currently using a reference "
            f"row from the {dataset_name} background sample (class unknown)"
        )
    else:
        st.caption(
            f"{len(feature_names)} features • currently using feature medians "
            f"from the {dataset_name} background sample"
        )


def get_current_defaults():
    if st.session_state["ids_input_mode"] == "median":
        return median_values
    return normal_values


def input_widget_key(feature):
    return (
        f"input_{dataset_name}_{feature}_"
        f"{st.session_state['ids_reset_nonce']}"
    )


input_values = {}


with st.form(
    "prediction_form"
):

    cols = st.columns(input_columns)


    for i, feature in enumerate(
        feature_names
    ):

        col = cols[
            i % input_columns
        ]

        current_defaults = get_current_defaults()

        default_value = current_defaults.get(
            feature,
            0.0
        )


        with col:

            input_values[
                feature
            ] = st.number_input(

                feature,

                value=default_value,

                format="%.6f",

                key=input_widget_key(feature)
            )


    st.markdown("---")


    opt1, opt2 = st.columns(2)


    with opt1:

        generate_xai = st.checkbox(

            "Generate SHAP + LIME explanations",

            value=True,

            help="Local feature-contribution explanations "
                 "for this exact input."
        )


    with opt2:

        generate_llm = st.checkbox(

            "Generate Gemini LLM explanation",

            value=False,

            help="Readable threat-assessment report written by "
                 "Gemini from the SHAP + LIME evidence "
                 "(needs GEMINI_API_KEY)."
        )


    submitted = st.form_submit_button(

        "🚀 Run Threat Analysis",

        type="primary"
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if submitted:

    tracker = None

    stage = 0

    try:

        # ----------------------------------------------------
        # Create input DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [
                input_values
            ],

            columns=feature_names
        )


        st.markdown("---")

        tracker = st.empty()


        # ----------------------------------------------------
        # Load selected model
        # ----------------------------------------------------

        stage = 1

        show_pipeline(tracker, done=1, active=1)

        with st.spinner(
            f"Loading {model_name}..."
        ):

            model = load_model(
                dataset_name,
                model_name
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        stage = 2

        show_pipeline(tracker, done=2, active=2)

        with st.spinner(
            "Running live inference..."
        ):

            prediction = make_prediction(

                model,

                model_name,

                input_df
            )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.subheader(
            "🎯 Prediction Result"
        )


        is_malicious = (
            prediction["class"] == 1
        )

        if is_malicious:

            verdict_sub = (
                "Model classified the input as <b>MALICIOUS</b>. "
                "Review the explanation below before taking action."
            )

        else:

            verdict_sub = (
                "Model classified the input as <b>NORMAL</b>. "
                "Open the explanation to see which features drove "
                "this result."
            )

        ui(
            verdict_html(

                "bad" if is_malicious else "ok",

                (
                    "⚠ " if is_malicious else "✓ "
                ) + prediction["label"].upper(),

                prediction["confidence"] * 100,

                "Live model verdict",

                verdict_sub,

                [
                    ("dataset", dataset_name),
                    ("model", model_name),
                    ("features", len(feature_names))
                ]
            )
        )


        ui(
            prob_html(
                prediction["probability_class_0"],
                prediction["probability_class_1"]
            )
        )


        # ====================================================
        # XAI
        # ====================================================

        if generate_xai:

            stage = 3

            show_pipeline(tracker, done=3, active=3)

            st.markdown("---")

            st.header(
                "🔎 Explainable AI"
            )


            with st.spinner(
                "Generating SHAP explanation..."
            ):

                shap_result = (
                    generate_shap_explanation(

                        model,

                        model_name,

                        input_df,

                        train_df,

                        feature_names
                    )
                )


            stage = 4

            show_pipeline(tracker, done=4, active=4)

            with st.spinner(
                "Generating LIME explanation..."
            ):

                lime_result = (
                    generate_lime_explanation(

                        model,

                        model_name,

                        input_df,

                        train_df,

                        feature_names,

                        prediction["class"]
                    )
                )


            stage = 5

            show_pipeline(
                tracker,
                done=5,
                active=5 if generate_llm else None,
                skipped=() if generate_llm else (5,)
            )


            # ------------------------------------------------
            # SHAP TOP FEATURES
            # ------------------------------------------------

            shap_rows = []


            for feature, value, shap_value in zip(

                feature_names,

                input_df.iloc[0].values,

                shap_result["values"]
            ):

                shap_rows.append({

                    "Feature":
                        feature,

                    "Value":
                        float(value),

                    "SHAP Value":
                        float(shap_value),

                    "Direction":
                        (
                            "Malicious"
                            if shap_value > 0
                            else
                            "Normal"
                        )
                })


            shap_df = pd.DataFrame(
                shap_rows
            )


            shap_df[
                "Absolute SHAP"
            ] = shap_df[
                "SHAP Value"
            ].abs()


            shap_df = (
                shap_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False
                )
                .head(10)
            )


            # ------------------------------------------------
            # LIME
            # ------------------------------------------------

            lime_rows = []


            for feature, contribution in (
                lime_result
            ):

                lime_rows.append({

                    "Feature / Condition":
                        feature,

                    "Contribution":
                        float(
                            contribution
                        ),

                    "Direction":
                        (
                            "Towards prediction"
                            if contribution > 0
                            else
                            "Away from prediction"
                        )
                })


            lime_df = pd.DataFrame(
                lime_rows
            )


            # ------------------------------------------------
            # STRUCTURED JSON
            # ------------------------------------------------

            xai_json = build_xai_json(

                dataset_name,

                model_name,

                prediction,

                input_df,

                shap_result,

                lime_result,

                feature_names
            )


            saved_path = save_xai_output(
                xai_json
            )


            # ------------------------------------------------
            # XAI TABS
            # ------------------------------------------------

            tab_shap, tab_lime, tab_json = st.tabs([
                "SHAP",
                "LIME",
                "LLM-ready JSON"
            ])


            with tab_shap:

                st.subheader(
                    "SHAP — Top Feature Contributions"
                )

                ui(
                    diverging_html(

                        [
                            (
                                row["Feature"],
                                row["Value"],
                                row["SHAP Value"]
                            )
                            for _, row in shap_df.iterrows()
                        ],

                        "var(--danger)",

                        "var(--safe)",

                        "Pushes toward Malicious",

                        "Pushes toward Normal"
                    )
                )

                with st.expander(
                    "📋 Table view"
                ):

                    ui(
                        table_html(
                            shap_df[
                                [
                                    "Feature",
                                    "Value",
                                    "SHAP Value",
                                    "Direction"
                                ]
                            ]
                        )
                    )


            with tab_lime:

                st.subheader(
                    "LIME — Local Conditions"
                )

                ui(
                    diverging_html(

                        [
                            (
                                row["Feature / Condition"],
                                None,
                                row["Contribution"]
                            )
                            for _, row in lime_df.iterrows()
                        ],

                        "var(--accent)",

                        "var(--warn)",

                        "Towards predicted class",

                        "Away from predicted class",

                        show_value=False
                    )
                )

                with st.expander(
                    "📋 Table view"
                ):

                    ui(
                        table_html(
                            lime_df
                        )
                    )


            with tab_json:

                ui(
                    json_pre_html(
                        xai_json
                    )
                )


            st.success(
                f"XAI explanation saved to:\n{saved_path}"
            )


            # =================================================
            # LLM EXPLANATION
            # =================================================

            if generate_llm:

                st.markdown("---")

                st.header(
                    "🤖 LLM Threat Explanation"
                )


                with st.spinner(
                    "Generating Gemini explanation..."
                ):

                    llm_result, llm_error = (
                        generate_llm_explanation(
                            xai_json
                        )
                    )


                if llm_result:

                    show_pipeline(tracker, done=6)

                    st.success(
                        "Gemini explanation generated successfully."
                    )

                    # -----------------------------------------
                    # READABLE THREAT ASSESSMENT REPORT
                    # -----------------------------------------

                    ui(
                        llm_report_html(
                            llm_result,
                            prediction["label"]
                        )
                    )

                    # -----------------------------------------
                    # RAW LLM JSON
                    # -----------------------------------------

                    with st.expander(
                        "📄 View Raw LLM JSON"
                    ):

                        ui(
                            json_pre_html(
                                llm_result
                            )
                        )

                    # -----------------------------------------
                    # SAVE LLM RESULT
                    # -----------------------------------------

                    llm_filename = (
                        f"{dataset_name}_"
                        f"{model_name}_"
                        f"custom_live_llm_explanation.json"
                    )

                    llm_path = (
                        XAI_OUTPUT_DIR /
                        llm_filename
                    )

                    with open(
                        llm_path,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        json.dump(
                            llm_result,
                            f,
                            indent=4
                        )

                    st.success(
                        f"LLM explanation saved to:\n{llm_path}"
                    )

                else:

                    show_pipeline(tracker, done=5, failed=5)

                    st.warning(
                        "LLM explanation could not be generated."
                    )

                    st.code(
                        llm_error
                    )

        else:

            show_pipeline(
                tracker,
                done=3,
                skipped=(3, 4, 5)
            )

            if generate_llm:

                st.info(
                    "The Gemini explanation is built from the "
                    "SHAP + LIME evidence. Enable "
                    "'Generate SHAP + LIME explanations' to use it."
                )


        # ====================================================
        # FINAL RAW INPUT
        # ====================================================

        with st.expander(
            "📋 View Submitted Input"
        ):

            ui(
                table_html(
                    input_df
                    .T
                    .reset_index()
                    .rename(
                        columns={
                            "index": "Feature",
                            0: "Value"
                        }
                    ),

                    max_height=420
                )
            )

    except Exception as e:

        if tracker is not None:

            show_pipeline(
                tracker,
                done=stage,
                failed=stage
            )

        st.error(
            "Prediction / explanation failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

ui(
    '<div class="ids-footer">ADAPTIVE EXPLAINABLE INTRUSION DETECTION '
    'SYSTEM &nbsp;|&nbsp; PROJECT 1 XAI</div>'
)
