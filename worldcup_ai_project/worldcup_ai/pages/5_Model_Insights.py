import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as _st
_st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#0d0d1a !important; color:#e0e0e0; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a0a1a 0%,#0d0d24 100%) !important; border-right:1px solid #1e1e3a; }
section[data-testid="stSidebar"] * { color:#ccc !important; }
.main .block-container { background-color:#0d0d1a !important; padding-top:1.5rem; }
.stButton > button { background:linear-gradient(135deg,#f0b429,#e67e22) !important; color:#1a1a1a !important; font-weight:700 !important; border:none !important; border-radius:8px !important; }
.stTabs [data-baseweb="tab-list"] { background:#12122a; border-radius:10px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#888 !important; border-radius:8px; font-weight:600; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#f0b429,#e67e22) !important; color:#1a1a1a !important; }
.stDataFrame { border:1px solid #1e1e3a !important; border-radius:10px !important; }
h1,h2,h3,h4 { font-family:'Rajdhani',sans-serif !important; color:#f0b429 !important; }
details { background:#12122a !important; border-radius:10px !important; border:1px solid #1e1e3a !important; }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0d0d1a; } ::-webkit-scrollbar-thumb { background:#2a2a4a; border-radius:3px; }
</style>
""", unsafe_allow_html=True)
from page_modules.model_insights import show
show()
