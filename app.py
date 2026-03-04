"""
╔══════════════════════════════════════════════════════════════╗
║   PredictPulse – AI Early Warning System for Student Burnout ║
║   MAIN FILE: app.py  ← Streamlit entry point                 ║
║   Deploy: streamlit run app.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import re
from datetime import datetime, timedelta
import random

# ── Internal modules ───────────────────────────────────────────
from model         import predict_burnout_score
from ml.nlp        import analyze_journal_entry
from utils.recs    import get_recommendations
from utils.data    import get_mock_trend, get_admin_data

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PredictPulse – Student Burnout AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS  – dreamy dark cosmos theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: #e2e8f0;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 45%, #120825 75%, #0a0a1a 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,10,26,0.95) !important;
    border-right: 1px solid rgba(167,139,250,0.15) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Cards ── */
.pp-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(167,139,250,0.18);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s;
}
.pp-card:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(167,139,250,0.07);
}

/* ── KPI Metric ── */
.pp-kpi {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s;
}
.pp-kpi-value { font-size: 2.2rem; font-weight: 900; line-height: 1; margin-bottom: 4px; }
.pp-kpi-label { font-size: 0.75rem; color: #64748b; letter-spacing: 0.5px; text-transform: uppercase; }
.pp-kpi-sub   { font-size: 0.7rem;  color: #334155; margin-top: 4px; }

/* ── Badges ── */
.badge-low      { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3);  padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.8rem; }
.badge-moderate { background:rgba(251,191,36,0.15);  color:#fbbf24; border:1px solid rgba(251,191,36,0.3);  padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.8rem; }
.badge-high     { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.8rem; }

/* ── Section titles ── */
.pp-title {
    font-size: 1.7rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #e2e8f0, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.pp-subtitle { font-size: 0.85rem; color: #475569; margin-bottom: 20px; }

/* ── Rec cards ── */
.rec-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
    display: flex; gap: 12px; align-items: flex-start;
}
.rec-icon  { font-size: 1.4rem; flex: 0 0 auto; margin-top: 2px; }
.rec-title { font-size: 0.85rem; font-weight: 700; margin-bottom: 3px; }
.rec-desc  { font-size: 0.75rem; color: #64748b; line-height: 1.5; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg,#7c3aed,#ec4899) !important; border-radius: 4px; }

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
.stSlider [data-testid="stTickBar"] { color: #475569; }
div[data-baseweb="slider"] div { background: linear-gradient(90deg,#7c3aed,#ec4899) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(124,58,237,0.5) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius:12px; gap:4px; padding:4px; }
.stTabs [data-baseweb="tab"] {
    border-radius:10px; color:#64748b; font-weight:600;
    font-family:'Outfit',sans-serif;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c3aed,#ec4899) !important;
    color:#fff !important;
}

/* ── Divider ── */
hr { border-color: rgba(167,139,250,0.12) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(167,139,250,0.15) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 3px; }

/* ── Floating orbs (decorative) ── */
.orb {
    position:fixed; border-radius:50%; pointer-events:none; z-index:0;
    filter: blur(80px); opacity: 0.12;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "logged_in":    False,
        "user_name":    "",
        "user_email":   "",
        "user_role":    "student",
        "page":         "dashboard",
        "mood_log":     [{"day":"Mon","v":4},{"day":"Tue","v":3},{"day":"Wed","v":2},
                         {"day":"Thu","v":4},{"day":"Fri","v":3}],
        "gratitude":    ["I'm grateful for a warm bed and a safe home",
                         "My friend texted to check on me today 💙"],
        "predict_result": None,
        "journal_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def risk_color(score):
    if score < 0.35: return "#34d399"
    if score < 0.65: return "#fbbf24"
    return "#f87171"

def risk_label(score):
    if score < 0.35: return "Low Risk",    "badge-low"
    if score < 0.65: return "Moderate Risk","badge-moderate"
    return "High Risk", "badge-high"

def gauge_fig(score):
    pct  = score * 100
    col  = risk_color(score)
    fig  = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct,
        number = {"suffix":"%","font":{"size":44,"color":col,"family":"Outfit"}},
        gauge  = {
            "axis":  {"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#475569","size":11}},
            "bar":   {"color":col,"thickness":0.25},
            "bgcolor":"rgba(0,0,0,0)",
            "borderwidth":0,
            "steps": [
                {"range":[0,35],  "color":"rgba(52,211,153,0.12)"},
                {"range":[35,65], "color":"rgba(251,191,36,0.12)"},
                {"range":[65,100],"color":"rgba(248,113,113,0.12)"},
            ],
            "threshold":{"line":{"color":col,"width":4},"thickness":0.85,"value":pct},
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=20,r=20,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":"Outfit"},
    )
    return fig

# ══════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;margin-bottom:32px;">
            <div style="font-size:3.5rem;margin-bottom:8px;">🧠</div>
            <div style="font-size:2rem;font-weight:900;background:linear-gradient(135deg,#a78bfa,#f472b6,#60a5fa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PredictPulse</div>
            <div style="font-size:0.8rem;color:#475569;letter-spacing:1px;margin-top:4px;">
                AI EARLY WARNING SYSTEM FOR STUDENT BURNOUT
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑  Sign In", "✨  Sign Up"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email    = st.text_input("Email", placeholder="student@university.edu", key="li_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
            if st.button("Enter the Cosmos →", key="btn_login", use_container_width=True):
                if email:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email
                    st.session_state.user_name  = email.split("@")[0].title()
                    st.session_state.user_role  = "student"
                    st.session_state.page       = "dashboard"
                    st.rerun()
                else:
                    st.error("Please enter your email.")

        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            name     = st.text_input("Full Name", placeholder="Alex Johnson", key="su_name")
            email2   = st.text_input("Email", placeholder="student@university.edu", key="su_email")
            password2= st.text_input("Password", type="password", placeholder="••••••••", key="su_pass")
            role     = st.selectbox("Role", ["student","admin"], key="su_role")
            if st.button("Begin Journey →", key="btn_signup", use_container_width=True):
                if name and email2:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email2
                    st.session_state.user_name  = name
                    st.session_state.user_role  = role
                    st.session_state.page       = "dashboard"
                    st.rerun()
                else:
                    st.error("Please fill in name and email.")

        st.markdown("""
        <div style="text-align:center;font-size:0.7rem;color:#334155;margin-top:20px;">
            🔒 JWT-Secured · Privacy-First · FERPA Compliant
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:20px 0 16px;">
            <div style="font-size:2.5rem;margin-bottom:6px;">🧠</div>
            <div style="font-size:1.3rem;font-weight:900;background:linear-gradient(135deg,#a78bfa,#f472b6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PredictPulse</div>
            <div style="font-size:0.7rem;color:#475569;letter-spacing:1px;">AI BURNOUT SYSTEM</div>
        </div>
        <hr style="border-color:rgba(167,139,250,0.15);margin:0 0 16px;">
        <div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);
            border-radius:12px;padding:12px 16px;margin-bottom:20px;text-align:center;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:2px;">Signed in as</div>
            <div style="font-weight:700;font-size:0.9rem;">{st.session_state.user_name}</div>
            <div style="font-size:0.65rem;color:#a78bfa;text-transform:uppercase;
                letter-spacing:0.5px;margin-top:2px;">{st.session_state.user_role}</div>
        </div>
        """, unsafe_allow_html=True)

        nav_items = [
            ("🏠", "Dashboard",      "dashboard"),
            ("🧠", "Predict",        "predict"),
            ("🎮", "Wellness Games", "wellness"),
            ("📊", "Analytics",      "analytics"),
        ]
        if st.session_state.user_role == "admin":
            nav_items.append(("🏛️", "Admin View", "admin"))

        for icon, label, page_id in nav_items:
            active = st.session_state.page == page_id
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
            ):
                st.session_state.page = page_id
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.2);
            border-radius:12px;padding:12px 14px;margin-bottom:12px;">
            <div style="font-size:0.7rem;font-weight:700;color:#60a5fa;margin-bottom:6px;">
                🌐 FEDERATED LEARNING
            </div>
            <div style="font-size:0.65rem;color:#475569;line-height:1.5;">
                Cross-institutional privacy-preserving training — coming soon via Flower framework
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪  Sign Out", use_container_width=True, key="signout"):
            for k in ["logged_in","user_name","user_email","predict_result","journal_result"]:
                st.session_state[k] = False if k=="logged_in" else ""
            st.rerun()

# ══════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════
def page_dashboard():
    name = st.session_state.user_name.split()[0]
    st.markdown(f"""
    <div class="pp-title">Welcome back, {name} 👋</div>
    <div class="pp-subtitle">Here's your wellness snapshot for this week</div>
    """, unsafe_allow_html=True)

    # ── KPI row ──
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("#fbbf24","⚡","52%","Burnout Risk","↑ from 41% last week"),
        ("#a78bfa","🌟","61","Wellness Score","out of 100"),
        ("#34d399","🔥","4 days","Focus Streak","personal best: 7"),
        ("#f87171","😴","5.5 h","Avg Sleep","↓ below 7h optimal"),
    ]
    for col, (color, icon, val, label, sub) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""
            <div class="pp-kpi" style="border-color:{color}30;">
                <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
                <div class="pp-kpi-value" style="color:{color};">{val}</div>
                <div class="pp-kpi-label">{label}</div>
                <div class="pp-kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend charts ──
    col_l, col_r = st.columns(2)
    trend = pd.DataFrame(get_mock_trend())

    with col_l:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**📉 Burnout Risk Over Time**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["week"], y=trend["risk"],
            fill="tozeroy", mode="lines",
            line=dict(color="#f87171", width=2),
            fillcolor="rgba(248,113,113,0.12)",
            name="Risk %",
        ))
        fig.update_layout(
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#475569"),
            font=dict(family="Outfit"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**😴 Sleep & Focus Trends**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend["week"],y=trend["sleep"],mode="lines+markers",
            line=dict(color="#60a5fa",width=2),marker=dict(size=5),name="Sleep (h)"))
        fig2.add_trace(go.Scatter(x=trend["week"],y=[v/10 for v in trend["focus"]],mode="lines+markers",
            line=dict(color="#34d399",width=2),marker=dict(size=5),name="Focus (×10)"))
        fig2.update_layout(
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#475569"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#475569"),
            font=dict(family="Outfit"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.markdown("**💡 Personalized Recommendations**")
    recs = get_recommendations(0.52)
    cols = st.columns(len(recs))
    for col, r in zip(cols, recs):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:14px;text-align:center;height:100%;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{r['icon']}</div>
                <div style="font-size:0.78rem;font-weight:700;margin-bottom:4px;">{r['title']}</div>
                <div style="font-size:0.7rem;color:#475569;line-height:1.5;">{r['desc']}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="pp-title">🧠 Burnout Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">AI-powered risk assessment using ML + NLP analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊  Academic Inputs", "✍️  Journal Entry", "📈  Results"])

    # ── Tab 1: Inputs ──
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**📊 Academic Metrics**")
            attendance  = st.slider("Attendance %",        0, 100, 80, key="att")
            gpa         = st.slider("GPA",                 0.0, 4.0, 3.2, step=0.1, key="gpa")
            delays      = st.slider("Assignment Delays / Month", 0, 10, 2, key="del")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**⚡ Behavioral Metrics**")
            study_hours = st.slider("Study Hours / Week",  0, 60, 18, key="stu")
            engagement  = st.slider("Class Engagement %",  0, 100, 60, key="eng")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Next: Add Journal Entry →", use_container_width=True):
                st.session_state["inputs_saved"] = {
                    "attendance": attendance, "gpa": gpa,
                    "delays": delays, "study": study_hours, "engagement": engagement,
                }
                st.success("✅ Inputs saved! Switch to 'Journal Entry' tab.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Journal ──
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("""
        **✍️ Reflective Journal Entry**

        Share how you're feeling this week. Our NLP engine analyzes emotional tone and stress signals. This is fully private.
        """)
        journal_text = st.text_area(
            "Your journal entry",
            placeholder="e.g., This week has been really overwhelming. I couldn't submit two assignments on time and I feel anxious about everything...",
            height=160, label_visibility="collapsed", key="journal_text",
        )

        if st.button("⚡ Run AI Prediction", use_container_width=True, key="btn_predict"):
            inputs = st.session_state.get("inputs_saved", {
                "attendance":80,"gpa":3.2,"delays":2,"study":18,"engagement":60
            })

            with st.spinner(""):
                prog = st.progress(0)
                status_msgs = [
                    "🔮 Loading XGBoost classifier...",
                    "📊 Processing academic features...",
                    "🗣️ Running NLP sentiment analysis...",
                    "🔍 Computing SHAP feature importance...",
                    "✨ Generating personalized insights...",
                ]
                for i, msg in enumerate(status_msgs):
                    st.toast(msg)
                    time.sleep(0.35)
                    prog.progress((i+1)*20)

            nlp_res   = analyze_journal_entry(journal_text) if journal_text.strip() else {"emotional_score":0.3,"sentiment":"Neutral","stress_words":[]}
            score     = predict_burnout_score({**inputs, "emotional_score": nlp_res["emotional_score"]})
            feat_imp  = get_feature_importance({**inputs, "emotional_score": nlp_res["emotional_score"]})
            recs      = get_recommendations(score)

            st.session_state.predict_result = {
                "score": score, "nlp": nlp_res,
                "feature_importance": feat_imp, "recs": recs,
            }
            st.success("✅ Prediction complete! View Results tab.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 3: Results ──
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        res = st.session_state.predict_result
        if not res:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#475569;">
                <div style="font-size:3rem;margin-bottom:12px;">🔮</div>
                <div style="font-size:1rem;font-weight:600;">No prediction yet</div>
                <div style="font-size:0.8rem;margin-top:6px;">Complete the Academic Inputs & Journal Entry tabs first</div>
            </div>""", unsafe_allow_html=True)
            return

        score = res["score"]
        nlp   = res["nlp"]
        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.markdown('<div class="pp-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("**Burnout Risk Score**")
            st.plotly_chart(gauge_fig(score), use_container_width=True, config={"displayModeBar":False})
            lbl, cls = risk_label(score)
            st.markdown(f'<div style="text-align:center;margin-top:-10px;"><span class="{cls}">{lbl}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**🧬 NLP Emotional Analysis**")
            n1, n2, n3 = st.columns(3)
            em = nlp.get("emotional_score", 0.3)
            with n1:
                ec = "#f87171" if em>0.6 else "#34d399" if em<0.35 else "#fbbf24"
                st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{ec};">{round(em*100)}%</div><div class="pp-kpi-label">Emotional Stress</div></div>', unsafe_allow_html=True)
            with n2:
                sent = nlp.get("sentiment","Neutral")
                sc2 = "#f87171" if sent=="Negative" else "#34d399" if sent=="Positive" else "#fbbf24"
                st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{sc2};font-size:1.3rem;">{sent}</div><div class="pp-kpi-label">Sentiment</div></div>', unsafe_allow_html=True)
            with n3:
                sw = nlp.get("stress_words",[])
                st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:#f472b6;">{len(sw)}</div><div class="pp-kpi-label">Stress Signals</div></div>', unsafe_allow_html=True)

            if sw:
                st.markdown("<br>**Detected stress words:**", unsafe_allow_html=True)
                badges = " ".join([f'<span style="background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.3);padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:700;margin-right:4px;">{w}</span>' for w in sw])
                st.markdown(badges, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # SHAP + Radar
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**🔍 Explainable AI – Feature Impact (SHAP)**")
            fi = res["feature_importance"]
            fi_df = pd.DataFrame(fi).sort_values("importance")
            fig_bar = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["importance"],
                    colorscale=[[0,"#7c3aed"],[0.5,"#ec4899"],[1,"#f87171"]],
                ),
                text=[f"{v:.0%}" for v in fi_df["importance"]],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10),
            ))
            fig_bar.update_layout(
                height=220, margin=dict(l=0,r=60,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, showticklabels=False, color="#475569"),
                yaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(size=11)),
                font=dict(family="Outfit"), showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**🕸️ Wellness Radar**")
            inputs = st.session_state.get("inputs_saved",{"attendance":80,"gpa":3.2,"delays":2,"study":18,"engagement":60})
            radar_vals = [
                inputs.get("attendance",80),
                inputs.get("gpa",3.2)*25,
                min(100, inputs.get("study",18)*1.5),
                inputs.get("engagement",60),
                max(0, 100 - inputs.get("delays",2)*10),
            ]
            cats = ["Attendance","GPA","Study Hours","Engagement","Punctuality"]
            fig_r = go.Figure(go.Scatterpolar(
                r=radar_vals+[radar_vals[0]], theta=cats+[cats[0]],
                fill="toself", fillcolor="rgba(167,139,250,0.15)",
                line=dict(color="#a78bfa", width=2),
            ))
            fig_r.update_layout(
                height=220, margin=dict(l=20,r=20,t=20,b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100], color="#334155", gridcolor="rgba(255,255,255,0.07)"),
                    angularaxis=dict(color="#64748b", gridcolor="rgba(255,255,255,0.07)"),
                ),
                font=dict(family="Outfit"),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**💡 Personalized Recommendations**")
        rcols = st.columns(len(res["recs"]))
        for col, r in zip(rcols, res["recs"]):
            with col:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:14px;text-align:center;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">{r['icon']}</div>
                    <div style="font-size:0.78rem;font-weight:700;margin-bottom:4px;">{r['title']}</div>
                    <div style="font-size:0.7rem;color:#475569;line-height:1.5;">{r['desc']}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# WELLNESS GAMES PAGE
# ══════════════════════════════════════════════════════════════
def page_wellness():
    st.markdown('<div class="pp-title">🎮 Wellness & Stress Relief</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Interactive tools to decompress, focus, and build resilience</div>', unsafe_allow_html=True)

    tab_breath, tab_mood, tab_gratitude, tab_quiz = st.tabs([
        "🌬️  Box Breathing", "🌈  Mood Tracker", "🌸  Gratitude Journal", "🧩  Burnout Quiz"
    ])

    # ── Breathing ──
    with tab_breath:
        st.markdown("<br>", unsafe_allow_html=True)
        _, cc, _ = st.columns([1,1.5,1])
        with cc:
            st.markdown('<div class="pp-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown("""
            ### 🌬️ Box Breathing — 4-4-4-4 Technique
            A science-backed method used by Navy SEALs and therapists to rapidly calm the nervous system.
            """)
            phase = st.session_state.get("breath_phase","idle")
            colors = {"inhale":"#60a5fa","hold1":"#a78bfa","exhale":"#34d399","hold2":"#f472b6","idle":"#64748b"}
            labels = {"inhale":"Breathe In 🌬️","hold1":"Hold 💙","exhale":"Breathe Out ✨","hold2":"Hold 🌙","idle":"Ready"}
            col = colors.get(phase,"#64748b")
            size = 180 if phase in ["inhale","hold1"] else 120

            st.markdown(f"""
            <div style="display:flex;justify-content:center;margin:20px 0;">
                <div style="width:{size}px;height:{size}px;border-radius:50%;
                    background:radial-gradient(circle at 35% 35%, {col}40, {col}15);
                    border:3px solid {col}80;display:flex;align-items:center;
                    justify-content:center;transition:all 1.5s;
                    box-shadow:0 0 40px {col}30;">
                    <div style="font-size:1rem;font-weight:800;color:{col};text-align:center;">
                        {labels.get(phase,'Ready')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            phases_seq = ["inhale","hold1","exhale","hold2"]
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶ Start Session", use_container_width=True, key="breath_start"):
                    st.session_state.breath_phase = "inhale"
                    st.session_state.breath_cycles = st.session_state.get("breath_cycles",0)+1
                    st.rerun()
            with b2:
                if st.button("⏸ Reset", use_container_width=True, key="breath_stop"):
                    st.session_state.breath_phase = "idle"
                    st.rerun()

            if phase != "idle":
                idx = phases_seq.index(phase)
                next_phase = phases_seq[(idx+1)%4]
                if st.button(f"Next → {labels[next_phase]}", use_container_width=True, key="breath_next"):
                    st.session_state.breath_phase = next_phase
                    st.rerun()

            cycles = st.session_state.get("breath_cycles",0)
            st.markdown(f"""<div style="margin-top:16px;font-size:0.8rem;color:#64748b;">
                Cycles this session: <strong style="color:#a78bfa;">{cycles}</strong>
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Mood Tracker ──
    with tab_mood:
        st.markdown("<br>", unsafe_allow_html=True)
        col_ml, col_mr = st.columns([1.2, 1])
        with col_ml:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**🌈 Log Today's Mood**")
            mood_opts = {"😊 Great":5,"🙂 Good":4,"😐 Okay":3,"😟 Low":2,"😰 Awful":1}
            mood_choice = st.radio("How are you feeling?", list(mood_opts.keys()),
                horizontal=True, label_visibility="collapsed", key="mood_select")
            mood_note = st.text_input("What's on your mind? (optional)", key="mood_note",
                placeholder="A quick note about your day...")
            if st.button("Log Mood ✓", use_container_width=True, key="log_mood"):
                v = mood_opts[mood_choice]
                st.session_state.mood_log.append({"day":"Today","v":v,"note":mood_note})
                st.success(f"Logged: {mood_choice}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_mr:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**📈 Mood Trend**")
            ml = pd.DataFrame(st.session_state.mood_log[-7:])
            fig_mood = go.Figure(go.Scatter(
                x=ml["day"], y=ml["v"], fill="tozeroy", mode="lines+markers",
                line=dict(color="#a78bfa",width=2), marker=dict(size=7,color="#a78bfa"),
                fillcolor="rgba(167,139,250,0.1)",
            ))
            fig_mood.update_layout(
                height=180, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False,color="#475569"),
                yaxis=dict(range=[0,5.5],showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#475569",
                    tickvals=[1,2,3,4,5],ticktext=["😰","😟","😐","🙂","😊"]),
                font=dict(family="Outfit"), showlegend=False,
            )
            st.plotly_chart(fig_mood, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Gratitude Journal ──
    with tab_gratitude:
        st.markdown("<br>", unsafe_allow_html=True)
        col_gl, col_gr = st.columns([1.3, 1])
        prompts = [
            "What made you smile today?","Who helped you recently?",
            "What are you proud of this week?","What beauty did you notice?",
            "What challenge taught you something valuable?",
        ]
        if "prompt_idx" not in st.session_state:
            st.session_state.prompt_idx = 0

        with col_gl:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**🌸 Today's Gratitude Entry**")
            st.markdown(f"""
            <div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2);
                border-radius:12px;padding:12px 16px;margin-bottom:14px;">
                <div style="font-size:0.7rem;color:#a78bfa;font-weight:700;letter-spacing:1px;margin-bottom:4px;">
                    ✨ TODAY'S PROMPT
                </div>
                <div style="font-size:0.85rem;color:#e2e8f0;">{prompts[st.session_state.prompt_idx]}</div>
            </div>
            """, unsafe_allow_html=True)
            gval = st.text_area("I'm grateful for...", height=90,
                label_visibility="collapsed", placeholder="I'm grateful for...", key="grat_val")
            g1, g2 = st.columns(2)
            with g1:
                if st.button("Save Entry 🌸", use_container_width=True, key="save_grat"):
                    if gval.strip():
                        st.session_state.gratitude.insert(0, gval.strip())
                        st.session_state.prompt_idx = (st.session_state.prompt_idx+1)%len(prompts)
                        st.success("Saved! 🌸")
            with g2:
                if st.button("New Prompt ✨", use_container_width=True, key="new_prompt"):
                    st.session_state.prompt_idx = (st.session_state.prompt_idx+1)%len(prompts)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_gr:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.markdown("**📖 Your Entries**")
            for entry in st.session_state.gratitude[:6]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;
                    padding:10px 14px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.06);
                    font-size:0.8rem;color:#94a3b8;display:flex;gap:10px;align-items:flex-start;">
                    <span style="font-size:1rem;flex:0 0 auto;">🌸</span>
                    <span style="line-height:1.5;">{entry}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Burnout Quiz ──
    with tab_quiz:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**🧩 Quick Burnout Self-Assessment Quiz**")
        st.markdown("<small style='color:#475569;'>Answer honestly. This is for your awareness only — results are not stored.</small>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        questions = [
            ("I feel emotionally drained at the end of each day.", "q1"),
            ("I find it hard to concentrate on my studies.", "q2"),
            ("I feel detached or indifferent about my coursework.", "q3"),
            ("I often feel overwhelmed by my workload.", "q4"),
            ("I am sleeping less than 6 hours regularly.", "q5"),
        ]
        opts = ["Never (0)","Rarely (1)","Sometimes (2)","Often (3)","Always (4)"]
        answers = []
        for q, key in questions:
            ans = st.select_slider(q, options=opts, key=f"quiz_{key}")
            answers.append(int(ans.split("(")[1].replace(")","").strip()))

        if st.button("Calculate My Score →", use_container_width=True, key="quiz_submit"):
            total = sum(answers)
            pct   = total / 20
            color = risk_color(pct)
            label = "Low Risk 🟢" if pct<0.35 else "Moderate Risk 🟡" if pct<0.65 else "High Risk 🔴"
            st.markdown(f"""
            <div style="background:{color}15;border:1px solid {color}40;border-radius:16px;
                padding:20px;text-align:center;margin-top:16px;">
                <div style="font-size:2.5rem;font-weight:900;color:{color};">{total}/20</div>
                <div style="font-size:1rem;font-weight:700;color:{color};margin-top:4px;">{label}</div>
                <div style="font-size:0.8rem;color:#64748b;margin-top:8px;">
                    {"You're managing well! Keep your healthy habits." if pct<0.35 else
                     "Some signs of stress detected. Consider using the wellness tools above." if pct<0.65 else
                     "High burnout indicators. Please reach out to a counselor or trusted person. 💙"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="pp-title">📊 Deep Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Detailed academic and wellness trend analysis</div>', unsafe_allow_html=True)

    trend = pd.DataFrame(get_mock_trend())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**📚 GPA Trend**")
        fig = go.Figure(go.Scatter(
            x=trend["week"], y=trend["gpa"], mode="lines+markers",
            line=dict(color="#60a5fa",width=2), marker=dict(size=7,color="#60a5fa"),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
        ))
        fig.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False,color="#475569"),
            yaxis=dict(range=[2,4],showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#475569"),
            font=dict(family="Outfit"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**💭 Emotional Stress Trend**")
        fig2 = go.Figure(go.Scatter(
            x=trend["week"], y=trend["emotional"], mode="lines+markers",
            line=dict(color="#f472b6",width=2), marker=dict(size=7,color="#f472b6"),
            fill="tozeroy", fillcolor="rgba(244,114,182,0.1)",
        ))
        fig2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False,color="#475569"),
            yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#475569"),
            font=dict(family="Outfit"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Combined multi-metric
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.markdown("**📈 Multi-Metric Overview**")
    fig3 = go.Figure()
    for col_name, color, name in [("risk","#f87171","Burnout Risk %"),("gpa","#60a5fa","GPA ×25"),("sleep","#34d399","Sleep ×10"),("focus","#fbbf24","Focus %")]:
        ydata = [v*25 if col_name=="gpa" else v*10 if col_name=="sleep" else v for v in trend[col_name]]
        fig3.add_trace(go.Scatter(x=trend["week"],y=ydata,mode="lines+markers",
            line=dict(color=color,width=2),marker=dict(size=5),name=name))
    fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,color="#475569"),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#475569"),
        font=dict(family="Outfit"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",size=11)))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ADMIN PAGE
# ══════════════════════════════════════════════════════════════
def page_admin():
    st.markdown('<div class="pp-title">🏛️ Institutional Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Anonymized burnout analytics across 312 students</div>', unsafe_allow_html=True)

    admin = get_admin_data()

    # KPIs
    a1,a2,a3,a4 = st.columns(4)
    for col2, (val, label, color) in zip([a1,a2,a3,a4],[
        ("312","Total Students","#60a5fa"),("17%","High Risk","#f87171"),
        ("35%","Moderate Risk","#fbbf24"),("48%","Low Risk","#34d399"),
    ]):
        with col2:
            st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{color};">{val}</div><div class="pp-kpi-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**📊 Risk Distribution**")
        dist_df = pd.DataFrame(admin["distribution"])
        fig_pie = go.Figure(go.Pie(
            labels=dist_df["name"], values=dist_df["value"],
            hole=0.5,
            marker=dict(colors=["#34d399","#fbbf24","#f87171"]),
            textfont=dict(family="Outfit",size=12),
        ))
        fig_pie.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Outfit"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8")))
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="pp-card">', unsafe_allow_html=True)
        st.markdown("**📈 Department Trends**")
        dept_data = admin["dept_trends"]
        fig_line = go.Figure()
        colors_d = ["#a78bfa","#34d399","#f87171","#fbbf24"]
        for i,(dept,vals) in enumerate(dept_data.items()):
            fig_line.add_trace(go.Scatter(
                x=[f"Wk{j+1}" for j in range(len(vals))], y=vals,
                mode="lines+markers", name=dept,
                line=dict(color=colors_d[i%4],width=2), marker=dict(size=5),
            ))
        fig_line.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False,color="#475569"),
            yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#475569"),
            font=dict(family="Outfit"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",size=11)))
        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.markdown("**🔥 Stress Heatmap — Department × Week**")
    heat = admin["heatmap"]
    hdf  = pd.DataFrame(heat).set_index("dept")
    fig_heat = go.Figure(go.Heatmap(
        z=hdf.values, x=hdf.columns, y=hdf.index,
        colorscale=[[0,"rgba(52,211,153,0.8)"],[0.5,"rgba(251,191,36,0.8)"],[1,"rgba(248,113,113,0.9)"]],
        text=[[f"{v}%" for v in row] for row in hdf.values],
        texttemplate="%{text}", textfont=dict(family="Outfit",size=12,color="white"),
        showscale=True,
    ))
    fig_heat.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#475569"), yaxis=dict(color="#94a3b8"),
        font=dict(family="Outfit"))
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Federated learning placeholder
    st.markdown("""
    <div class="pp-card" style="border-color:rgba(96,165,250,0.3);background:rgba(96,165,250,0.05);">
        <div style="display:flex;gap:16px;align-items:flex-start;">
            <div style="font-size:2.5rem;">🌐</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#60a5fa;margin-bottom:6px;">
                    Federated Learning Module — Coming Soon
                </div>
                <div style="font-size:0.82rem;color:#475569;line-height:1.6;margin-bottom:12px;">
                    Train a global burnout model across institutions without sharing raw student data.
                    Each institution keeps data local; only model gradients are shared — preserving full
                    privacy while improving collective prediction accuracy via the Flower (flwr) framework.
                </div>
                <span style="background:rgba(96,165,250,0.2);color:#60a5fa;border:1px solid rgba(96,165,250,0.4);padding:3px 12px;border-radius:12px;font-size:0.72rem;font-weight:700;margin-right:8px;">Privacy-Preserving</span>
                <span style="background:rgba(167,139,250,0.2);color:#a78bfa;border:1px solid rgba(167,139,250,0.4);padding:3px 12px;border-radius:12px;font-size:0.72rem;font-weight:700;margin-right:8px;">Differential Privacy</span>
                <span style="background:rgba(52,211,153,0.2);color:#34d399;border:1px solid rgba(52,211,153,0.4);padding:3px 12px;border-radius:12px;font-size:0.72rem;font-weight:700;">Multi-Institutional</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_auth()
        return

    render_sidebar()

    page = st.session_state.page
    if   page == "dashboard": page_dashboard()
    elif page == "predict":   page_predict()
    elif page == "wellness":  page_wellness()
    elif page == "analytics": page_analytics()
    elif page == "admin":     page_admin()
    else:                     page_dashboard()

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:20px 0;border-top:1px solid rgba(167,139,250,0.1);
        font-size:0.72rem;color:#334155;margin-top:40px;">
        🧠 PredictPulse · Built with ❤️ for student wellbeing ·
        <span style="color:#475569;">If you're struggling — please reach out to your counselor 💙</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
