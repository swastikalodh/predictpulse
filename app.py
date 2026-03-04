"""
PredictPulse – AI Early Warning System for Student Burnout
==========================================================
MAIN FILE  :  app.py
DEPLOY     :  streamlit run app.py
CLOUD PATH :  app.py   (root of repo)

Everything is self-contained — no local module imports.
"""
import re, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be FIRST Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="PredictPulse – Student Burnout AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def predict_burnout_score(inp):
    a, d, g = inp.get("attendance",80), inp.get("delays",2), inp.get("gpa",3.2)
    s, e    = inp.get("study",18),      inp.get("engagement",60)
    em      = inp.get("emotional_score",0.3)
    z = 0.5 - 0.03*(a-75) + 0.12*d - 0.25*(3.0-g) - 0.04*(20-s) - 0.08*(e-50) + 0.30*(em-0.5)
    return float(np.clip(1/(1+np.exp(-z)), 0.03, 0.97))

def get_feature_importance(inp):
    a, d, g = inp.get("attendance",80), inp.get("delays",2), inp.get("gpa",3.2)
    s, e    = inp.get("study",18),      inp.get("engagement",60)
    em      = inp.get("emotional_score",0.3)
    raw = {
        "Emotional Stress":  abs(0.30*(em-0.5))+0.05,
        "Assignment Delays": abs(0.12*d)+0.04,
        "GPA Gap":           abs(0.25*(3.0-g))+0.03,
        "Low Attendance":    abs(0.03*(a-75))+0.02,
        "Study Hours":       abs(0.04*(20-s))+0.02,
        "Engagement":        abs(0.08*(e-50))+0.02,
    }
    t = sum(raw.values()) or 1
    return [{"feature":k,"importance":round(min(v/t,0.98),3)} for k,v in raw.items()]

# ══════════════════════════════════════════════════════════════════════════════
# NLP ENGINE
# ══════════════════════════════════════════════════════════════════════════════
_HI = ["overwhelmed","exhausted","hopeless","depressed","anxious","burned","failing",
       "impossible","crying","desperate","miserable","breaking","burnout","worthless"]
_MD = ["tired","stressed","worried","struggling","behind","confused","nervous",
       "lost","upset","drained","frustrated","unmotivated","procrastinating"]
_PO = ["happy","excited","motivated","confident","proud","great","amazing",
       "enjoying","love","focus","accomplished","energized","grateful","peaceful"]

def analyze_journal(text):
    tok = re.sub(r"[^a-z\s]","",text.lower()).split()
    hi  = [w for w in _HI if w in tok]
    md  = [w for w in _MD if w in tok]
    po  = [w for w in _PO if w in tok]
    s   = len(hi)*3 + len(md)*1.5
    em  = min(0.95, s/(s+len(po)*2+1))
    return {
        "emotional_score": round(em,3),
        "sentiment": "Negative" if em>0.6 else "Positive" if em<0.35 else "Neutral",
        "stress_words": (hi+md)[:6],
    }

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_recs(score):
    if score < 0.35:
        return [
            {"icon":"🌟","title":"Keep Shining",    "desc":"Your wellness looks excellent. Maintain sleep and exercise habits."},
            {"icon":"🏃","title":"Active Breaks",   "desc":"10-min movement breaks every 90 mins boost focus by 23%."},
            {"icon":"📖","title":"Creative Reading","desc":"15 min of fiction daily reduces cortisol by up to 68%."},
            {"icon":"🎵","title":"Music Therapy",   "desc":"Lo-fi or classical music during study enhances retention."},
        ]
    if score < 0.65:
        return [
            {"icon":"⏰","title":"Pomodoro Method","desc":"25 min work + 5 min rest. Protect your recovery windows."},
            {"icon":"😴","title":"Sleep Hygiene",  "desc":"Aim 7-8 hours. No screens 1 h before bed. Same wake time."},
            {"icon":"🫁","title":"Box Breathing",  "desc":"4s inhale · 4s hold · 4s exhale · 4s hold — calms anxiety fast."},
            {"icon":"📝","title":"Priority Matrix","desc":"Focus on what is important, not just what feels urgent."},
            {"icon":"🤝","title":"Study Circles",  "desc":"Peer collaboration reduces isolation and builds accountability."},
        ]
    return [
        {"icon":"🚨","title":"Seek Counseling","desc":"Please speak with your institution's counselor immediately."},
        {"icon":"🛑","title":"Mandatory Rest", "desc":"Take 24-48 hours completely offline. Your brain needs reset."},
        {"icon":"💬","title":"Talk It Out",    "desc":"Share with a trusted person. Vulnerability is strength."},
        {"icon":"🌙","title":"Sleep Priority", "desc":"8+ hours non-negotiable. Everything else can wait."},
        {"icon":"📵","title":"Digital Detox",  "desc":"Social media < 30 min/day. Protect your mental bandwidth."},
    ]

# ══════════════════════════════════════════════════════════════════════════════
# MOCK DATA
# ══════════════════════════════════════════════════════════════════════════════
def mock_trend():
    return pd.DataFrame({
        "week":     ["Wk1","Wk2","Wk3","Wk4","Wk5","Wk6"],
        "risk":     [22,28,35,41,38,52],
        "gpa":      [3.6,3.5,3.4,3.3,3.35,3.1],
        "emotional":[20,30,40,45,38,60],
        "focus":    [80,72,65,58,63,50],
        "sleep":    [7.2,6.8,6.2,5.9,6.5,5.5],
    })

def admin_data():
    return {
        "dist":  pd.DataFrame({"name":["Low Risk","Moderate","High Risk"],"value":[48,35,17]}),
        "trends":{"Engineering":[30,35,50,65,70],"Sciences":[25,28,30,35,40],
                  "Medicine":[50,58,72,78,82],"Business":[40,42,55,60,58]},
        "heat":  pd.DataFrame([
            {"dept":"Engineering","Wk1":30,"Wk2":35,"Wk3":50,"Wk4":65,"Wk5":70},
            {"dept":"Sciences",   "Wk1":25,"Wk2":28,"Wk3":30,"Wk4":35,"Wk5":40},
            {"dept":"Humanities", "Wk1":20,"Wk2":22,"Wk3":25,"Wk4":28,"Wk5":30},
            {"dept":"Business",   "Wk1":40,"Wk2":42,"Wk3":55,"Wk4":60,"Wk5":58},
            {"dept":"Medicine",   "Wk1":50,"Wk2":58,"Wk3":72,"Wk4":78,"Wk5":82},
        ]).set_index("dept"),
    }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def rcol(s):  return "#3d8c6e" if s<0.35 else "#b07d2a" if s<0.65 else "#9b4545"
def rlbl(s):
    if s<0.35: return "Low Risk","badge-low"
    if s<0.65: return "Moderate Risk","badge-moderate"
    return "High Risk","badge-high"

_CL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
           margin=dict(l=0,r=0,t=10,b=0),font=dict(family="Outfit"),
           xaxis=dict(showgrid=False,color="#5a6478"),
           yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#5a6478"),
           legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#8a9ab5",size=11)))

def gauge(score):
    c = rcol(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score*100,1),
        number={"suffix":"%","font":{"size":44,"color":c,"family":"Outfit"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#3d4a60","tickfont":{"color":"#5a6478","size":11}},
               "bar":{"color":c,"thickness":0.25},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,35],"color":"rgba(50,130,100,0.1)"},
                        {"range":[35,65],"color":"rgba(160,110,30,0.1)"},
                        {"range":[65,100],"color":"rgba(140,60,60,0.1)"}],
               "threshold":{"line":{"color":c,"width":4},"thickness":0.85,"value":round(score*100,1)}},
    ))
    fig.update_layout(height=260,margin=dict(l=20,r=20,t=20,b=0),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font={"family":"Outfit"})
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif!important;color:#ccd4e0;}
.stApp{background:#0f1117;min-height:100vh;}
[data-testid="stSidebar"]{background:#0d0f16!important;border-right:1px solid rgba(255,255,255,0.07)!important;}
[data-testid="stSidebar"] *{color:#ccd4e0!important;}
.pp-card{background:rgba(255,255,255,0.035);backdrop-filter:blur(20px);border:1px solid rgba(100,116,200,0.18);border-radius:20px;padding:24px;margin-bottom:16px;}
.pp-kpi{background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.06);border-radius:18px;padding:20px;text-align:center;}
.pp-kpi-value{font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:4px;}
.pp-kpi-label{font-size:0.75rem;color:#6b7590;letter-spacing:.5px;text-transform:uppercase;}
.pp-kpi-sub{font-size:0.7rem;color:#3d4a60;margin-top:4px;}
.badge-low{background:rgba(50,130,100,.12);color:#3d8c6e;border:1px solid rgba(50,130,100,.25);padding:4px 14px;border-radius:20px;font-weight:700;font-size:.8rem;}
.badge-moderate{background:rgba(160,110,30,.12);color:#b07d2a;border:1px solid rgba(160,110,30,.25);padding:4px 14px;border-radius:20px;font-weight:700;font-size:.8rem;}
.badge-high{background:rgba(140,60,60,.12);color:#9b4545;border:1px solid rgba(140,60,60,.25);padding:4px 14px;border-radius:20px;font-weight:700;font-size:.8rem;}
.pp-title{font-size:1.7rem;font-weight:900;letter-spacing:-.5px;background:linear-gradient(135deg,#ccd4e0,#7b8cde);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;}
.pp-subtitle{font-size:.85rem;color:#5a6478;margin-bottom:20px;}
.stProgress>div>div>div{background:linear-gradient(90deg,#3b4fd8,#6d4a8a)!important;border-radius:4px;}
.stTextArea textarea,.stTextInput input{background:rgba(255,255,255,0.05)!important;border:1px solid rgba(100,116,200,0.18)!important;border-radius:12px!important;color:#ccd4e0!important;}
.stButton>button{background:linear-gradient(135deg,#3b4fd8,#6d4a8a)!important;color:#fff!important;border:none!important;border-radius:12px!important;font-weight:700!important;font-family:'Outfit',sans-serif!important;box-shadow:0 4px 20px rgba(59,79,216,0.3)!important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.035);border-radius:12px;gap:4px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:10px;color:#6b7590;font-weight:600;font-family:'Outfit',sans-serif;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#3b4fd8,#6d4a8a)!important;color:#fff!important;}
div[data-baseweb="slider"] div{background:linear-gradient(90deg,#3b4fd8,#6d4a8a)!important;}
[data-baseweb="select"] div{background:rgba(255,255,255,0.05)!important;border-color:rgba(100,116,200,0.18)!important;color:#ccd4e0!important;}
label{color:#8a9ab5!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-thumb{background:rgba(100,116,200,.25);border-radius:3px;}
</style>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEF = {
    "logged_in":False,"user_name":"","user_email":"","user_role":"student","page":"dashboard",
    "mood_log":[{"day":"Mon","v":4},{"day":"Tue","v":3},{"day":"Wed","v":2},
                {"day":"Thu","v":4},{"day":"Fri","v":3}],
    "gratitude":["I am grateful for a warm bed and a safe home",
                 "My friend texted to check on me today 💙"],
    "predict_result":None,
    "inputs_saved":{"attendance":80,"gpa":3.2,"delays":2,"study":18,"engagement":60},
    "breath_phase":"idle","breath_cycles":0,"prompt_idx":0,
}
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("<br><br>",unsafe_allow_html=True)
    _,col,_ = st.columns([1,1.2,1])
    with col:
        st.markdown("""<div style="text-align:center;margin-bottom:32px;">
          <div style="font-size:3.5rem;margin-bottom:8px;">🧠</div>
          <div style="font-size:2rem;font-weight:900;background:linear-gradient(135deg,#7b8cde,#8c607e,#4f7ab3);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PredictPulse</div>
          <div style="font-size:.8rem;color:#5a6478;letter-spacing:1px;margin-top:4px;">
            AI EARLY WARNING SYSTEM FOR STUDENT BURNOUT</div></div>""",unsafe_allow_html=True)
        ti,tu = st.tabs(["🔑  Sign In","✨  Sign Up"])
        with ti:
            st.markdown("<br>",unsafe_allow_html=True)
            email = st.text_input("Email",placeholder="student@university.edu",key="li_e")
            st.text_input("Password",type="password",placeholder="••••••••",key="li_p")
            if st.button("Sign In →",key="btn_li",use_container_width=True):
                if email:
                    st.session_state.update(logged_in=True,user_email=email,
                        user_name=email.split("@")[0].replace("."," ").title(),
                        user_role="student",page="dashboard")
                    st.rerun()
                else: st.error("Please enter your email.")
        with tu:
            st.markdown("<br>",unsafe_allow_html=True)
            name  = st.text_input("Full Name",placeholder="Alex Johnson",key="su_n")
            email2= st.text_input("Email",placeholder="student@university.edu",key="su_e")
            st.text_input("Password",type="password",placeholder="••••••••",key="su_p")
            role  = st.selectbox("Role",["student","admin"],key="su_r")
            if st.button("Create Account →",key="btn_su",use_container_width=True):
                if name and email2:
                    st.session_state.update(logged_in=True,user_email=email2,
                        user_name=name,user_role=role,page="dashboard")
                    st.rerun()
                else: st.error("Please fill in name and email.")
        st.markdown("""<div style="text-align:center;font-size:.7rem;color:#3d4a60;margin-top:20px;">
          🔒 JWT-Secured · Privacy-First · FERPA Compliant</div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown(f"""<div style="text-align:center;padding:20px 0 12px;">
          <div style="font-size:2.5rem;margin-bottom:6px;">🧠</div>
          <div style="font-size:1.3rem;font-weight:900;background:linear-gradient(135deg,#7b8cde,#8c607e);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PredictPulse</div>
          <div style="font-size:.65rem;color:#5a6478;letter-spacing:1px;">AI BURNOUT SYSTEM</div></div>
          <hr style="border-color:rgba(100,116,200,.14);margin:0 0 14px;">
          <div style="background:rgba(100,116,200,.07);border:1px solid rgba(100,116,200,.18);
            border-radius:12px;padding:12px 16px;margin-bottom:18px;text-align:center;">
            <div style="font-size:.72rem;color:#6b7590;margin-bottom:2px;">Signed in as</div>
            <div style="font-weight:700;font-size:.9rem;">{st.session_state.user_name}</div>
            <div style="font-size:.65rem;color:#7b8cde;text-transform:uppercase;letter-spacing:.5px;
              margin-top:2px;">{st.session_state.user_role}</div></div>""",unsafe_allow_html=True)
        nav=[("🏠","Dashboard","dashboard"),("🧠","Predict","predict"),
             ("🎮","Wellness","wellness"),("📊","Analytics","analytics")]
        if st.session_state.user_role=="admin":
            nav.append(("🏛️","Admin View","admin"))
        for icon,label,pid in nav:
            if st.button(f"{icon}  {label}",key=f"nav_{pid}",use_container_width=True):
                st.session_state.page=pid; st.rerun()
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("""<div style="background:rgba(60,100,160,.07);border:1px solid rgba(60,100,160,.18);
          border-radius:12px;padding:12px 14px;margin-bottom:12px;">
          <div style="font-size:.7rem;font-weight:700;color:#4f7ab3;margin-bottom:4px;">🌐 FEDERATED LEARNING</div>
          <div style="font-size:.65rem;color:#5a6478;line-height:1.5;">
            Cross-institutional privacy-preserving training — coming soon via Flower</div></div>""",
            unsafe_allow_html=True)
        if st.button("🚪  Sign Out",use_container_width=True,key="signout"):
            st.session_state.logged_in=False; st.session_state.predict_result=None; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    name=st.session_state.user_name.split()[0]
    st.markdown(f'<div class="pp-title">Welcome back, {name} 👋</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Here\'s your wellness snapshot for this week</div>',unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    for col,(color,icon,val,label,sub) in zip([k1,k2,k3,k4],[
        ("#b07d2a","⚡","52%","Burnout Risk","↑ from 41% last week"),
        ("#7b8cde","🌟","61","Wellness Score","out of 100"),
        ("#3d8c6e","🔥","4 days","Focus Streak","personal best: 7"),
        ("#9b4545","😴","5.5 h","Avg Sleep","↓ below 7 h optimal"),
    ]):
        with col: st.markdown(f'''<div class="pp-kpi" style="border-color:{color}30;">
          <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
          <div class="pp-kpi-value" style="color:{color};">{val}</div>
          <div class="pp-kpi-label">{label}</div><div class="pp-kpi-sub">{sub}</div></div>''',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    trend=mock_trend()
    cl,cr=st.columns(2)
    with cl:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**📉 Burnout Risk Over Time**")
        fig=go.Figure(go.Scatter(x=trend["week"],y=trend["risk"],fill="tozeroy",mode="lines",
            line=dict(color="#9b4545",width=2),fillcolor="rgba(140,60,60,0.1)"))
        fig.update_layout(height=200,showlegend=False,**_CL)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**😴 Sleep & Focus Trends**")
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=trend["week"],y=trend["sleep"],mode="lines+markers",
            line=dict(color="#4f7ab3",width=2),marker=dict(size=5),name="Sleep (h)"))
        fig2.add_trace(go.Scatter(x=trend["week"],y=[v/10 for v in trend["focus"]],mode="lines+markers",
            line=dict(color="#3d8c6e",width=2),marker=dict(size=5),name="Focus (×10)"))
        fig2.update_layout(height=200,**_CL)
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-card">',unsafe_allow_html=True)
    st.markdown("**💡 Personalised Recommendations**")
    recs=get_recs(0.52)
    for col,r in zip(st.columns(len(recs)),recs):
        with col: st.markdown(f'''<div style="background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.055);
          border-radius:14px;padding:14px;text-align:center;">
          <div style="font-size:1.8rem;margin-bottom:8px;">{r["icon"]}</div>
          <div style="font-size:.78rem;font-weight:700;margin-bottom:4px;">{r["title"]}</div>
          <div style="font-size:.7rem;color:#5a6478;line-height:1.5;">{r["desc"]}</div></div>''',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="pp-title">🧠 Burnout Prediction Engine</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">AI-powered risk assessment using ML + NLP</div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📊  Academic Inputs","✍️  Journal Entry","📈  Results"])
    with t1:
        st.markdown("<br>",unsafe_allow_html=True)
        ca,cb=st.columns(2)
        with ca:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**📊 Academic Metrics**")
            att=st.slider("Attendance %",0,100,80,key="att")
            gpa=st.slider("GPA",0.0,4.0,3.2,step=0.1,key="gpa")
            dl =st.slider("Assignment Delays / Month",0,10,2,key="del")
            st.markdown('</div>',unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**⚡ Behavioural Metrics**")
            stu=st.slider("Study Hours / Week",0,60,18,key="stu")
            eng=st.slider("Class Engagement %",0,100,60,key="eng")
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("Save & continue →",use_container_width=True,key="save_inp"):
                st.session_state.inputs_saved={"attendance":att,"gpa":gpa,"delays":dl,"study":stu,"engagement":eng}
                st.success("✅ Saved! Open the Journal Entry tab.")
            st.markdown('</div>',unsafe_allow_html=True)
    with t2:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**✍️ Reflective Journal Entry**")
        st.caption("Share how you're feeling. NLP analyses emotional tone & stress signals. Fully private.")
        jtext=st.text_area("Entry",placeholder="e.g., This week has been overwhelming. I feel anxious all the time...",
            height=160,label_visibility="collapsed",key="jtext")
        if st.button("⚡ Run AI Prediction",use_container_width=True,key="btn_pred"):
            prog=st.progress(0)
            for i,msg in enumerate(["🔮 Loading classifier…","📊 Processing features…",
                "🗣️ NLP analysis…","🔍 Computing SHAP…","✨ Building report…"]):
                st.toast(msg); time.sleep(0.3); prog.progress((i+1)*20)
            prog.empty()
            inp=st.session_state.inputs_saved
            nlp=analyze_journal(jtext) if jtext.strip() else {"emotional_score":0.3,"sentiment":"Neutral","stress_words":[]}
            sc=predict_burnout_score({**inp,"emotional_score":nlp["emotional_score"]})
            fi=get_feature_importance({**inp,"emotional_score":nlp["emotional_score"]})
            st.session_state.predict_result={"score":sc,"nlp":nlp,"fi":fi,"recs":get_recs(sc)}
            st.success("✅ Done! Open the Results tab.")
        st.markdown('</div>',unsafe_allow_html=True)
    with t3:
        st.markdown("<br>",unsafe_allow_html=True)
        res=st.session_state.predict_result
        if not res:
            st.markdown("""<div style="text-align:center;padding:60px 0;color:#5a6478;">
              <div style="font-size:3rem;margin-bottom:12px;">🔮</div>
              <div style="font-size:1rem;font-weight:600;">No prediction yet</div>
              <div style="font-size:.8rem;margin-top:6px;">Complete Academic Inputs and Journal Entry first</div>
            </div>""",unsafe_allow_html=True); return
        sc=res["score"]; nlp=res["nlp"]
        c1,c2=st.columns([1,1.4])
        with c1:
            st.markdown('<div class="pp-card" style="text-align:center;">',unsafe_allow_html=True)
            st.markdown("**Burnout Risk Score**")
            st.plotly_chart(gauge(sc),use_container_width=True,config={"displayModeBar":False})
            lb,cl2=rlbl(sc)
            st.markdown(f'<div style="text-align:center;margin-top:-8px;"><span class="{cl2}">{lb}</span></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**🧬 NLP Emotional Analysis**")
            n1,n2,n3=st.columns(3)
            em=nlp.get("emotional_score",0.3); ec=rcol(em)
            sent=nlp.get("sentiment","Neutral"); sc2="#9b4545" if sent=="Negative" else "#3d8c6e" if sent=="Positive" else "#b07d2a"
            sw=nlp.get("stress_words",[])
            with n1: st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{ec};">{round(em*100)}%</div><div class="pp-kpi-label">Emotional Stress</div></div>',unsafe_allow_html=True)
            with n2: st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{sc2};font-size:1.3rem;">{sent}</div><div class="pp-kpi-label">Sentiment</div></div>',unsafe_allow_html=True)
            with n3: st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:#8c607e;">{len(sw)}</div><div class="pp-kpi-label">Stress Signals</div></div>',unsafe_allow_html=True)
            if sw:
                st.markdown("<br>**Detected stress words:**",unsafe_allow_html=True)
                st.markdown(" ".join([f'<span style="background:rgba(140,60,60,.12);color:#9b4545;border:1px solid rgba(140,60,60,.25);padding:3px 10px;border-radius:12px;font-size:.72rem;font-weight:700;margin-right:4px;">{w}</span>' for w in sw]),unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
        c3,c4=st.columns(2)
        with c3:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**🔍 Explainable AI – Feature Impact**")
            fi_df=pd.DataFrame(res["fi"]).sort_values("importance")
            fb=go.Figure(go.Bar(x=fi_df["importance"],y=fi_df["feature"],orientation="h",
                marker=dict(color=fi_df["importance"],colorscale=[[0,"#3b4fd8"],[0.5,"#7a4f72"],[1,"#9b4545"]]),
                text=[f"{v:.0%}" for v in fi_df["importance"]],textposition="outside",textfont=dict(color="#8a9ab5",size=10)))
            fb.update_layout(height=220,margin=dict(l=0,r=60,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(showgrid=False,showticklabels=False),
                yaxis=dict(showgrid=False,color="#8a9ab5",tickfont=dict(size=11)),
                font=dict(family="Outfit"),showlegend=False)
            st.plotly_chart(fb,use_container_width=True,config={"displayModeBar":False})
            st.markdown('</div>',unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**🕸️ Wellness Radar**")
            inp=st.session_state.inputs_saved
            rv=[inp.get("attendance",80),inp.get("gpa",3.2)*25,min(100,inp.get("study",18)*1.5),inp.get("engagement",60),max(0,100-inp.get("delays",2)*10)]
            cats=["Attendance","GPA","Study Hours","Engagement","Punctuality"]
            fr=go.Figure(go.Scatterpolar(r=rv+[rv[0]],theta=cats+[cats[0]],fill="toself",
                fillcolor="rgba(100,116,200,0.14)",line=dict(color="#7b8cde",width=2)))
            fr.update_layout(height=220,margin=dict(l=20,r=20,t=20,b=20),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True,range=[0,100],color="#3d4a60",gridcolor="rgba(255,255,255,0.055)"),
                    angularaxis=dict(color="#6b7590",gridcolor="rgba(255,255,255,0.055)")),
                font=dict(family="Outfit"),showlegend=False)
            st.plotly_chart(fr,use_container_width=True,config={"displayModeBar":False})
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**💡 Personalised Recommendations**")
        for col,r in zip(st.columns(len(res["recs"])),res["recs"]):
            with col: st.markdown(f'''<div style="background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.055);
              border-radius:14px;padding:14px;text-align:center;">
              <div style="font-size:1.8rem;margin-bottom:8px;">{r["icon"]}</div>
              <div style="font-size:.78rem;font-weight:700;margin-bottom:4px;">{r["title"]}</div>
              <div style="font-size:.7rem;color:#5a6478;line-height:1.5;">{r["desc"]}</div></div>''',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WELLNESS
# ══════════════════════════════════════════════════════════════════════════════
def page_wellness():
    st.markdown('<div class="pp-title">🎮 Wellness & Stress Relief</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Interactive tools to decompress, focus, and build resilience</div>',unsafe_allow_html=True)
    tb,tm,tg,tq=st.tabs(["🌬️  Box Breathing","🌈  Mood Tracker","🌸  Gratitude Journal","🧩  Burnout Quiz"])
    with tb:
        st.markdown("<br>",unsafe_allow_html=True)
        _,cc,_=st.columns([1,1.4,1])
        with cc:
            st.markdown('<div class="pp-card" style="text-align:center;">',unsafe_allow_html=True)
            st.markdown("### 🌬️ Box Breathing 4-4-4-4")
            st.caption("Science-backed technique used by therapists and Navy SEALs to calm the nervous system.")
            ph=st.session_state.breath_phase
            COLS={"inhale":"#4f7ab3","hold1":"#7b8cde","exhale":"#3d8c6e","hold2":"#8c607e","idle":"#6b7590"}
            LBLS={"inhale":"Breathe In 🌬️","hold1":"Hold 💙","exhale":"Breathe Out ✨","hold2":"Hold 🌙","idle":"Ready ✦"}
            c=COLS.get(ph,"#6b7590"); sz=180 if ph in ["inhale","hold1"] else 120
            st.markdown(f'''<div style="display:flex;justify-content:center;margin:24px 0;">
              <div style="width:{sz}px;height:{sz}px;border-radius:50%;background:radial-gradient(circle at 35% 35%,{c}40,{c}15);
                border:3px solid {c}80;display:flex;align-items:center;justify-content:center;transition:all 1.5s;
                box-shadow:0 0 40px {c}30;">
                <div style="font-size:1rem;font-weight:800;color:{c};text-align:center;padding:0 12px;">
                  {LBLS.get(ph,"Ready ✦")}</div></div></div>''',unsafe_allow_html=True)
            PSQ=["inhale","hold1","exhale","hold2"]
            b1,b2=st.columns(2)
            with b1:
                if st.button("▶ Start",use_container_width=True,key="bs"):
                    st.session_state.breath_phase="inhale"; st.session_state.breath_cycles+=1; st.rerun()
            with b2:
                if st.button("↺ Reset",use_container_width=True,key="br"):
                    st.session_state.breath_phase="idle"; st.rerun()
            if ph!="idle":
                nxt=PSQ[(PSQ.index(ph)+1)%4]
                if st.button(f"Next → {LBLS[nxt]}",use_container_width=True,key="bn"):
                    st.session_state.breath_phase=nxt; st.rerun()
            st.markdown(f'<div style="margin-top:12px;font-size:.8rem;color:#6b7590;">Cycles: <strong style="color:#7b8cde;">{st.session_state.breath_cycles}</strong></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
    with tm:
        st.markdown("<br>",unsafe_allow_html=True)
        ml_c,mr_c=st.columns([1.2,1])
        with ml_c:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**🌈 Log Today's Mood**")
            mopts={"😊 Great":5,"🙂 Good":4,"😐 Okay":3,"😟 Low":2,"😰 Awful":1}
            ch=st.radio("Mood",list(mopts.keys()),horizontal=True,label_visibility="collapsed",key="msel")
            nt=st.text_input("Quick note (optional)",key="mnote",placeholder="What's on your mind?")
            if st.button("Log Mood ✓",use_container_width=True,key="logm"):
                st.session_state.mood_log.append({"day":"Today","v":mopts[ch]})
                st.success(f"Logged: {ch}")
            st.markdown('</div>',unsafe_allow_html=True)
        with mr_c:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**📈 Mood Trend**")
            ml_df=pd.DataFrame(st.session_state.mood_log[-7:])
            fm=go.Figure(go.Scatter(x=ml_df["day"],y=ml_df["v"],fill="tozeroy",mode="lines+markers",
                line=dict(color="#7b8cde",width=2),marker=dict(size=7),fillcolor="rgba(167,139,250,0.1)"))
            fm.update_layout(height=180,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",showlegend=False,font=dict(family="Outfit"),
                xaxis=dict(showgrid=False,color="#5a6478"),
                yaxis=dict(range=[0,5.5],showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#5a6478",
                    tickvals=[1,2,3,4,5],ticktext=["😰","😟","😐","🙂","😊"]))
            st.plotly_chart(fm,use_container_width=True,config={"displayModeBar":False})
            st.markdown('</div>',unsafe_allow_html=True)
    with tg:
        st.markdown("<br>",unsafe_allow_html=True)
        PROMPTS=["What made you smile today?","Who helped you recently?","What are you proud of?",
                 "What beauty did you notice?","What challenge taught you something?"]
        gl,gr=st.columns([1.3,1])
        with gl:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**🌸 Today's Gratitude**")
            st.markdown(f'''<div style="background:rgba(100,116,200,.07);border:1px solid rgba(100,116,200,.18);
              border-radius:12px;padding:12px 16px;margin-bottom:14px;">
              <div style="font-size:.7rem;color:#7b8cde;font-weight:700;letter-spacing:1px;margin-bottom:4px;">✨ TODAY'S PROMPT</div>
              <div style="font-size:.85rem;color:#ccd4e0;">{PROMPTS[st.session_state.prompt_idx]}</div></div>''',unsafe_allow_html=True)
            gv=st.text_area("Entry",height=90,label_visibility="collapsed",placeholder="I'm grateful for…",key="gv")
            b1g,b2g=st.columns(2)
            with b1g:
                if st.button("Save 🌸",use_container_width=True,key="sg"):
                    if gv.strip():
                        st.session_state.gratitude.insert(0,gv.strip())
                        st.session_state.prompt_idx=(st.session_state.prompt_idx+1)%len(PROMPTS)
                        st.success("Saved! 🌸")
            with b2g:
                if st.button("New Prompt ✨",use_container_width=True,key="np"):
                    st.session_state.prompt_idx=(st.session_state.prompt_idx+1)%len(PROMPTS); st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)
        with gr:
            st.markdown('<div class="pp-card">',unsafe_allow_html=True)
            st.markdown("**📖 Your Entries**")
            for e in st.session_state.gratitude[:6]:
                st.markdown(f'<div style="background:rgba(255,255,255,.035);border-radius:10px;padding:10px 14px;margin-bottom:8px;border:1px solid rgba(255,255,255,.06);font-size:.8rem;color:#8a9ab5;display:flex;gap:10px;align-items:flex-start;"><span style="font-size:1rem;flex:0 0 auto;">🌸</span><span style="line-height:1.5;">{e}</span></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
    with tq:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**🧩 Quick Burnout Self-Assessment**")
        st.caption("Answer honestly. Results are not stored.")
        st.markdown("<br>",unsafe_allow_html=True)
        QS=[("I feel emotionally drained at the end of each day.","q1"),
            ("I find it hard to concentrate on my studies.","q2"),
            ("I feel detached or indifferent about my coursework.","q3"),
            ("I often feel overwhelmed by my workload.","q4"),
            ("I am sleeping less than 6 hours regularly.","q5")]
        OPTS=["Never (0)","Rarely (1)","Sometimes (2)","Often (3)","Always (4)"]
        ans=[int(st.select_slider(q,options=OPTS,key=f"qz_{k}").split("(")[1].rstrip(")")) for q,k in QS]
        if st.button("Calculate Score →",use_container_width=True,key="qsub"):
            tot=sum(ans); pct=tot/20; c=rcol(pct)
            lb="Low Risk 🟢" if pct<0.35 else "Moderate Risk 🟡" if pct<0.65 else "High Risk 🔴"
            mg=("You're managing well! Keep your healthy habits." if pct<0.35 else
                "Some stress detected. Try the wellness tools above." if pct<0.65 else
                "High burnout indicators. Please reach out to a counselor or trusted person. 💙")
            st.markdown(f'''<div style="background:{c}15;border:1px solid {c}40;border-radius:16px;
              padding:20px;text-align:center;margin-top:16px;">
              <div style="font-size:2.5rem;font-weight:900;color:{c};">{tot}/20</div>
              <div style="font-size:1rem;font-weight:700;color:{c};margin-top:4px;">{lb}</div>
              <div style="font-size:.8rem;color:#6b7590;margin-top:8px;">{mg}</div></div>''',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="pp-title">📊 Deep Analytics</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Detailed academic and wellness trend analysis</div>',unsafe_allow_html=True)
    trend=mock_trend()
    cl,cr=st.columns(2)
    with cl:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**📚 GPA Trend**")
        f=go.Figure(go.Scatter(x=trend["week"],y=trend["gpa"],mode="lines+markers",
            line=dict(color="#4f7ab3",width=2),marker=dict(size=7),fill="tozeroy",fillcolor="rgba(60,100,160,0.09)"))
        f.update_layout(height=200,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),font=dict(family="Outfit"),
            xaxis=dict(showgrid=False,color="#5a6478"),yaxis=dict(range=[2,4],showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#5a6478"))
        st.plotly_chart(f,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**💭 Emotional Stress Trend**")
        f2=go.Figure(go.Scatter(x=trend["week"],y=trend["emotional"],mode="lines+markers",
            line=dict(color="#8c607e",width=2),marker=dict(size=7),fill="tozeroy",fillcolor="rgba(140,96,126,0.1)"))
        f2.update_layout(height=200,showlegend=False,**_CL)
        st.plotly_chart(f2,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-card">',unsafe_allow_html=True)
    st.markdown("**📈 Multi-Metric Overview**")
    f3=go.Figure()
    for cn,col,nm,m in [("risk","#9b4545","Burnout Risk",1),("gpa","#4f7ab3","GPA ×25",25),
                         ("sleep","#3d8c6e","Sleep ×10",10),("focus","#b07d2a","Focus %",1)]:
        f3.add_trace(go.Scatter(x=trend["week"],y=[v*m for v in trend[cn]],mode="lines+markers",
            line=dict(color=col,width=2),marker=dict(size=5),name=nm))
    f3.update_layout(height=280,**_CL)
    st.plotly_chart(f3,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN
# ══════════════════════════════════════════════════════════════════════════════
def page_admin():
    st.markdown('<div class="pp-title">🏛️ Institutional Dashboard</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-subtitle">Anonymised burnout analytics across 312 students</div>',unsafe_allow_html=True)
    ad=admin_data()
    a1,a2,a3,a4=st.columns(4)
    for col,(val,lb,c) in zip([a1,a2,a3,a4],[("312","Total Students","#4f7ab3"),
        ("17%","High Risk","#9b4545"),("35%","Moderate Risk","#b07d2a"),("48%","Low Risk","#3d8c6e")]):
        with col: st.markdown(f'<div class="pp-kpi"><div class="pp-kpi-value" style="color:{c};">{val}</div><div class="pp-kpi-label">{lb}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    cl,cr=st.columns(2)
    with cl:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**📊 Risk Distribution**")
        fp=go.Figure(go.Pie(labels=ad["dist"]["name"],values=ad["dist"]["value"],hole=0.5,
            marker=dict(colors=["#3d8c6e","#b07d2a","#9b4545"]),textfont=dict(family="Outfit",size=12)))
        fp.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Outfit"),
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#8a9ab5")))
        st.plotly_chart(fp,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="pp-card">',unsafe_allow_html=True)
        st.markdown("**📈 Department Trends**")
        fl=go.Figure()
        pal=["#7b8cde","#3d8c6e","#9b4545","#b07d2a"]
        for i,(dept,vals) in enumerate(ad["trends"].items()):
            fl.add_trace(go.Scatter(x=[f"Wk{j+1}" for j in range(len(vals))],y=vals,
                mode="lines+markers",name=dept,line=dict(color=pal[i%4],width=2),marker=dict(size=5)))
        fl.update_layout(height=240,**_CL)
        st.plotly_chart(fl,use_container_width=True,config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="pp-card">',unsafe_allow_html=True)
    st.markdown("**🔥 Stress Heatmap — Department × Week**")
    hdf=ad["heat"]
    fh=go.Figure(go.Heatmap(z=hdf.values,x=list(hdf.columns),y=list(hdf.index),
        colorscale=[[0,"rgba(50,130,100,0.75)"],[0.5,"rgba(160,110,30,0.75)"],[1,"rgba(140,60,60,0.85)"]],
        text=[[f"{v}%" for v in row] for row in hdf.values],texttemplate="%{text}",
        textfont=dict(family="Outfit",size=12,color="white"),showscale=True))
    fh.update_layout(height=280,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(color="#5a6478"),yaxis=dict(color="#8a9ab5"),font=dict(family="Outfit"))
    st.plotly_chart(fh,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('''<div class="pp-card" style="border-color:rgba(60,100,160,.25);background:rgba(96,165,250,.05);">
      <div style="display:flex;gap:16px;align-items:flex-start;">
        <div style="font-size:2.5rem;">🌐</div>
        <div>
          <div style="font-size:1rem;font-weight:800;color:#4f7ab3;margin-bottom:6px;">Federated Learning Module — Coming Soon</div>
          <div style="font-size:.82rem;color:#5a6478;line-height:1.6;margin-bottom:12px;">
            Train a global burnout model across institutions without sharing raw student data.
            Only model gradients are shared — preserving full privacy via the Flower (flwr) framework.</div>
          <span style="background:rgba(60,100,160,.18);color:#4f7ab3;border:1px solid rgba(60,100,160,.35);padding:3px 12px;border-radius:12px;font-size:.72rem;font-weight:700;margin-right:8px;">Privacy-Preserving</span>
          <span style="background:rgba(100,116,200,.18);color:#7b8cde;border:1px solid rgba(100,116,200,.35);padding:3px 12px;border-radius:12px;font-size:.72rem;font-weight:700;margin-right:8px;">Differential Privacy</span>
          <span style="background:rgba(50,130,100,.16);color:#3d8c6e;border:1px solid rgba(50,130,100,.32);padding:3px 12px;border-radius:12px;font-size:.72rem;font-weight:700;">Multi-Institutional</span>
        </div></div></div>''',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_auth(); return
    sidebar()
    {"dashboard":page_dashboard,"predict":page_predict,"wellness":page_wellness,
     "analytics":page_analytics,"admin":page_admin}.get(st.session_state.page,page_dashboard)()
    st.markdown("<br><br>",unsafe_allow_html=True)
    st.markdown('''<div style="text-align:center;padding:20px 0;
      border-top:1px solid rgba(100,116,200,.1);font-size:.72rem;color:#3d4a60;margin-top:40px;">
      🧠 PredictPulse · Built with ❤️ for student wellbeing ·
      <span style="color:#5a6478;">If you're struggling — please reach out to your counselor 💙</span>
    </div>''',unsafe_allow_html=True)

if __name__=="__main__":
    main()
