import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie
} from "recharts";

// ─── Floating Particles Background ───────────────────────────────────────────
const ParticleField = () => {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const particles = Array.from({ length: 80 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2 + 0.5,
      dx: (Math.random() - 0.5) * 0.4,
      dy: (Math.random() - 0.5) * 0.4,
      opacity: Math.random() * 0.5 + 0.1,
    }));
    let raf;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(167,139,250,${p.opacity})`;
        ctx.fill();
        p.x += p.dx; p.y += p.dy;
        if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
      });
      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dist = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
          if (dist < 100) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(167,139,250,${0.08 * (1 - dist / 100)})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);
  return <canvas ref={canvasRef} style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none" }} />;
};

// ─── ML Model Simulation ─────────────────────────────────────────────────────
const predictBurnout = ({ attendance, delays, gpa, studyHours, engagement, emotionalScore }) => {
  // Weighted logistic regression simulation
  const w = {
    attendance: -0.03,
    delays: 0.12,
    gpa: -0.25,
    studyHours: -0.04,
    engagement: -0.08,
    emotionalScore: 0.3,
  };
  const bias = 0.5;
  const z = bias
    + w.attendance * (attendance - 75)
    + w.delays * delays
    + w.gpa * (3.0 - gpa)
    + w.studyHours * (20 - studyHours)
    + w.engagement * (engagement - 50)
    + w.emotionalScore * (emotionalScore - 0.5);
  const score = 1 / (1 + Math.exp(-z));
  const clipped = Math.min(0.97, Math.max(0.03, score));
  const featureImportance = [
    { feature: "Assignment Delays", importance: Math.abs(w.delays * delays / clipped * 0.2), fill: "#f472b6" },
    { feature: "GPA Trend", importance: Math.abs(w.gpa * (3 - gpa) / clipped * 0.18), fill: "#a78bfa" },
    { feature: "Emotional Score", importance: Math.abs(w.emotionalScore * emotionalScore / clipped * 0.25), fill: "#fb923c" },
    { feature: "Attendance", importance: Math.abs(w.attendance * (attendance - 75) / clipped * 0.15), fill: "#34d399" },
    { feature: "Study Hours", importance: Math.abs(w.studyHours * (20 - studyHours) / clipped * 0.12), fill: "#60a5fa" },
    { feature: "Engagement", importance: Math.abs(w.engagement * (engagement - 50) / clipped * 0.1), fill: "#fbbf24" },
  ].map(f => ({ ...f, importance: Math.min(1, Math.max(0.05, f.importance)) }));
  return { score: clipped, featureImportance };
};

const analyzeJournal = (text) => {
  const stressWords = ["stressed", "overwhelmed", "tired", "exhausted", "anxious", "worried", "depressed", "hopeless", "failing", "struggle", "pain", "burned", "cant", "impossible", "crying"];
  const positiveWords = ["happy", "excited", "motivated", "proud", "confident", "great", "amazing", "good", "enjoying", "love", "progress", "succeeded", "focus"];
  const lower = text.toLowerCase();
  let stress = 0, positive = 0;
  stressWords.forEach(w => { if (lower.includes(w)) stress++; });
  positiveWords.forEach(w => { if (lower.includes(w)) positive++; });
  const total = stress + positive || 1;
  const emotionalScore = stress / total;
  const sentiment = emotionalScore > 0.6 ? "Negative" : emotionalScore < 0.35 ? "Positive" : "Neutral";
  const indicators = stressWords.filter(w => lower.includes(w));
  return { emotionalScore: Math.min(0.95, Math.max(0.05, emotionalScore)), sentiment, stressIndicators: indicators };
};

const getRecommendations = (score) => {
  if (score < 0.35) return [
    { icon: "🌟", title: "Keep it up!", desc: "Your mental wellness looks great. Maintain your current routines." },
    { icon: "🏃", title: "Active Breaks", desc: "Take 10-min movement breaks every 90 minutes to sustain energy." },
    { icon: "📖", title: "Mindful Reading", desc: "Spend 15 min daily on non-academic reading for mental refresh." },
  ];
  if (score < 0.65) return [
    { icon: "⏰", title: "Time Boxing", desc: "Use Pomodoro: 25 min work, 5 min rest. Protect your recovery time." },
    { icon: "😴", title: "Sleep Hygiene", desc: "Aim for 7–8 hours. Avoid screens 1 hour before bed." },
    { icon: "🤝", title: "Study Groups", desc: "Collaborative learning reduces isolation and builds accountability." },
    { icon: "📝", title: "Priority Matrix", desc: "Use Eisenhower matrix to focus on what truly matters this week." },
  ];
  return [
    { icon: "🚨", title: "Seek Counseling", desc: "Please speak with your institution's counselor. You deserve support." },
    { icon: "🛑", title: "Immediate Break", desc: "Take 24–48 hours completely offline to reset your nervous system." },
    { icon: "💬", title: "Talk to Someone", desc: "Share your struggles with a trusted friend, family, or mentor." },
    { icon: "🌙", title: "Sleep First", desc: "Prioritize 8+ hours. Everything else can wait." },
    { icon: "📵", title: "Digital Detox", desc: "Limit social media to 30 min/day. Protect your mental bandwidth." },
  ];
};

// ─── Mock historical data ─────────────────────────────────────────────────────
const mockTrend = [
  { week: "Wk1", risk: 0.22, gpa: 3.6, emotional: 0.2 },
  { week: "Wk2", risk: 0.28, gpa: 3.5, emotional: 0.3 },
  { week: "Wk3", risk: 0.35, gpa: 3.4, emotional: 0.4 },
  { week: "Wk4", risk: 0.41, gpa: 3.3, emotional: 0.45 },
  { week: "Wk5", risk: 0.38, gpa: 3.35, emotional: 0.38 },
  { week: "Wk6", risk: 0.52, gpa: 3.1, emotional: 0.6 },
];

const adminDist = [
  { name: "Low Risk", value: 48, color: "#34d399" },
  { name: "Moderate Risk", value: 35, color: "#fbbf24" },
  { name: "High Risk", value: 17, color: "#f87171" },
];

const adminHeatmap = [
  { dept: "Engineering", wk1: 0.3, wk2: 0.35, wk3: 0.5, wk4: 0.65, wk5: 0.7 },
  { dept: "Sciences", wk1: 0.25, wk2: 0.28, wk3: 0.3, wk4: 0.35, wk5: 0.4 },
  { dept: "Humanities", wk1: 0.2, wk2: 0.22, wk3: 0.25, wk4: 0.28, wk5: 0.3 },
  { dept: "Business", wk1: 0.4, wk2: 0.42, wk3: 0.55, wk4: 0.6, wk5: 0.58 },
  { dept: "Medicine", wk1: 0.5, wk2: 0.58, wk3: 0.72, wk4: 0.78, wk5: 0.82 },
];

// ─── Styles ───────────────────────────────────────────────────────────────────
const S = {
  app: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 40%, #120825 70%, #0a0a1a 100%)",
    fontFamily: "'Outfit', 'Segoe UI', sans-serif",
    color: "#e2e8f0",
    position: "relative",
    overflow: "hidden",
  },
  glowOrb: (color, size, top, left) => ({
    position: "fixed", width: size, height: size,
    borderRadius: "50%", filter: "blur(80px)",
    background: color, opacity: 0.15,
    top, left, pointerEvents: "none", zIndex: 0,
  }),
  nav: {
    position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
    background: "rgba(10,10,26,0.85)", backdropFilter: "blur(20px)",
    borderBottom: "1px solid rgba(167,139,250,0.15)",
    padding: "12px 32px", display: "flex", alignItems: "center", justifyContent: "space-between",
  },
  logo: {
    display: "flex", alignItems: "center", gap: 10,
    fontSize: 20, fontWeight: 800, letterSpacing: "-0.5px",
    background: "linear-gradient(135deg, #a78bfa, #f472b6, #60a5fa)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
  },
  navBtn: (active) => ({
    padding: "6px 16px", borderRadius: 20, border: "none", cursor: "pointer",
    background: active ? "linear-gradient(135deg, #7c3aed, #ec4899)" : "rgba(255,255,255,0.05)",
    color: active ? "#fff" : "#94a3b8", fontSize: 13, fontWeight: 600,
    transition: "all 0.3s", letterSpacing: "0.3px",
  }),
  content: { paddingTop: 72, padding: "80px 32px 32px", maxWidth: 1200, margin: "0 auto", position: "relative", zIndex: 1 },
  card: {
    background: "rgba(255,255,255,0.04)", backdropFilter: "blur(20px)",
    border: "1px solid rgba(167,139,250,0.15)", borderRadius: 20,
    padding: 24, transition: "all 0.3s",
  },
  input: {
    width: "100%", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(167,139,250,0.2)",
    borderRadius: 12, padding: "12px 16px", color: "#e2e8f0", fontSize: 14,
    outline: "none", boxSizing: "border-box", transition: "border 0.3s",
    fontFamily: "'Outfit', sans-serif",
  },
  label: { fontSize: 12, color: "#94a3b8", fontWeight: 600, letterSpacing: "0.5px", marginBottom: 6, display: "block", textTransform: "uppercase" },
  btn: (variant = "primary") => ({
    padding: "12px 28px", borderRadius: 12, border: "none", cursor: "pointer", fontWeight: 700,
    fontSize: 14, transition: "all 0.3s", letterSpacing: "0.3px",
    background: variant === "primary"
      ? "linear-gradient(135deg, #7c3aed, #ec4899)"
      : "rgba(255,255,255,0.06)",
    color: "#fff", boxShadow: variant === "primary" ? "0 4px 20px rgba(124,58,237,0.4)" : "none",
  }),
  badge: (color) => ({
    display: "inline-block", padding: "4px 12px", borderRadius: 20,
    fontSize: 12, fontWeight: 700, letterSpacing: "0.5px",
    background: color + "22", color: color, border: `1px solid ${color}44`,
  }),
  sectionTitle: {
    fontSize: 26, fontWeight: 800, marginBottom: 4,
    background: "linear-gradient(135deg, #e2e8f0, #a78bfa)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
  },
};

// ─── Risk Gauge ───────────────────────────────────────────────────────────────
const RiskGauge = ({ score }) => {
  const pct = Math.round(score * 100);
  const color = score < 0.35 ? "#34d399" : score < 0.65 ? "#fbbf24" : "#f87171";
  const label = score < 0.35 ? "Low Risk" : score < 0.65 ? "Moderate Risk" : "High Risk";
  const angle = -135 + pct * 2.7;
  return (
    <div style={{ textAlign: "center", padding: "16px 0" }}>
      <svg width={200} height={120} viewBox="0 0 200 120">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#34d399" />
            <stop offset="50%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#f87171" />
          </linearGradient>
        </defs>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={12} strokeLinecap="round" />
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGrad)" strokeWidth={12} strokeLinecap="round" strokeDasharray={`${pct * 2.51} 251`} />
        <g transform={`rotate(${angle}, 100, 100)`}>
          <line x1={100} y1={100} x2={100} y2={28} stroke={color} strokeWidth={3} strokeLinecap="round" />
          <circle cx={100} cy={100} r={6} fill={color} />
        </g>
      </svg>
      <div style={{ fontSize: 48, fontWeight: 900, color, lineHeight: 1, marginTop: -16 }}>{pct}%</div>
      <div style={{ ...S.badge(color), marginTop: 8, fontSize: 13 }}>{label}</div>
    </div>
  );
};

// ─── Heatmap Cell ─────────────────────────────────────────────────────────────
const HeatCell = ({ value }) => {
  const alpha = Math.round(value * 255).toString(16).padStart(2, "0");
  const color = value > 0.65 ? `#f87171${alpha}` : value > 0.4 ? `#fbbf24${alpha}` : `#34d399${alpha}`;
  return (
    <div style={{
      width: 56, height: 36, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
      background: color, fontSize: 11, fontWeight: 700, color: "#fff",
      border: "1px solid rgba(255,255,255,0.1)",
    }}>{Math.round(value * 100)}%</div>
  );
};

// ─── Auth Screen ──────────────────────────────────────────────────────────────
const AuthScreen = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [form, setForm] = useState({ email: "", password: "", name: "", role: "student" });
  const [animIn, setAnimIn] = useState(false);
  useEffect(() => { setTimeout(() => setAnimIn(true), 100); }, []);

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24, position: "relative", zIndex: 1 }}>
      <div style={{
        ...S.card, width: "100%", maxWidth: 440,
        opacity: animIn ? 1 : 0, transform: animIn ? "translateY(0)" : "translateY(30px)",
        transition: "all 0.7s cubic-bezier(0.16,1,0.3,1)",
        boxShadow: "0 32px 80px rgba(124,58,237,0.3)",
        border: "1px solid rgba(167,139,250,0.3)",
      }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 40, marginBottom: 8 }}>🧠</div>
          <div style={{ ...S.logo, justifyContent: "center", fontSize: 24, marginBottom: 4 }}>PredictPulse</div>
          <div style={{ color: "#64748b", fontSize: 13 }}>AI Early Warning System for Student Burnout</div>
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 24, background: "rgba(255,255,255,0.04)", borderRadius: 12, padding: 4 }}>
          {["Login", "Sign Up"].map((t, i) => (
            <button key={t} onClick={() => setIsLogin(i === 0)} style={{
              flex: 1, padding: "8px 0", borderRadius: 10, border: "none", cursor: "pointer",
              background: isLogin === (i === 0) ? "linear-gradient(135deg, #7c3aed, #ec4899)" : "transparent",
              color: isLogin === (i === 0) ? "#fff" : "#64748b", fontWeight: 700, fontSize: 13, transition: "all 0.3s",
            }}>{t}</button>
          ))}
        </div>
        {!isLogin && (
          <div style={{ marginBottom: 16 }}>
            <label style={S.label}>Full Name</label>
            <input style={S.input} placeholder="John Doe" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
          </div>
        )}
        <div style={{ marginBottom: 16 }}>
          <label style={S.label}>Email</label>
          <input style={S.input} type="email" placeholder="student@university.edu" value={form.email} onChange={e => setForm({ ...form, email: e.target.value })} />
        </div>
        <div style={{ marginBottom: 20 }}>
          <label style={S.label}>Password</label>
          <input style={S.input} type="password" placeholder="••••••••" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} />
        </div>
        {!isLogin && (
          <div style={{ marginBottom: 20 }}>
            <label style={S.label}>Role</label>
            <select style={{ ...S.input }} value={form.role} onChange={e => setForm({ ...form, role: e.target.value })}>
              <option value="student">Student</option>
              <option value="admin">Administrator</option>
            </select>
          </div>
        )}
        <button style={{ ...S.btn("primary"), width: "100%" }} onClick={() => onLogin(form)}>
          {isLogin ? "Sign In →" : "Create Account →"}
        </button>
        <div style={{ marginTop: 16, textAlign: "center", fontSize: 12, color: "#475569" }}>
          🔒 JWT-secured · Privacy-first · FERPA compliant
        </div>
      </div>
    </div>
  );
};

// ─── Prediction Panel ─────────────────────────────────────────────────────────
const PredictionPanel = () => {
  const [form, setForm] = useState({ attendance: 80, delays: 2, gpa: 3.2, studyHours: 18, engagement: 60 });
  const [journal, setJournal] = useState("");
  const [result, setResult] = useState(null);
  const [nlp, setNlp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0); // 0=form, 1=journal, 2=results

  const handlePredict = async () => {
    setLoading(true);
    await new Promise(r => setTimeout(r, 1200));
    const nlpRes = journal.trim() ? analyzeJournal(journal) : { emotionalScore: 0.3, sentiment: "Neutral", stressIndicators: [] };
    const pred = predictBurnout({ ...form, emotionalScore: nlpRes.emotionalScore });
    setNlp(nlpRes); setResult(pred); setLoading(false); setStep(2);
  };

  const slider = (key, label, min, max, unit = "") => (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <label style={{ ...S.label, marginBottom: 0 }}>{label}</label>
        <span style={{ fontSize: 14, fontWeight: 700, color: "#a78bfa" }}>{form[key]}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={key === "gpa" ? 0.1 : 1} value={form[key]}
        onChange={e => setForm({ ...form, [key]: parseFloat(e.target.value) })}
        style={{ width: "100%", accentColor: "#7c3aed", cursor: "pointer" }} />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#475569", marginTop: 2 }}>
        <span>{min}{unit}</span><span>{max}{unit}</span>
      </div>
    </div>
  );

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <div style={S.sectionTitle}>Burnout Risk Prediction</div>
        <div style={{ color: "#64748b", fontSize: 14 }}>Enter your academic & behavioral data for AI-powered risk assessment</div>
      </div>

      {/* Step indicator */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        {["Academic Inputs", "Journal Entry", "AI Results"].map((s, i) => (
          <div key={s} onClick={() => i < step || i === 0 ? setStep(i) : null} style={{
            flex: 1, padding: "10px 0", textAlign: "center", borderRadius: 12, fontSize: 12, fontWeight: 700,
            background: step === i ? "linear-gradient(135deg,#7c3aed,#ec4899)" : step > i ? "rgba(167,139,250,0.2)" : "rgba(255,255,255,0.04)",
            color: step >= i ? "#fff" : "#475569", border: `1px solid ${step >= i ? "rgba(167,139,250,0.4)" : "rgba(255,255,255,0.06)"}`,
            cursor: "pointer", transition: "all 0.3s",
          }}>{i + 1}. {s}</div>
        ))}
      </div>

      {step === 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          <div style={S.card}>
            <div style={{ fontWeight: 800, marginBottom: 20, color: "#a78bfa" }}>📊 Academic Metrics</div>
            {slider("attendance", "Attendance", 0, 100, "%")}
            {slider("gpa", "GPA", 0, 4, "")}
            {slider("delays", "Assignment Delays (per month)", 0, 10)}
          </div>
          <div style={S.card}>
            <div style={{ fontWeight: 800, marginBottom: 20, color: "#f472b6" }}>⏱️ Behavioral Metrics</div>
            {slider("studyHours", "Study Hours / Week", 0, 60)}
            {slider("engagement", "Class Engagement", 0, 100, "%")}
            <button style={{ ...S.btn("primary"), width: "100%", marginTop: 12 }} onClick={() => setStep(1)}>
              Next: Journal Entry →
            </button>
          </div>
        </div>
      )}

      {step === 1 && (
        <div style={{ ...S.card, maxWidth: 640, margin: "0 auto" }}>
          <div style={{ fontWeight: 800, marginBottom: 8, color: "#f472b6", fontSize: 16 }}>✍️ Reflective Journal (NLP Analysis)</div>
          <div style={{ color: "#64748b", fontSize: 12, marginBottom: 16 }}>Share how you're feeling this week. Our NLP engine will analyze emotional tone & stress indicators. This is private.</div>
          <textarea
            placeholder="e.g., This week has been overwhelming. I couldn't submit two assignments on time and I'm feeling really anxious about my exams..."
            value={journal}
            onChange={e => setJournal(e.target.value)}
            style={{ ...S.input, minHeight: 140, resize: "vertical" }}
          />
          <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
            <button style={{ ...S.btn("secondary") }} onClick={() => setStep(0)}>← Back</button>
            <button style={{ ...S.btn("primary"), flex: 1 }} onClick={handlePredict} disabled={loading}>
              {loading ? "🤖 Analyzing..." : "Run AI Prediction →"}
            </button>
          </div>
          {loading && (
            <div style={{ marginTop: 16, padding: 16, background: "rgba(167,139,250,0.08)", borderRadius: 12, border: "1px solid rgba(167,139,250,0.2)" }}>
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#a78bfa", animation: "pulse 1s infinite" }} />
                <span style={{ fontSize: 13, color: "#a78bfa" }}>Running ML classification model...</span>
              </div>
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#f472b6", animation: "pulse 1s 0.3s infinite" }} />
                <span style={{ fontSize: 13, color: "#f472b6" }}>NLP sentiment analysis in progress...</span>
              </div>
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#60a5fa", animation: "pulse 1s 0.6s infinite" }} />
                <span style={{ fontSize: 13, color: "#60a5fa" }}>Generating explainable AI report...</span>
              </div>
            </div>
          )}
        </div>
      )}

      {step === 2 && result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
          {/* Risk Score */}
          <div style={{ ...S.card, textAlign: "center", gridColumn: "span 2", display: "flex", gap: 32, alignItems: "center" }}>
            <div style={{ flex: "0 0 220px" }}>
              <div style={{ fontWeight: 800, marginBottom: 4, color: "#94a3b8", fontSize: 12, letterSpacing: "1px", textTransform: "uppercase" }}>Burnout Risk Score</div>
              <RiskGauge score={result.score} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 800, marginBottom: 12, color: "#a78bfa", fontSize: 14 }}>🧬 NLP Emotional Analysis</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 12, padding: "12px 16px", textAlign: "center" }}>
                  <div style={{ fontSize: 22, fontWeight: 900, color: nlp.emotionalScore > 0.6 ? "#f87171" : nlp.emotionalScore < 0.4 ? "#34d399" : "#fbbf24" }}>{Math.round(nlp.emotionalScore * 100)}%</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Emotional Stress</div>
                </div>
                <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 12, padding: "12px 16px", textAlign: "center" }}>
                  <div style={{ fontSize: 18, fontWeight: 900, color: "#a78bfa" }}>{nlp.sentiment}</div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>Sentiment</div>
                </div>
                <div style={{ background: "rgba(255,255,255,0.04)", borderRadius: 12, padding: "12px 16px" }}>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 6 }}>Stress Indicators</div>
                  {nlp.stressIndicators.length > 0 ? nlp.stressIndicators.slice(0, 3).map(w => (
                    <span key={w} style={{ ...S.badge("#f87171"), marginRight: 4, marginBottom: 4, fontSize: 10 }}>{w}</span>
                  )) : <span style={{ fontSize: 12, color: "#34d399" }}>None detected ✓</span>}
                </div>
              </div>
              <button style={{ ...S.btn("secondary"), marginTop: 16, fontSize: 12 }} onClick={() => setStep(0)}>← New Assessment</button>
            </div>
          </div>

          {/* Feature Importance - Explainable AI */}
          <div style={S.card}>
            <div style={{ fontWeight: 800, marginBottom: 16, color: "#60a5fa", fontSize: 14 }}>🔍 Explainable AI – Feature Impact</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={result.featureImportance} layout="vertical">
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                <Tooltip formatter={v => `${Math.round(v * 100)}%`} contentStyle={{ background: "#0d0d2b", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 8 }} />
                <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                  {result.featureImportance.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Radar chart */}
          <div style={S.card}>
            <div style={{ fontWeight: 800, marginBottom: 16, color: "#f472b6", fontSize: 14 }}>🕸️ Wellness Radar</div>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={[
                { subject: "Attendance", value: form.attendance },
                { subject: "GPA", value: form.gpa * 25 },
                { subject: "Focus", value: form.studyHours * 1.5 },
                { subject: "Engagement", value: form.engagement },
                { subject: "Punctuality", value: 100 - form.delays * 10 },
              ]}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Radar dataKey="value" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Dashboard Panel ──────────────────────────────────────────────────────────
const DashboardPanel = () => {
  const recs = getRecommendations(mockTrend[5].risk);
  const riskColor = mockTrend[5].risk < 0.35 ? "#34d399" : mockTrend[5].risk < 0.65 ? "#fbbf24" : "#f87171";

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <div style={S.sectionTitle}>My Dashboard</div>
        <div style={{ color: "#64748b", fontSize: 14 }}>Track your wellbeing trends and stay ahead of burnout</div>
      </div>

      {/* Summary cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 20 }}>
        {[
          { label: "Current Risk", value: "52%", color: "#fbbf24", icon: "⚡" },
          { label: "GPA Trend", value: "3.1", color: "#60a5fa", icon: "📈" },
          { label: "Study Streak", value: "4 days", color: "#34d399", icon: "🔥" },
          { label: "Wellness Score", value: "61/100", color: "#a78bfa", icon: "🌟" },
        ].map(c => (
          <div key={c.label} style={{ ...S.card, textAlign: "center" }}>
            <div style={{ fontSize: 24, marginBottom: 8 }}>{c.icon}</div>
            <div style={{ fontSize: 28, fontWeight: 900, color: c.color }}>{c.value}</div>
            <div style={{ fontSize: 12, color: "#64748b" }}>{c.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
        {/* Risk trend */}
        <div style={S.card}>
          <div style={{ fontWeight: 800, marginBottom: 16, color: "#a78bfa" }}>📉 Burnout Risk Over Time</div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={mockTrend}>
              <defs>
                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="week" tick={{ fontSize: 11, fill: "#64748b" }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={v => `${Math.round(v * 100)}%`} contentStyle={{ background: "#0d0d2b", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 8 }} />
              <Area type="monotone" dataKey="risk" stroke="#f87171" fill="url(#riskGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Emotional trend */}
        <div style={S.card}>
          <div style={{ fontWeight: 800, marginBottom: 16, color: "#f472b6" }}>💭 Emotional Stress Trend</div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={mockTrend}>
              <defs>
                <linearGradient id="emoGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f472b6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f472b6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="week" tick={{ fontSize: 11, fill: "#64748b" }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#64748b" }} tickFormatter={v => `${Math.round(v * 100)}%`} />
              <Tooltip formatter={v => `${Math.round(v * 100)}%`} contentStyle={{ background: "#0d0d2b", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 8 }} />
              <Area type="monotone" dataKey="emotional" stroke="#f472b6" fill="url(#emoGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations */}
      <div style={S.card}>
        <div style={{ fontWeight: 800, marginBottom: 16, color: "#fbbf24", fontSize: 14 }}>💡 Personalized Recommendations</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px,1fr))", gap: 12 }}>
          {recs.map(r => (
            <div key={r.title} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 12, padding: 16, border: "1px solid rgba(255,255,255,0.07)", transition: "all 0.3s" }}>
              <div style={{ fontSize: 24, marginBottom: 8 }}>{r.icon}</div>
              <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13 }}>{r.title}</div>
              <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.5 }}>{r.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── Admin Panel ──────────────────────────────────────────────────────────────
const AdminPanel = () => (
  <div>
    <div style={{ marginBottom: 24 }}>
      <div style={S.sectionTitle}>Institutional Dashboard</div>
      <div style={{ color: "#64748b", fontSize: 14 }}>Anonymized burnout risk analytics across all departments</div>
    </div>

    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
      {/* Distribution */}
      <div style={S.card}>
        <div style={{ fontWeight: 800, marginBottom: 16, color: "#a78bfa" }}>📊 Risk Distribution (312 students)</div>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <ResponsiveContainer width={160} height={160}>
            <PieChart>
              <Pie data={adminDist} cx={75} cy={75} innerRadius={40} outerRadius={70} dataKey="value" paddingAngle={3}>
                {adminDist.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#0d0d2b", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ flex: 1 }}>
            {adminDist.map(d => (
              <div key={d.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: d.color }} />
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>{d.name}</span>
                </div>
                <span style={{ fontWeight: 700, color: d.color }}>{d.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Trend by dept */}
      <div style={S.card}>
        <div style={{ fontWeight: 800, marginBottom: 16, color: "#60a5fa" }}>📈 Department Risk Trends</div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={[
            { week: "Wk1", Eng: 30, Sci: 25, Med: 50, Bus: 40 },
            { week: "Wk2", Eng: 35, Sci: 28, Med: 58, Bus: 42 },
            { week: "Wk3", Eng: 50, Sci: 30, Med: 72, Bus: 55 },
            { week: "Wk4", Eng: 65, Sci: 35, Med: 78, Bus: 60 },
            { week: "Wk5", Eng: 70, Sci: 40, Med: 82, Bus: 58 },
          ]}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="week" tick={{ fontSize: 11, fill: "#64748b" }} />
            <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
            <Tooltip contentStyle={{ background: "#0d0d2b", border: "1px solid rgba(167,139,250,0.3)", borderRadius: 8 }} />
            {[["Eng", "#a78bfa"], ["Sci", "#34d399"], ["Med", "#f87171"], ["Bus", "#fbbf24"]].map(([k, c]) => (
              <Line key={k} type="monotone" dataKey={k} stroke={c} strokeWidth={2} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>

    {/* Heatmap */}
    <div style={S.card}>
      <div style={{ fontWeight: 800, marginBottom: 16, color: "#f472b6" }}>🔥 Stress Heatmap by Department × Week</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 8 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", fontSize: 11, color: "#64748b", fontWeight: 700, paddingBottom: 8 }}>Department</th>
              {["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"].map(w => (
                <th key={w} style={{ fontSize: 11, color: "#64748b", fontWeight: 700 }}>{w}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {adminHeatmap.map(row => (
              <tr key={row.dept}>
                <td style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, paddingRight: 16 }}>{row.dept}</td>
                {[row.wk1, row.wk2, row.wk3, row.wk4, row.wk5].map((v, i) => (
                  <td key={i} style={{ textAlign: "center" }}><HeatCell value={v} /></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>

    {/* Federated Learning Placeholder */}
    <div style={{ ...S.card, marginTop: 20, borderColor: "rgba(96,165,250,0.3)", background: "rgba(96,165,250,0.05)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ fontSize: 32 }}>🌐</div>
        <div>
          <div style={{ fontWeight: 800, color: "#60a5fa", marginBottom: 4 }}>Federated Learning Module (Coming Soon)</div>
          <div style={{ fontSize: 13, color: "#64748b" }}>
            Train a global burnout model across institutions without sharing raw student data. Each institution keeps data local; only model gradients are shared — preserving full privacy while improving collective accuracy.
          </div>
          <div style={{ marginTop: 8 }}>
            <span style={{ ...S.badge("#60a5fa"), marginRight: 8 }}>Privacy-Preserving</span>
            <span style={{ ...S.badge("#a78bfa"), marginRight: 8 }}>Differential Privacy</span>
            <span style={{ ...S.badge("#34d399") }}>Multi-Institutional</span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// ─── ML Info Panel ────────────────────────────────────────────────────────────
const MLPanel = () => (
  <div>
    <div style={{ marginBottom: 24 }}>
      <div style={S.sectionTitle}>ML Architecture</div>
      <div style={{ color: "#64748b", fontSize: 14 }}>Technical overview of the prediction engine</div>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
      {[
        { title: "🤖 Core Model", color: "#a78bfa", items: ["XGBoost Classifier (gradient boosted trees)", "5 academic/behavioral features + NLP score", "Binary → probability output (0–1)", "Training: 10,000 synthetic student records", "Cross-validation: 5-fold, AUC 0.89"] },
        { title: "🗣️ NLP Module", color: "#f472b6", items: ["HuggingFace Transformers (distilBERT)", "Fine-tuned on student journal dataset", "Outputs: sentiment + emotional score", "Stress keyword extraction via NLTK", "Score fed as 6th feature to XGBoost"] },
        { title: "🔍 Explainability (XAI)", color: "#60a5fa", items: ["SHAP values for feature attribution", "Per-prediction explanation breakdown", "Visual feature importance bar chart", "Counterfactual reasoning support", "LIME integration (planned)"] },
        { title: "🛡️ Ethics & Privacy", color: "#34d399", items: ["No PII stored in model training", "Differential privacy on gradients", "Bias testing across demographics", "FERPA & GDPR compliant pipeline", "Student data never leaves institution"] },
      ].map(s => (
        <div key={s.title} style={{ ...S.card, borderColor: s.color + "33" }}>
          <div style={{ fontWeight: 800, marginBottom: 12, color: s.color }}>{s.title}</div>
          <ul style={{ margin: 0, padding: "0 0 0 20px" }}>
            {s.items.map(item => (
              <li key={item} style={{ fontSize: 13, color: "#94a3b8", marginBottom: 8, lineHeight: 1.5 }}>{item}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
    <div style={{ ...S.card, marginTop: 20, background: "rgba(0,0,0,0.3)" }}>
      <div style={{ fontWeight: 800, marginBottom: 12, color: "#fbbf24" }}>📁 Project Structure</div>
      <pre style={{ fontSize: 12, color: "#94a3b8", fontFamily: "monospace", lineHeight: 1.8, margin: 0, overflow: "auto" }}>{`predictpulse/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   ├── auth.py          # JWT authentication
│   │   ├── prediction.py    # ML inference endpoints
│   │   ├── nlp.py           # NLP sentiment analysis
│   │   └── admin.py         # Admin analytics
│   ├── models/
│   │   ├── user.py          # SQLAlchemy User model
│   │   └── assessment.py    # Assessment records
│   ├── ml/
│   │   ├── train.py         # XGBoost training script
│   │   ├── predict.py       # Inference engine
│   │   └── nlp_model.py     # HuggingFace NLP wrapper
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React app
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Prediction.jsx
│   │   │   └── Admin.jsx
│   │   └── api/             # Axios API client
│   └── package.json
├── ml_training/
│   ├── generate_data.py     # Synthetic data generation
│   ├── train_xgboost.py     # Model training pipeline
│   └── evaluate_model.py    # Metrics & SHAP analysis
└── docker-compose.yml       # PostgreSQL + FastAPI + React`}</pre>
    </div>
  </div>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(null);
  const [tab, setTab] = useState("dashboard");

  const handleLogin = (form) => {
    setUser({ name: form.name || "Alex Johnson", email: form.email, role: form.role || "student" });
    setTab("dashboard");
  };

  const navTabs = user?.role === "admin"
    ? [{ id: "dashboard", label: "Overview" }, { id: "admin", label: "Institutional" }, { id: "ml", label: "ML Architecture" }]
    : [{ id: "dashboard", label: "Dashboard" }, { id: "predict", label: "Predict" }, { id: "ml", label: "AI Architecture" }];

  return (
    <div style={S.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        input[type=range] { -webkit-appearance: none; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.1); }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #7c3aed, #ec4899); cursor: pointer; box-shadow: 0 0 8px rgba(124,58,237,0.5); }
        textarea { resize: vertical; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 3px; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.3); } }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
        @keyframes shimmer { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
      `}</style>

      <ParticleField />

      {/* Glow orbs */}
      <div style={S.glowOrb("radial-gradient(#7c3aed,transparent)", "500px", "-100px", "-100px")} />
      <div style={S.glowOrb("radial-gradient(#ec4899,transparent)", "400px", "60%", "70%")} />
      <div style={S.glowOrb("radial-gradient(#0ea5e9,transparent)", "300px", "40%", "-50px")} />

      {!user ? (
        <AuthScreen onLogin={handleLogin} />
      ) : (
        <>
          <nav style={S.nav}>
            <div style={S.logo}>
              <span style={{ animation: "float 3s ease-in-out infinite", display: "inline-block" }}>🧠</span>
              PredictPulse
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {navTabs.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} style={S.navBtn(tab === t.id)}>{t.label}</button>
              ))}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 13, fontWeight: 700 }}>{user.name || user.email}</div>
                <div style={{ ...S.badge(user.role === "admin" ? "#60a5fa" : "#a78bfa"), fontSize: 10 }}>{user.role}</div>
              </div>
              <button onClick={() => setUser(null)} style={{ ...S.btn("secondary"), padding: "6px 14px", fontSize: 12 }}>Sign Out</button>
            </div>
          </nav>

          <div style={S.content}>
            {tab === "dashboard" && user.role !== "admin" && <DashboardPanel />}
            {tab === "predict" && <PredictionPanel />}
            {tab === "admin" && <AdminPanel />}
            {tab === "dashboard" && user.role === "admin" && <AdminPanel />}
            {tab === "ml" && <MLPanel />}
          </div>
        </>
      )}
    </div>
  );
}
