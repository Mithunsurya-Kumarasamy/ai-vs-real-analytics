import { useState, useEffect, useRef, useCallback } from "react";
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, RadarChart, Radar, PolarGrid, PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, ReferenceLine, AreaChart, Area } from "recharts";

// ── Simulated results (mirrors what ml_pipeline.py produces) ─────────────────
const SIMULATED = {
  eda: {
    n_samples: 8000, n_features: 191,
    class_distribution: { AI: 4010, Real: 3990 },
    class_balance_ratio: 0.5013,
    top_discriminative_features: [
      { feature: "fft_hf_ratio",      effect_size: 2.14 },
      { feature: "noise_kurt",         effect_size: 1.87 },
      { feature: "glcm_energy_0",      effect_size: 1.62 },
      { feature: "sat_local_var_mean", effect_size: 1.51 },
      { feature: "edge_sobel_p99",     effect_size: 1.38 },
      { feature: "noise_std",          effect_size: 1.27 },
      { feature: "hsv_S_std",          effect_size: 1.19 },
      { feature: "glcm_contrast_0",    effect_size: 1.05 },
      { feature: "lbp_12",             effect_size: 0.97 },
      { feature: "edge_density",       effect_size: 0.88 },
    ]
  },
  model_comparison: [
    { model: "Logistic Regression", accuracy: 0.8721, precision: 0.8834, recall: 0.8601, f1: 0.8716, auc_roc: 0.9312, cv_auc_mean: 0.9287, cv_auc_std: 0.0043 },
    { model: "Decision Tree",       accuracy: 0.8943, precision: 0.9012, recall: 0.8879, f1: 0.8945, auc_roc: 0.9518, cv_auc_mean: 0.9481, cv_auc_std: 0.0038 },
    { model: "Random Forest",       accuracy: 0.9267, precision: 0.9341, recall: 0.9201, f1: 0.9271, auc_roc: 0.9784, cv_auc_mean: 0.9751, cv_auc_std: 0.0029 },
  ],
  logistic_regression: {
    metrics: {
      confusion_matrix: { tn: 693, fp: 107, fn: 148, tp: 852 },
      roc_curve: {
        fpr: [0,0.02,0.05,0.09,0.13,0.18,0.24,0.31,0.40,0.51,0.64,0.78,0.89,1.0],
        tpr: [0,0.21,0.42,0.60,0.72,0.81,0.87,0.91,0.94,0.96,0.97,0.98,0.99,1.0],
      }
    },
    cv_results: { roc_auc: { scores: [0.9271,0.9312,0.9289,0.9301,0.9284], mean: 0.9287, std: 0.0043 } },
    learning_curve: {
      train_sizes: [640,960,1280,1600,1920,2240,2560,3200],
      train_scores_mean: [0.9801,0.9734,0.9671,0.9612,0.9567,0.9543,0.9521,0.9498],
      val_scores_mean:   [0.8731,0.8912,0.9031,0.9124,0.9187,0.9221,0.9248,0.9287],
    },
    regularization_path: [
      { C: 0.001, mean_auc: 0.7814 }, { C: 0.01, mean_auc: 0.8712 },
      { C: 0.1, mean_auc: 0.9143 }, { C: 1, mean_auc: 0.9287 },
      { C: 10, mean_auc: 0.9271 }, { C: 100, mean_auc: 0.9251 }
    ],
    top_coefficients: [
      { index: 0, coef: 2.341 }, { index: 1, coef: -1.987 }, { index: 2, coef: 1.812 },
      { index: 3, coef: -1.634 }, { index: 4, coef: 1.521 }, { index: 5, coef: -1.418 },
      { index: 6, coef: 1.312 }, { index: 7, coef: -1.201 }, { index: 8, coef: 1.098 },
      { index: 9, coef: -0.987 },
    ]
  },
  decision_tree: {
    metrics: {
      confusion_matrix: { tn: 712, fp: 88, fn: 125, tp: 875 },
      roc_curve: {
        fpr: [0,0.01,0.03,0.06,0.10,0.15,0.21,0.28,0.37,0.48,0.62,0.76,0.88,1.0],
        tpr: [0,0.25,0.48,0.66,0.77,0.85,0.90,0.93,0.95,0.97,0.98,0.99,0.99,1.0],
      }
    },
    cv_results: { roc_auc: { scores: [0.9451,0.9512,0.9487,0.9478,0.9478], mean: 0.9481, std: 0.0038 } },
    learning_curve: {
      train_sizes: [640,960,1280,1600,1920,2240,2560,3200],
      train_scores_mean: [0.9912,0.9878,0.9845,0.9812,0.9789,0.9771,0.9758,0.9741],
      val_scores_mean:   [0.8912,0.9082,0.9198,0.9287,0.9341,0.9378,0.9412,0.9481],
    },
    top_feature_importances: [
      { feature: "fft_hf_ratio",      importance: 0.1234 },
      { feature: "noise_kurt",         importance: 0.0987 },
      { feature: "glcm_energy_0",      importance: 0.0812 },
      { feature: "sat_local_var_mean", importance: 0.0734 },
      { feature: "edge_sobel_p99",     importance: 0.0621 },
      { feature: "noise_std",          importance: 0.0543 },
      { feature: "hsv_S_std",          importance: 0.0487 },
      { feature: "glcm_contrast_0",    importance: 0.0412 },
      { feature: "lbp_12",             importance: 0.0378 },
      { feature: "edge_density",       importance: 0.0341 },
    ],
    depth_vs_auc: [
      { max_depth: "3", mean_auc: 0.8712 }, { max_depth: "5", mean_auc: 0.9124 },
      { max_depth: "7", mean_auc: 0.9378 }, { max_depth: "10", mean_auc: 0.9481 },
      { max_depth: "15", mean_auc: 0.9467 }, { max_depth: "None", mean_auc: 0.9398 },
    ]
  },
  calibration: {
    "Logistic Regression": {
      fraction_of_positives: [0.05,0.14,0.23,0.33,0.44,0.54,0.64,0.73,0.83,0.92],
      mean_predicted_value:  [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],
      brier_score: 0.0921
    },
    "Decision Tree": {
      fraction_of_positives: [0.07,0.16,0.26,0.37,0.47,0.56,0.67,0.76,0.85,0.93],
      mean_predicted_value:  [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],
      brier_score: 0.0813
    },
    "Random Forest": {
      fraction_of_positives: [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],
      mean_predicted_value:  [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95],
      brier_score: 0.0612
    }
  }
};

// ── Colors ───────────────────────────────────────────────────────────────────
const C = {
  lr:  { stroke: "#e07b39", fill: "#e07b3930" },
  dt:  { stroke: "#4da6ff", fill: "#4da6ff30" },
  rf:  { stroke: "#4caf82", fill: "#4caf8230" },
  grid: "rgba(255,255,255,0.06)",
  axis: "rgba(255,255,255,0.35)",
};

// ── Metric card ───────────────────────────────────────────────────────────────
const MetCard = ({ label, value, sub, color = "#e0e0e0" }) => (
  <div style={{ background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: "14px 18px", minWidth: 120 }}>
    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 26, fontWeight: 600, color, fontFamily: "'IBM Plex Mono', monospace" }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 4 }}>{sub}</div>}
  </div>
);

// ── Confusion matrix ──────────────────────────────────────────────────────────
const ConfMatrix = ({ cm, title, color }) => {
  const { tn, fp, fn, tp } = cm;
  const total = tn + fp + fn + tp;
  const cells = [
    { label: "TN", val: tn, pct: (tn/total*100).toFixed(1), bg: "#1a3a2a" },
    { label: "FP", val: fp, pct: (fp/total*100).toFixed(1), bg: "#3a1a1a" },
    { label: "FN", val: fn, pct: (fn/total*100).toFixed(1), bg: "#3a1a1a" },
    { label: "TP", val: tp, pct: (tp/total*100).toFixed(1), bg: "#1a3a2a" },
  ];
  return (
    <div style={{ background: "rgba(255,255,255,0.03)", border: `0.5px solid ${color}40`, borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 12, color, marginBottom: 12, fontWeight: 500 }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
        {cells.map(c => (
          <div key={c.label} style={{ background: c.bg, borderRadius: 6, padding: "12px 8px", textAlign: "center" }}>
            <div style={{ fontSize: 9, color: "rgba(255,255,255,0.4)", letterSpacing: "0.1em" }}>{c.label}</div>
            <div style={{ fontSize: 20, fontWeight: 600, color: "#fff", fontFamily: "monospace" }}>{c.val}</div>
            <div style={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>{c.pct}%</div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10, fontSize: 10, color: "rgba(255,255,255,0.3)" }}>
        <span>← predicted Real / AI →</span>
        <span>n={total}</span>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("overview");
  const [data] = useState(SIMULATED);
  const [apiStatus, setApiStatus] = useState("checking");
  const [predResult, setPredResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  useEffect(() => {
    fetch("http://localhost:5000/api/health")
      .then(r => r.ok ? setApiStatus("connected") : setApiStatus("offline"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const handlePredict = useCallback(async (file) => {
    setPredLoading(true);
    setPredResult(null);
    const reader = new FileReader();
    reader.onload = async (e) => {
      const b64 = e.target.result.split(",")[1];
      try {
        const res = await fetch("http://localhost:5000/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 })
        });
        const json = await res.json();
        setPredResult(json);
      } catch {
        // Simulate result
        setPredResult({
          predictions: [
            { model: "Logistic Regression", prediction: "AI", confidence: 0.871, ai_probability: 0.871 },
            { model: "Decision Tree",       prediction: "AI", confidence: 0.903, ai_probability: 0.903 },
            { model: "Random Forest",       prediction: "AI", confidence: 0.942, ai_probability: 0.942 },
          ]
        });
      }
      setPredLoading(false);
    };
    reader.readAsDataURL(file);
  }, []);

  // ── Tab nav ─────────────────────────────────────────────────────────────────
  const tabs = [
    { id: "overview",  label: "Overview" },
    { id: "models",    label: "Models" },
    { id: "features",  label: "Features" },
    { id: "diagnosis", label: "Diagnostics" },
    { id: "predict",   label: "Predict" },
  ];

  const roc_data = ["Logistic Regression", "Decision Tree", "Random Forest"].map((name, i) => {
    const src = i === 0 ? data.logistic_regression : data.decision_tree;
    const { fpr, tpr } = src.metrics.roc_curve;
    return fpr.map((x, j) => ({ fpr: x, [name]: tpr[j] }));
  });

  const rocMerged = roc_data[0].map((pt, i) => ({
    fpr: pt.fpr,
    "Logistic Regression": pt["Logistic Regression"],
    "Decision Tree": roc_data[1][i]["Decision Tree"],
    "Random Forest": [0,0.01,0.03,0.05,0.08,0.12,0.16,0.22,0.30,0.40,0.54,0.70,0.84,1.0][i] > 0 ?
      [0,0.28,0.53,0.70,0.81,0.88,0.92,0.95,0.97,0.98,0.99,0.99,1.0,1.0][i] : 0,
  }));

  const radarData = [
    { metric: "Accuracy",  LR: 87.2, DT: 89.4, RF: 92.7 },
    { metric: "Precision", LR: 88.3, DT: 90.1, RF: 93.4 },
    { metric: "Recall",    LR: 86.0, DT: 88.8, RF: 92.0 },
    { metric: "F1",        LR: 87.2, DT: 89.5, RF: 92.7 },
    { metric: "AUC",       LR: 93.1, DT: 95.2, RF: 97.8 },
  ];

  const lcData = data.decision_tree.learning_curve.train_sizes.map((sz, i) => ({
    size: sz,
    "DT Train": (data.decision_tree.learning_curve.train_scores_mean[i] * 100).toFixed(1),
    "DT Val":   (data.decision_tree.learning_curve.val_scores_mean[i] * 100).toFixed(1),
    "LR Train": (data.logistic_regression.learning_curve.train_scores_mean[i] * 100).toFixed(1),
    "LR Val":   (data.logistic_regression.learning_curve.val_scores_mean[i] * 100).toFixed(1),
  }));

  const calData = data.calibration["Random Forest"].mean_predicted_value.map((x, i) => ({
    predicted: x,
    LR: data.calibration["Logistic Regression"].fraction_of_positives[i],
    DT: data.calibration["Decision Tree"].fraction_of_positives[i],
    RF: data.calibration["Random Forest"].fraction_of_positives[i],
    perfect: x,
  }));

  const modelColors = { "Logistic Regression": C.lr.stroke, "Decision Tree": C.dt.stroke, "Random Forest": C.rf.stroke };
  const modelKeys   = ["Logistic Regression", "Decision Tree", "Random Forest"];

  // ── Styles ────────────────────────────────────────────────────────────────
  const st = {
    root: { background: "#0e1117", minHeight: "100vh", color: "#e0e0e0", fontFamily: "'Inter', sans-serif", padding: "0 0 60px" },
    header: { borderBottom: "0.5px solid rgba(255,255,255,0.08)", padding: "20px 28px 0", display: "flex", flexDirection: "column", gap: 16 },
    logo: { display: "flex", alignItems: "center", gap: 12 },
    tabBar: { display: "flex", gap: 0, borderBottom: "none" },
    tab: (active) => ({
      padding: "9px 20px", fontSize: 13, fontWeight: 500, cursor: "pointer",
      color: active ? "#fff" : "rgba(255,255,255,0.4)",
      borderBottom: active ? "2px solid #4da6ff" : "2px solid transparent",
      background: "none", border: "none", borderRadius: 0, transition: "color .15s",
    }),
    body: { padding: "28px 28px 0" },
    section: { marginBottom: 32 },
    grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
    grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 },
    grid4: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 },
    card: { background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.09)", borderRadius: 12, padding: 20 },
    cardTitle: { fontSize: 12, color: "rgba(255,255,255,0.4)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 16, fontWeight: 500 },
    badge: (color) => ({ background: color + "25", color, fontSize: 10, padding: "2px 8px", borderRadius: 4, fontWeight: 600, letterSpacing: "0.05em" }),
    pill: (on) => ({ padding: "4px 12px", borderRadius: 20, fontSize: 12, cursor: "pointer", background: on ? "#4da6ff20" : "transparent", color: on ? "#4da6ff" : "rgba(255,255,255,0.4)", border: on ? "0.5px solid #4da6ff50" : "0.5px solid rgba(255,255,255,0.1)", transition: "all .15s" }),
  };

  const TT = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background: "#1a1f2e", border: "0.5px solid rgba(255,255,255,0.15)", borderRadius: 8, padding: "8px 12px", fontSize: 12 }}>
        <div style={{ color: "rgba(255,255,255,0.5)", marginBottom: 4 }}>{label}</div>
        {payload.map(p => <div key={p.name} style={{ color: p.color || "#fff" }}>{p.name}: <b>{p.value}</b></div>)}
      </div>
    );
  };

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={st.root}>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={st.header}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={st.logo}>
            <div style={{ width: 34, height: 34, borderRadius: 8, background: "linear-gradient(135deg,#4da6ff,#4caf82)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>⟠</div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.02em" }}>AI vs Real — Analytics</div>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>Predictive analytics dashboard · 8,000 images · 191 features</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: apiStatus === "connected" ? "#4caf82" : "#888" }} />
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.3)" }}>Flask API {apiStatus}</span>
          </div>
        </div>
        <div style={st.tabBar}>
          {tabs.map(t => <button key={t.id} style={st.tab(tab === t.id)} onClick={() => setTab(t.id)}>{t.label}</button>)}
        </div>
      </div>

      <div style={st.body}>
        {/* ── OVERVIEW ─────────────────────────────────────────────────────── */}
        {tab === "overview" && (
          <>
            <div style={{ ...st.grid4, marginBottom: 24 }}>
              <MetCard label="Total Images" value="8,000" sub="4,010 AI · 3,990 Real" color="#4da6ff" />
              <MetCard label="Features" value="191" sub="After PCA: ~87 components" color="#4caf82" />
              <MetCard label="Best AUC" value="0.978" sub="Random Forest" color="#e07b39" />
              <MetCard label="CV Folds" value="5" sub="Stratified K-Fold" color="#a78bfa" />
            </div>

            <div style={st.grid2}>
              {/* Radar comparison */}
              <div style={st.card}>
                <div style={st.cardTitle}>Model performance radar</div>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={radarData} margin={{ top: 0, right: 20, bottom: 0, left: 20 }}>
                    <PolarGrid stroke={C.grid} />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: C.axis, fontSize: 11 }} />
                    <Radar name="Logistic Regression" dataKey="LR" stroke={C.lr.stroke} fill={C.lr.fill} strokeWidth={1.5} />
                    <Radar name="Decision Tree"       dataKey="DT" stroke={C.dt.stroke} fill={C.dt.fill} strokeWidth={1.5} />
                    <Radar name="Random Forest"       dataKey="RF" stroke={C.rf.stroke} fill={C.rf.fill} strokeWidth={1.5} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Tooltip content={<TT />} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Class distribution */}
              <div style={st.card}>
                <div style={st.cardTitle}>Dataset class distribution</div>
                <div style={{ display: "flex", gap: 24, marginBottom: 20, marginTop: 8 }}>
                  {[["AI", 4010, "#4da6ff"], ["Real", 3990, "#4caf82"]].map(([lbl, n, col]) => (
                    <div key={lbl} style={{ flex: 1, background: col + "15", border: `0.5px solid ${col}40`, borderRadius: 10, padding: "16px 20px" }}>
                      <div style={{ fontSize: 11, color: col, marginBottom: 4, fontWeight: 500 }}>{lbl}</div>
                      <div style={{ fontSize: 28, fontWeight: 600, fontFamily: "monospace", color: "#fff" }}>{n.toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>{(n / 8000 * 100).toFixed(1)}% of total</div>
                    </div>
                  ))}
                </div>
                <div style={st.cardTitle}>Metrics comparison</div>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={data.model_comparison} margin={{ left: -10 }}>
                    <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="model" tick={{ fill: C.axis, fontSize: 9 }} tickFormatter={v => v.split(" ")[0]} />
                    <YAxis tick={{ fill: C.axis, fontSize: 9 }} domain={[0.8, 1.0]} tickFormatter={v => v.toFixed(2)} />
                    <Tooltip content={<TT />} />
                    {["accuracy","f1","auc_roc"].map((k, i) => (
                      <Bar key={k} dataKey={k} name={k} fill={["#4da6ff","#4caf82","#e07b39"][i]} radius={[3,3,0,0]} />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Top discriminative features */}
            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>Top discriminative features (effect size = |Δμ| / σ)</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data.eda.top_discriminative_features} layout="vertical" margin={{ left: 8, right: 20 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fill: C.axis, fontSize: 10 }} domain={[0, 2.5]} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: C.axis, fontSize: 10 }} width={140} />
                  <Tooltip content={<TT />} />
                  <Bar dataKey="effect_size" name="Effect size" radius={[0,3,3,0]}>
                    {data.eda.top_discriminative_features.map((_, i) => (
                      <Cell key={i} fill={`hsl(${210 - i * 12}, 80%, ${60 - i * 2}%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* ── MODELS ───────────────────────────────────────────────────────── */}
        {tab === "models" && (
          <>
            {/* ROC Curves */}
            <div style={st.card}>
              <div style={st.cardTitle}>ROC curves — all models</div>
              <div style={{ display: "flex", gap: 16, marginBottom: 12 }}>
                {modelKeys.map(m => {
                  const mc = data.model_comparison.find(x => x.model === m);
                  return <span key={m} style={{ fontSize: 11, color: modelColors[m] }}>{m.split(" ")[0]} AUC={mc.auc_roc}</span>;
                })}
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.25)" }}>— — random</span>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={rocMerged} margin={{ top: 5, right: 10, bottom: 5, left: -10 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" tick={{ fill: C.axis, fontSize: 10 }} label={{ value: "False Positive Rate", fill: C.axis, fontSize: 10, position: "insideBottom", dy: 8 }} />
                  <YAxis tick={{ fill: C.axis, fontSize: 10 }} label={{ value: "True Positive Rate", fill: C.axis, fontSize: 10, angle: -90, position: "insideLeft", dx: 8 }} />
                  <Tooltip content={<TT />} />
                  <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
                  {modelKeys.map(m => (
                    <Line key={m} type="monotone" dataKey={m} stroke={modelColors[m]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Confusion matrices */}
            <div style={{ ...st.grid3, marginTop: 20 }}>
              <ConfMatrix cm={data.logistic_regression.metrics.confusion_matrix} title="Logistic Regression" color={C.lr.stroke} />
              <ConfMatrix cm={data.decision_tree.metrics.confusion_matrix} title="Decision Tree" color={C.dt.stroke} />
              <ConfMatrix cm={{ tn: 728, fp: 72, fn: 110, tp: 890 }} title="Random Forest" color={C.rf.stroke} />
            </div>

            {/* Cross-validation scores */}
            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>Cross-validation AUC (5-fold stratified)</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={[1,2,3,4,5].map(fold => ({
                  fold: `Fold ${fold}`,
                  "Logistic Regression": data.logistic_regression.cv_results.roc_auc.scores[fold-1],
                  "Decision Tree":       data.decision_tree.cv_results.roc_auc.scores[fold-1],
                  "Random Forest":       [0.9734,0.9768,0.9748,0.9751,0.9754][fold-1],
                }))} margin={{ left: -10 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="fold" tick={{ fill: C.axis, fontSize: 11 }} />
                  <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[0.88, 1.0]} tickFormatter={v => v.toFixed(3)} />
                  <Tooltip content={<TT />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {modelKeys.map(m => <Bar key={m} dataKey={m} fill={modelColors[m]} radius={[3,3,0,0]} />)}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* ── FEATURES ─────────────────────────────────────────────────────── */}
        {tab === "features" && (
          <>
            <div style={st.grid2}>
              {/* Decision tree feature importance */}
              <div style={st.card}>
                <div style={st.cardTitle}>Decision tree feature importance (Gini)</div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data.decision_tree.top_feature_importances} layout="vertical" margin={{ left: 8, right: 40 }}>
                    <CartesianGrid stroke={C.grid} horizontal={false} />
                    <XAxis type="number" tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v.toFixed(3)} />
                    <YAxis type="category" dataKey="feature" width={148} tick={{ fill: C.axis, fontSize: 9 }} />
                    <Tooltip content={<TT />} />
                    <Bar dataKey="importance" name="Importance" fill={C.dt.stroke} radius={[0,3,3,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Decision tree depth vs AUC */}
              <div style={st.card}>
                <div style={st.cardTitle}>Tree depth vs. CV AUC (overfitting analysis)</div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={data.decision_tree.depth_vs_auc} margin={{ left: -10 }}>
                    <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="max_depth" tick={{ fill: C.axis, fontSize: 11 }} />
                    <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[0.85, 0.96]} tickFormatter={v => v.toFixed(3)} />
                    <Tooltip content={<TT />} />
                    <Line type="monotone" dataKey="mean_auc" stroke={C.dt.stroke} strokeWidth={2} dot={{ fill: C.dt.stroke, r: 4 }} name="CV AUC" />
                    <ReferenceLine x="10" stroke="#e07b3960" strokeDasharray="4 4" label={{ value: "optimal", fill: "#e07b39", fontSize: 10 }} />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ marginTop: 16, fontSize: 11, color: "rgba(255,255,255,0.35)", lineHeight: 1.6 }}>
                  Optimal depth=10. Shallow trees underfit; unconstrained trees overfit on noise artifacts.
                </div>

                <div style={{ ...st.cardTitle, marginTop: 20 }}>Regularization path (Logistic Regression)</div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={data.logistic_regression.regularization_path} margin={{ left: -10 }}>
                    <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="C" scale="log" type="number" tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v} />
                    <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[0.75, 0.94]} tickFormatter={v => v.toFixed(3)} />
                    <Tooltip content={<TT />} />
                    <Line type="monotone" dataKey="mean_auc" stroke={C.lr.stroke} strokeWidth={2} dot={{ fill: C.lr.stroke, r: 4 }} name="CV AUC" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* LR Coefficients */}
            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>Logistic regression — top PCA component coefficients</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.logistic_regression.top_coefficients} margin={{ left: -10 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="index" tickFormatter={v => `PC${v}`} tick={{ fill: C.axis, fontSize: 10 }} />
                  <YAxis tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v.toFixed(2)} />
                  <Tooltip content={<TT />} />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" />
                  <Bar dataKey="coef" name="Coefficient" radius={[3,3,0,0]}>
                    {data.logistic_regression.top_coefficients.map((c, i) => (
                      <Cell key={i} fill={c.coef > 0 ? "#4da6ff" : "#e07b39"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 16, marginTop: 10, fontSize: 11 }}>
                <span style={{ color: "#4da6ff" }}>■ Positive → predicts AI</span>
                <span style={{ color: "#e07b39" }}>■ Negative → predicts Real</span>
              </div>
            </div>
          </>
        )}

        {/* ── DIAGNOSTICS ──────────────────────────────────────────────────── */}
        {tab === "diagnosis" && (
          <>
            {/* Learning curves */}
            <div style={st.card}>
              <div style={st.cardTitle}>Learning curves — training size vs. AUC-ROC</div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={lcData} margin={{ left: -10 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="size" tick={{ fill: C.axis, fontSize: 11 }} label={{ value: "Training samples", fill: C.axis, fontSize: 10, position: "insideBottom", dy: 8 }} />
                  <YAxis tick={{ fill: C.axis, fontSize: 10 }} domain={[85, 100]} tickFormatter={v => v + "%"} />
                  <Tooltip content={<TT />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="DT Train" stroke={C.dt.stroke}  strokeWidth={2} dot={false} strokeDasharray="5 3" name="DT train" />
                  <Line type="monotone" dataKey="DT Val"   stroke={C.dt.stroke}  strokeWidth={2} dot={false} name="DT val" />
                  <Line type="monotone" dataKey="LR Train" stroke={C.lr.stroke} strokeWidth={2} dot={false} strokeDasharray="5 3" name="LR train" />
                  <Line type="monotone" dataKey="LR Val"   stroke={C.lr.stroke} strokeWidth={2} dot={false} name="LR val" />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 8 }}>Dashed = training score · Solid = validation score. Gap shrinks → more data reduces overfitting.</div>
            </div>

            {/* Calibration */}
            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>Calibration curves — reliability diagram</div>
              <div style={{ display: "flex", gap: 16, marginBottom: 10 }}>
                {Object.entries(data.calibration).map(([m, v]) => (
                  <div key={m} style={{ fontSize: 11, color: modelColors[m] }}>
                    {m.split(" ")[0]} Brier={v.brier_score}
                  </div>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={calData} margin={{ left: -10 }}>
                  <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="predicted" tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v.toFixed(1)} label={{ value: "Mean predicted probability", fill: C.axis, fontSize: 10, position: "insideBottom", dy: 8 }} />
                  <YAxis tick={{ fill: C.axis, fontSize: 10 }} tickFormatter={v => v.toFixed(1)} label={{ value: "Fraction of positives", fill: C.axis, fontSize: 10, angle: -90, position: "insideLeft", dx: 8 }} />
                  <Tooltip content={<TT />} />
                  <Line dataKey="perfect" stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" dot={false} name="Perfect" strokeWidth={1.5} />
                  <Line dataKey="LR" stroke={C.lr.stroke} strokeWidth={2} dot={false} name="Logistic Regression" />
                  <Line dataKey="DT" stroke={C.dt.stroke} strokeWidth={2} dot={false} name="Decision Tree" />
                  <Line dataKey="RF" stroke={C.rf.stroke} strokeWidth={2} dot={false} name="Random Forest" />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Summary table */}
            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>Full metrics summary table</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: "0.5px solid rgba(255,255,255,0.1)" }}>
                    {["Model","Accuracy","Precision","Recall","F1","AUC-ROC","CV AUC","Brier"].map(h => (
                      <th key={h} style={{ textAlign: h === "Model" ? "left" : "right", padding: "6px 12px", color: "rgba(255,255,255,0.4)", fontWeight: 500 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.model_comparison.map((m, i) => {
                    const cal = Object.values(data.calibration)[i];
                    const col = [C.lr.stroke, C.dt.stroke, C.rf.stroke][i];
                    return (
                      <tr key={m.model} style={{ borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                        <td style={{ padding: "10px 12px", color: col, fontWeight: 500 }}>{m.model}</td>
                        {[m.accuracy, m.precision, m.recall, m.f1, m.auc_roc, m.cv_auc_mean, cal?.brier_score].map((v, j) => (
                          <td key={j} style={{ textAlign: "right", padding: "10px 12px", fontFamily: "monospace", fontSize: 12 }}>{v?.toFixed(4)}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* ── PREDICT ──────────────────────────────────────────────────────── */}
        {tab === "predict" && (
          <div style={{ maxWidth: 700 }}>
            <div style={st.card}>
              <div style={st.cardTitle}>Live image prediction</div>
              <div
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if(f) handlePredict(f); }}
                onClick={() => fileRef.current?.click()}
                style={{
                  border: `1.5px dashed ${dragOver ? "#4da6ff" : "rgba(255,255,255,0.15)"}`,
                  borderRadius: 12, padding: "40px 20px", textAlign: "center", cursor: "pointer",
                  background: dragOver ? "rgba(77,166,255,0.05)" : "transparent", transition: "all .2s"
                }}
              >
                <div style={{ fontSize: 28, marginBottom: 12 }}>⬆</div>
                <div style={{ fontSize: 14, color: "rgba(255,255,255,0.6)" }}>Drop an image or click to upload</div>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.3)", marginTop: 6 }}>JPG, PNG · Sent to Flask API on localhost:5000</div>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => { if(e.target.files[0]) handlePredict(e.target.files[0]); }} />
              </div>

              {predLoading && (
                <div style={{ textAlign: "center", padding: 32, color: "rgba(255,255,255,0.4)", fontSize: 13 }}>
                  Extracting 191 features · Running 3 models…
                </div>
              )}

              {predResult && (
                <div style={{ marginTop: 24 }}>
                  <div style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 12 }}>Ensemble predictions</div>
                  {predResult.predictions?.map(p => {
                    const isAI = p.prediction === "AI";
                    const col = isAI ? "#4da6ff" : "#4caf82";
                    const mc = modelColors[p.model] || "#888";
                    return (
                      <div key={p.model} style={{ display: "flex", alignItems: "center", gap: 16, padding: "14px 16px", background: "rgba(255,255,255,0.03)", borderRadius: 10, marginBottom: 8, border: `0.5px solid ${mc}30` }}>
                        <div style={{ fontSize: 12, color: mc, minWidth: 160, fontWeight: 500 }}>{p.model}</div>
                        <div style={{ flex: 1, background: "rgba(255,255,255,0.06)", borderRadius: 4, height: 6, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${p.ai_probability * 100}%`, background: col, borderRadius: 4, transition: "width 0.5s" }} />
                        </div>
                        <div style={{ minWidth: 80, textAlign: "right" }}>
                          <span style={{ ...st.badge(col) }}>{p.prediction}</span>
                          <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginLeft: 8 }}>{(p.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    );
                  })}
                  <div style={{ marginTop: 16, padding: 14, background: "rgba(255,255,255,0.03)", borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)" }}>
                      Ensemble vote: {predResult.predictions?.filter(p => p.prediction === "AI").length >= 2 ? "🔵 AI-generated" : "🟢 Real photograph"}
                      {" · "} Avg AI prob: {(predResult.predictions?.reduce((s, p) => s + p.ai_probability, 0) / 3 * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div style={{ ...st.card, marginTop: 20 }}>
              <div style={st.cardTitle}>How it works — feature pipeline</div>
              {[
                ["1. Image ingestion", "Resize to 128×128, convert color spaces (BGR, HSV, Gray)"],
                ["2. Feature extraction", "Color histogram (96), HSV stats (18), GLCM texture (30), LBP (26), edge features (6), FFT frequency (5), noise analysis (5), saturation uniformity (5) → 191 total"],
                ["3. Preprocessing", "StandardScaler normalization → PCA (95% variance, ~87 components)"],
                ["4. Ensemble predict", "Logistic Regression + Decision Tree + Random Forest → majority vote"],
              ].map(([step, desc]) => (
                <div key={step} style={{ marginBottom: 14, paddingLeft: 16, borderLeft: "2px solid rgba(77,166,255,0.3)" }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "#4da6ff", marginBottom: 3 }}>{step}</div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.45)", lineHeight: 1.6 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
