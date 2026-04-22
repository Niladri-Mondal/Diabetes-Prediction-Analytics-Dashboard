"""
🩺 Diabetes Prediction & Analytics Dashboard
A colorful, professional Streamlit app using an SVM model trained on the Pima Indians Diabetes dataset.

Requirements:
    pip install streamlit plotly pandas scikit-learn numpy

Run:
    streamlit run diabetes_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global colour palette ─────────────────────────────────────────────────────
COLORS = {
    "primary":   "#7C3AED",   # violet
    "secondary": "#06B6D4",   # cyan
    "accent":    "#F59E0B",   # amber
    "danger":    "#EF4444",   # red
    "success":   "#10B981",   # emerald
    "bg":        "#0F172A",   # dark navy
    "card":      "#1E293B",   # slate
    "text":      "#F1F5F9",   # light
    "muted":     "#94A3B8",   # grey-blue
}

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {COLORS['bg']};
      color: {COLORS['text']};
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
      border-right: 1px solid #312e81;
  }}
  [data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

  /* ── Headers ── */
  h1 {{ background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700; font-size: 2.4rem; }}
  h2 {{ color: {COLORS['secondary']}; font-weight: 600; }}
  h3 {{ color: {COLORS['accent']}; font-weight: 600; }}

  /* ── Metric cards ── */
  [data-testid="metric-container"] {{
      background: {COLORS['card']};
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px 20px;
      box-shadow: 0 4px 24px rgba(0,0,0,.4);
  }}
  [data-testid="metric-container"] label {{ color: {COLORS['muted']} !important; font-size:.85rem; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
      color: {COLORS['text']} !important; font-size: 1.7rem; font-weight: 700;
  }}

  /* ── Input widgets ── */
  .stSlider > div > div > div {{ background: {COLORS['primary']}; }}
  .stNumberInput input, .stTextInput input {{
      background: {COLORS['card']}; color: {COLORS['text']};
      border: 1px solid #475569; border-radius: 8px;
  }}

  /* ── Buttons ── */
  .stButton > button {{
      background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
      color: white; border: none; border-radius: 10px;
      font-weight: 600; font-size: 1rem; padding: 0.6rem 2rem;
      width: 100%; transition: transform .15s, box-shadow .15s;
  }}
  .stButton > button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(124,58,237,.5);
  }}

  /* ── Tabs ── */
  [data-testid="stTabs"] button {{
      color: {COLORS['muted']}; font-weight: 500;
      border-bottom: 2px solid transparent;
  }}
  [data-testid="stTabs"] button[aria-selected="true"] {{
      color: {COLORS['primary']} !important;
      border-bottom: 2px solid {COLORS['primary']};
  }}

  /* ── Result banners ── */
  .result-positive {{
      background: linear-gradient(135deg, #450a0a, #7f1d1d);
      border-left: 5px solid {COLORS['danger']};
      border-radius: 12px; padding: 20px 24px; margin: 12px 0;
  }}
  .result-negative {{
      background: linear-gradient(135deg, #052e16, #14532d);
      border-left: 5px solid {COLORS['success']};
      border-radius: 12px; padding: 20px 24px; margin: 12px 0;
  }}

  /* ── Expander ── */
  [data-testid="stExpander"] {{
      background: {COLORS['card']}; border: 1px solid #334155;
      border-radius: 10px;
  }}

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {{ background: {COLORS['card']}; border-radius: 10px; }}

  /* ── Divider ── */
  hr {{ border-color: #1e293b; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_diabetic", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

def gauge_chart(value: float, title: str):
    """Speedometer-style gauge for a probability score 0–1."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"color": COLORS["text"], "size": 14}},
        number={"suffix": "%", "font": {"color": COLORS["text"], "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["danger"] if value >= 0.5 else COLORS["success"]},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["muted"],
            "steps": [
                {"range": [0,  40], "color": "#052e16"},
                {"range": [40, 60], "color": "#451a03"},
                {"range": [60,100], "color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": COLORS["accent"], "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )
    return fig

def styled_hist(df, col, color):
    fig = px.histogram(
        df, x=col, color="Outcome",
        color_discrete_map={0: COLORS["success"], 1: COLORS["danger"]},
        barmode="overlay", opacity=0.75,
        labels={"Outcome": "Diabetes"},
        title=f"Distribution of {col}",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]}, legend_font_color=COLORS["text"],
        title_font_color=COLORS["secondary"], xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )
    return fig

def radar_chart(input_data: dict, df: pd.DataFrame):
    """Radar comparing patient vs dataset mean."""
    features = list(input_data.keys())
    patient   = [input_data[f] for f in features]
    means_0   = [df[df["Outcome"]==0][f].mean() for f in features]
    means_1   = [df[df["Outcome"]==1][f].mean() for f in features]

    # normalise to [0,1] per feature
    maxv = [df[f].max() for f in features]
    pat_n = [p/m if m else 0 for p,m in zip(patient, maxv)]
    m0_n  = [v/m if m else 0 for v,m in zip(means_0, maxv)]
    m1_n  = [v/m if m else 0 for v,m in zip(means_1, maxv)]

    cats = features + [features[0]]
    fig = go.Figure()
    for vals, name, color in [
        (pat_n + [pat_n[0]], "Your Values",       COLORS["accent"]),
        (m0_n  + [m0_n[0]],  "Non-Diabetic Avg",  COLORS["success"]),
        (m1_n  + [m1_n[0]],  "Diabetic Avg",      COLORS["danger"]),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=name,
            line_color=color, opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
        legend_font_color=COLORS["text"], title="Patient Profile vs Population",
        title_font_color=COLORS["secondary"], height=380,
    )
    return fig


# ── Load assets ───────────────────────────────────────────────────────────────
model = load_model()
df    = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Diabetes Predictor")
    st.markdown("---")
    st.markdown("### 📋 Patient Parameters")

    preg   = st.slider("Pregnancies",            0, 17,  3,  help="Number of times pregnant")
    gluc   = st.slider("Glucose (mg/dL)",        0,199,120,  help="Plasma glucose (2hr oral test)")
    bp     = st.slider("Blood Pressure (mmHg)",  0,122, 72,  help="Diastolic blood pressure")
    skin   = st.slider("Skin Thickness (mm)",    0,100, 23,  help="Triceps skin fold thickness")
    ins    = st.slider("Insulin (μU/mL)",         0,846, 30,  help="2-Hour serum insulin")
    bmi    = st.slider("BMI",                    0.0,70.0,32.0,step=0.1, help="Body mass index")
    dpf    = st.slider("Diabetes Pedigree",      0.0, 2.5, 0.47,step=0.01,help="Diabetes pedigree function")
    age    = st.slider("Age",                    21, 81,  29,  help="Age in years")

    st.markdown("---")
    predict_btn = st.button("🔬 Run Prediction", use_container_width=True)
    st.markdown("---")

    # Sidebar info
    total     = len(df)
    diabetic  = df["Outcome"].sum()
    non_diab  = total - diabetic
    st.markdown(f"**Dataset:** {total} patients")
    st.markdown(f"🔴 Diabetic: **{diabetic}** ({diabetic/total*100:.1f}%)")
    st.markdown(f"🟢 Non-Diabetic: **{non_diab}** ({non_diab/total*100:.1f}%)")

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🩺 Diabetes Prediction & Analytics Dashboard")
st.markdown("##### AI-powered early diabetes screening using Support Vector Machine (SVM)")
st.markdown("---")

# Key KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Dataset Size",    f"{len(df):,} patients")
col2.metric("🔴 Diabetic Cases",  f"{df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
col3.metric("📈 Avg Glucose",     f"{df['Glucose'].mean():.1f} mg/dL")
col4.metric("📐 Avg BMI",         f"{df['BMI'].mean():.1f}")

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔬 Prediction", "📊 Analytics", "🔎 Data Explorer", "ℹ️ About"]
)

# ── TAB 1 · Prediction ────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 🔬 Diabetes Risk Prediction")

    input_dict = {
        "Pregnancies": preg, "Glucose": gluc, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": ins, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }
    input_arr = np.array([[preg, gluc, bp, skin, ins, bmi, dpf, age]])

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 📋 Current Input Summary")
        summary_df = pd.DataFrame(
            {"Feature": list(input_dict.keys()),
             "Value":   list(input_dict.values())},
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Feature": st.column_config.TextColumn("Feature"),
                         "Value":   st.column_config.NumberColumn("Value", format="%.2f"),
                     })

        if predict_btn:
            prediction = model.predict(input_arr)[0]

            # Try to get decision score for pseudo-probability
            try:
                score     = model.decision_function(input_arr)[0]
                prob_like = 1 / (1 + np.exp(-score))   # sigmoid transform
            except Exception:
                prob_like = float(prediction)

            st.markdown("### 🎯 Prediction Result")
            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                  <h3 style="color:#fca5a5;margin:0">⚠️ High Diabetes Risk Detected</h3>
                  <p style="margin:6px 0 0 0;color:#fecaca">
                    The model predicts a <strong>diabetic</strong> outcome.<br>
                    Please consult a healthcare professional for a complete evaluation.
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                  <h3 style="color:#6ee7b7;margin:0">✅ Low Diabetes Risk</h3>
                  <p style="margin:6px 0 0 0;color:#a7f3d0">
                    The model predicts a <strong>non-diabetic</strong> outcome.<br>
                    Maintain a healthy lifestyle to keep risk low.
                  </p>
                </div>
                """, unsafe_allow_html=True)

    with right:
        if predict_btn:
            st.markdown("### 📡 Risk Score Gauge")
            st.plotly_chart(gauge_chart(prob_like, "Diabetes Risk Score"),
                            use_container_width=True)

        st.markdown("### 🕸️ Patient Profile Radar")
        st.plotly_chart(radar_chart(input_dict, df), use_container_width=True)

    if not predict_btn:
        st.info("⬅️  Adjust patient parameters in the sidebar, then click **Run Prediction**.")


# ── TAB 2 · Analytics ────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Dataset Analytics")

    # Outcome pie
    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(
            values=df["Outcome"].value_counts().values,
            names=["Non-Diabetic", "Diabetic"],
            color_discrete_sequence=[COLORS["success"], COLORS["danger"]],
            title="Outcome Distribution",
            hole=0.45,
        )
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={"color": COLORS["text"]},
            title_font_color=COLORS["secondary"],
        )
        st.plotly_chart(pie, use_container_width=True)

    with c2:
        feature_sel = st.selectbox(
            "Select feature for histogram",
            ["Glucose", "BMI", "Age", "BloodPressure", "Insulin",
             "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies"],
        )
        st.plotly_chart(styled_hist(df, feature_sel, COLORS["primary"]),
                        use_container_width=True)

    # Correlation heatmap
    st.markdown("### 🌡️ Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Viridis", text=corr.round(2).values,
        texttemplate="%{text}", textfont_size=11,
        colorbar=dict(tickfont=dict(color=COLORS["text"])),
    ))
    heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]}, title_font_color=COLORS["secondary"],
        title="Feature Correlation Matrix", height=480,
        xaxis=dict(tickfont=dict(color=COLORS["text"])),
        yaxis=dict(tickfont=dict(color=COLORS["text"])),
    )
    st.plotly_chart(heat, use_container_width=True)

    # Box plots grid
    st.markdown("### 📦 Feature Distributions by Outcome")
    features = ["Glucose", "BMI", "Age", "BloodPressure",
                "Insulin", "DiabetesPedigreeFunction"]
    fig_box = make_subplots(rows=2, cols=3,
                            subplot_titles=features,
                            vertical_spacing=0.15)
    for i, feat in enumerate(features):
        row, col = divmod(i, 3)
        for outcome, color, label in [(0, COLORS["success"], "Non-Diabetic"),
                                       (1, COLORS["danger"],  "Diabetic")]:
            fig_box.add_trace(
                go.Box(y=df[df["Outcome"]==outcome][feat],
                       name=label, marker_color=color,
                       showlegend=(i == 0)),
                row=row+1, col=col+1,
            )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]}, height=550,
        legend_font_color=COLORS["text"],
    )
    for ax in fig_box.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig_box.layout[ax].update(gridcolor="#1e293b", color=COLORS["muted"])
    for ann in fig_box.layout.annotations:
        ann.font.color = COLORS["accent"]
    st.plotly_chart(fig_box, use_container_width=True)

    # Scatter
    st.markdown("### 🔵 Scatter: Glucose vs BMI")
    scatter = px.scatter(
        df, x="Glucose", y="BMI", color="Outcome",
        color_discrete_map={0: COLORS["success"], 1: COLORS["danger"]},
        size="Age", hover_data=["Age", "Pregnancies"],
        labels={"Outcome": "Diabetes"},
        title="Glucose vs BMI (size = Age)",
        opacity=0.75,
    )
    scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]}, title_font_color=COLORS["secondary"],
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(scatter, use_container_width=True)


# ── TAB 3 · Data Explorer ────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🔎 Raw Data Explorer")

    c1, c2, c3 = st.columns(3)
    outcome_filter = c1.selectbox("Filter by Outcome", ["All", "Diabetic (1)", "Non-Diabetic (0)"])
    age_range      = c2.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()),
                                (21, 60))
    gluc_range     = c3.slider("Glucose Range", int(df["Glucose"].min()),
                                int(df["Glucose"].max()), (50, 199))

    filtered = df.copy()
    if outcome_filter == "Diabetic (1)":
        filtered = filtered[filtered["Outcome"] == 1]
    elif outcome_filter == "Non-Diabetic (0)":
        filtered = filtered[filtered["Outcome"] == 0]
    filtered = filtered[
        filtered["Age"].between(*age_range) &
        filtered["Glucose"].between(*gluc_range)
    ]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.dataframe(
        filtered.style.background_gradient(cmap="YlOrRd", subset=["Glucose"])
                       .background_gradient(cmap="Blues",  subset=["BMI"])
                       .format({"BMI": "{:.1f}", "DiabetesPedigreeFunction": "{:.3f}"}),
        use_container_width=True, height=420,
    )

    with st.expander("📈 Descriptive Statistics (filtered)"):
        st.dataframe(filtered.describe().T.style.background_gradient(cmap="magma"),
                     use_container_width=True)


# ── TAB 4 · About ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## ℹ️ About This App")
    left, right = st.columns(2)

    with left:
        st.markdown("""
### 📌 Overview
This dashboard combines a trained **Support Vector Machine (SVM)** model with an 
interactive analytics suite to assist in early diabetes screening.

### 🤖 Model Details
| Property | Value |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | Linear |
| Dataset | Pima Indians Diabetes Database |
| Records | 768 patients |
| Features | 8 clinical measurements |

### 📊 Feature Descriptions
| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose (mg/dL, 2hr OGTT) |
| BloodPressure | Diastolic BP (mmHg) |
| SkinThickness | Triceps fold (mm) |
| Insulin | 2-hr serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic risk score |
| Age | Age in years |
""")

    with right:
        st.markdown("""
### ⚠️ Important Disclaimer
> This tool is **for educational purposes only**.  
> It is **not** a substitute for professional medical diagnosis.  
> Always consult a qualified healthcare provider.

### 🛠️ Technical Stack
- **Frontend:** Streamlit
- **Visualisation:** Plotly
- **ML:** scikit-learn (SVM)
- **Data:** pandas, NumPy

### 📁 Files Required
```
diabetes_app.py       ← this script
diabetes.csv          ← dataset
model_diabetic        ← trained SVM model (pickle)
```
""")
        # Feature importance proxy: absolute SVM coefficients
        try:
            coef = np.abs(model.coef_[0])
            feat_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                          "Insulin","BMI","DiabetesPedigreeFunction","Age"]
            fi = pd.DataFrame({"Feature": feat_names, "Importance": coef})
            fi = fi.sort_values("Importance", ascending=True)
            bar = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title="SVM Feature Weights (|coef|)",
                         color="Importance", color_continuous_scale="Viridis")
            bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]}, title_font_color=COLORS["secondary"],
                coloraxis_showscale=False, height=320,
                xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(bar, use_container_width=True)
        except Exception:
            st.info("Feature weight chart not available for this model configuration.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569;font-size:.8rem;'>"
    "🩺 Diabetes Prediction Dashboard · Built with Streamlit & Plotly · Educational use only"
    "</center>",
    unsafe_allow_html=True,
)
