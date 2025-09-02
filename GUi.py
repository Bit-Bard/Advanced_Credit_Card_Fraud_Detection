import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS for Dark Theme -----------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #b0b3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Custom metrics */
    .metric-container {
        background: linear-gradient(145deg, #1e1e2e, #2a2d3a);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #3a3d4a;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .metric-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #4ecdc4;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Alert boxes */
    .fraud-alert {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #2ed573, #1abc9c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #3a3d4a;
        color: #b0b3b8;
    }
    
    .footer a {
        color: #4ecdc4;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        color: #45b7d1;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1e1e2e !important;
    }
    
    /* Upload area enhancement */
    .uploadedFile {
        border: 2px dashed #4ecdc4;
        border-radius: 10px;
        padding: 2rem;
        background: linear-gradient(145deg, #1e1e2e, #2a2d3a);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Load Models & Scalers -----------------
@st.cache_resource
def load_models():
    try:
        lgbm_model = joblib.load("lgbm_Base_model.pkl")
        xgb_model = joblib.load("xgb_Base_model.pkl")
        meta_model = load_model("NN_Meta_Model.keras")
        scaler_base = joblib.load("Scaler_Base.pkl")
        scaler_meta = joblib.load("Scaler_meta.pkl")
        return lgbm_model, xgb_model, meta_model, scaler_base, scaler_meta
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

# ----------------- Header -----------------
st.markdown('<h1 class="main-header">🔎 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-Powered Stacking Model for Real-time Fraud Detection</p>', unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("## ⚙️ Model Information")
    st.info("""
    **Ensemble Architecture:**
    - 🌲 LightGBM Base Model
    - 🚀 XGBoost Base Model
    - 🧠 Neural Network Meta Model
    
    **Features:**
    - Real-time predictions
    - Probability scoring
    - Advanced preprocessing
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Upload Guidelines")
    st.warning("""
    **Required Format:**
    - CSV file with PCA-transformed features
    - Include 'Time' and 'Amount' columns
    - Max file size: 200MB
    """)

# ----------------- Main Content -----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Upload Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose your CSV file", 
        type=["csv"],
        help="Upload your PCA-transformed transaction dataset"
    )

with col2:
    if uploaded_file is not None:
        st.markdown("### ✅ File Status")
        st.success(f"**File:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

if uploaded_file is not None:
    # Load models
    lgbm_model, xgb_model, meta_model, scaler_base, scaler_meta = load_models()
    
    if all([lgbm_model, xgb_model, meta_model, scaler_base, scaler_meta]):
        # Load and process data
        with st.spinner("🔄 Processing your data..."):
            data = pd.read_csv(uploaded_file)
            
        st.markdown("### 👀 Data Preview")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Sample Data", "📈 Data Statistics", "🔍 Data Info"])
        
        with tab1:
            st.dataframe(data.head(10), use_container_width=True)
            
        with tab2:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-container"><div class="metric-title">Total Rows</div><div class="metric-value">{len(data):,}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-container"><div class="metric-title">Features</div><div class="metric-value">{len(data.columns)}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-container"><div class="metric-title">Memory Usage</div><div class="metric-value">{data.memory_usage().sum() / 1024:.1f} KB</div></div>', unsafe_allow_html=True)
            with col4:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.markdown(f'<div class="metric-container"><div class="metric-title">Missing %</div><div class="metric-value">{missing_pct:.1f}%</div></div>', unsafe_allow_html=True)
                
        with tab3:
            st.text(f"Dataset shape: {data.shape}")
            st.text(f"Data types:\n{data.dtypes.value_counts().to_string()}")
            
        # Process predictions
        with st.spinner("🤖 Running AI models..."):
            # Preprocessing
            data_processed = data.copy()
            if "Amount" in data.columns and "Time" in data.columns:
                data_processed[["Time", "Amount"]] = scaler_base.transform(data_processed[["Time", "Amount"]])
            
            # Base model predictions
            base_pred_lgbm = lgbm_model.predict_proba(data_processed)[:, 1]
            base_pred_xgb = xgb_model.predict_proba(data_processed)[:, 1]
            
            # Stack predictions
            stacked_features = np.column_stack([base_pred_lgbm, base_pred_xgb])
            stacked_scaled = scaler_meta.transform(stacked_features)
            
            # Meta model prediction
            meta_pred_prob = meta_model.predict(stacked_scaled).ravel()
            meta_pred_class = (meta_pred_prob > 0.5).astype(int)
        
        # Results
        st.markdown("### 🎯 Prediction Results")
        
        # Summary metrics
        fraud_count = np.sum(meta_pred_class)
        legit_count = len(meta_pred_class) - fraud_count
        fraud_rate = (fraud_count / len(data)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">✅ Legitimate Transactions</div>
                <div class="metric-value">{legit_count:,}</div>
                <div style="color: #2ed573; font-size: 0.9rem;">{((legit_count/len(data))*100):.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">⚠️ Fraudulent Transactions</div>
                <div class="metric-value" style="color: #ff4757;">{fraud_count:,}</div>
                <div style="color: #ff4757; font-size: 0.9rem;">{fraud_rate:.2f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            avg_fraud_prob = np.mean(meta_pred_prob[meta_pred_class == 1]) if fraud_count > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">🎯 Avg Fraud Confidence</div>
                <div class="metric-value">{avg_fraud_prob:.1%}</div>
                <div style="color: #4ecdc4; font-size: 0.9rem;">Detection accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📊 Interactive Analytics")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Distribution Analysis", "🎯 Probability Heatmap", "⚡ Model Performance"])
        
        with viz_tab1:
            # Fraud probability distribution
            fig = go.Figure()
            
            # Legitimate transactions
            legit_probs = meta_pred_prob[meta_pred_class == 0]
            fraud_probs = meta_pred_prob[meta_pred_class == 1]
            
            fig.add_trace(go.Histogram(
                x=legit_probs,
                name="Legitimate",
                opacity=0.7,
                marker_color="#2ed573",
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=fraud_probs,
                name="Fraudulent",
                opacity=0.7,
                marker_color="#ff4757",
                nbinsx=50
            ))
            
            fig.update_layout(
                title="Fraud Probability Distribution",
                xaxis_title="Fraud Probability",
                yaxis_title="Count",
                template="plotly_dark",
                barmode="overlay",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            # Create probability bins for heatmap
            prob_bins = pd.cut(meta_pred_prob, bins=10, labels=[f"{i*0.1:.1f}-{(i+1)*0.1:.1f}" for i in range(10)])
            heatmap_data = pd.crosstab(prob_bins, meta_pred_class, normalize='index') * 100
            
            fig = px.imshow(
                heatmap_data.T,
                labels=dict(x="Probability Range", y="Prediction", color="Percentage"),
                x=heatmap_data.index,
                y=['Legitimate', 'Fraudulent'],
                color_continuous_scale="RdYlBu_r",
                title="Prediction Confidence Heatmap"
            )
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            # Model comparison
            model_scores = pd.DataFrame({
                'LightGBM': base_pred_lgbm,
                'XGBoost': base_pred_xgb,
                'Meta Model': meta_pred_prob
            })
            
            fig = go.Figure()
            for col in model_scores.columns:
                fig.add_trace(go.Box(
                    y=model_scores[col],
                    name=col,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Model Prediction Comparison",
                yaxis_title="Fraud Probability",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        results = data.copy()
        results["Fraud_Probability"] = meta_pred_prob
        results["Fraud_Prediction"] = meta_pred_class
        results["Risk_Level"] = pd.cut(
            meta_pred_prob, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=["Low", "Medium", "High"]
        )
        
        # Add color coding for fraud prediction
        def highlight_fraud(row):
            if row["Fraud_Prediction"] == 1:
                return ["background-color: rgba(255, 71, 87, 0.3)"] * len(row)
            else:
                return ["background-color: rgba(46, 213, 115, 0.1)"] * len(row)
        
        st.dataframe(
            results.head(20).style.apply(highlight_fraud, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Risk level summary
        risk_summary = results['Risk_Level'].value_counts()
        if not risk_summary.empty:
            st.markdown("### 🚨 Risk Level Breakdown")
            risk_cols = st.columns(len(risk_summary))
            
            colors = {"Low": "#2ed573", "Medium": "#ffa502", "High": "#ff4757"}
            
            for i, (risk, count) in enumerate(risk_summary.items()):
                with risk_cols[i]:
                    percentage = (count / len(results)) * 100
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-title">{risk} Risk</div>
                        <div class="metric-value" style="color: {colors.get(risk, '#4ecdc4')};">{count:,}</div>
                        <div style="color: {colors.get(risk, '#4ecdc4')}; font-size: 0.9rem;">{percentage:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Download section
        st.markdown("### 💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "fraud_predictions_full.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            fraud_only = results[results["Fraud_Prediction"] == 1]
            if len(fraud_only) > 0:
                fraud_csv = fraud_only.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "🚨 Download Fraud Cases Only (CSV)",
                    fraud_csv,
                    "fraud_cases_only.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.success("🎉 No fraudulent transactions detected!")
    
    else:
        st.error("❌ Error loading ML models. Please check if all model files are available.")

# ----------------- Footer -----------------
st.markdown("""
<div class="footer">
    <h3>🚀 Created by Dhruv Devaliya</h3>
    <p>AI/ML Engineer & Data Scientist</p>
    <p>
        <a href="https://www.linkedin.com/in/dhruv-devaliya/" target="_blank">
            🔗 Connect on LinkedIn
        </a>
    </p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        Built with ❤️ using Streamlit, TensorFlow, and Advanced ML Techniques
    </p>
</div>
""", unsafe_allow_html=True)