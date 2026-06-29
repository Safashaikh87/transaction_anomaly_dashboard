"""
Transaction Anomaly Detection Dashboard
Deployed on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from feature_engineer import FeatureEngineer
from model_trainer import AnomalyDetector
from data_generator import generate_transactions

# Page configuration
st.set_page_config(
    page_title="Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf5;
        margin-bottom: 5px;
    }
    .main-subtitle {
        font-size: 14px;
        color: #8892b0;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #131b26;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #1a2332;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
    }
    .metric-label {
        font-size: 12px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .high-risk { color: #ff6b6b; }
    .medium-risk { color: #ffd93d; }
    .low-risk { color: #6bcb77; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=60)
    st.markdown("## 🛡️ Anomaly Detection")
    st.markdown("---")
    
    # Data source selection
    st.markdown("### 📊 Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Generate Synthetic Data", "Upload CSV"],
        index=0
    )
    
    if data_source == "Generate Synthetic Data":
        n_transactions = st.slider("Number of transactions:", 1000, 20000, 5000, 1000)
        anomaly_rate = st.slider("Anomaly rate:", 0.01, 0.20, 0.05, 0.01)
        n_users = st.slider("Number of users:", 10, 500, 100, 10)
        
        if st.button("🔄 Generate Data", use_container_width=True):
            with st.spinner("Generating transaction data..."):
                st.session_state.df = generate_transactions(
                    n=n_transactions,
                    anomaly_rate=anomaly_rate,
                    n_users=n_users
                )
                st.session_state.data_loaded = True
                st.session_state.model_trained = False
                st.success(f"✅ Generated {len(st.session_state.df)} transactions")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Ensure required columns exist
            required_cols = ['transaction_id', 'timestamp', 'amount', 'sender_country', 'receiver_country']
            if all(col in df.columns for col in required_cols):
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.model_trained = False
                st.success(f"✅ Loaded {len(df)} transactions")
            else:
                st.error(f"CSV must contain: {', '.join(required_cols)}")
    
    st.markdown("---")
    
    # Model training
    if st.session_state.data_loaded and not st.session_state.model_trained:
        st.markdown("### 🤖 Model Training")
        if st.button("🚀 Train Isolation Forest", use_container_width=True, type="primary"):
            with st.spinner("Training model..."):
                # Generate features
                engineer = st.session_state.feature_engineer
                df = engineer.generate_features(st.session_state.df)
                st.session_state.df = df
                
                # Prepare features
                features = engineer.get_feature_names()
                X = df[features].fillna(0)
                
                # Train model
                model = AnomalyDetector(contamination=0.05)
                results = model.train(X, df.get('is_anomaly'))
                st.session_state.model = model
                st.session_state.results = results
                
                # Add predictions to dataframe
                predictions, risk_scores = model.predict(X)
                df['predicted_anomaly'] = (predictions == -1).astype(int)
                df['anomaly_score'] = risk_scores
                df['risk_level'] = model.get_risk_level(risk_scores)
                
                st.session_state.df = df
                st.session_state.model_trained = True
                st.success("✅ Model trained successfully!")
    
    if st.session_state.model_trained:
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        st.success("🟢 Model Active")
        if st.session_state.results:
            st.caption(f"Anomaly rate: {st.session_state.results['anomaly_rate']:.2%}")
            st.caption(f"Features: {st.session_state.results['n_features']}")
    
    st.markdown("---")
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# MAIN CONTENT
# ============================================

if not st.session_state.data_loaded:
    st.markdown('<div class="main-header">🛡️ Transaction Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">ISO 20022 Compliant · Isolation Forest Powered · Deployed on Streamlit Cloud</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Generate Data**\n\nCreate synthetic transaction data with realistic patterns")
    with col2:
        st.info("📁 **Upload CSV**\n\nUse your own transaction data for analysis")
    with col3:
        st.info("🚀 **Train Model**\n\nTrain Isolation Forest on your data instantly")
    
    st.markdown("---")
    st.markdown("""
    ### 🔍 What This Dashboard Does
    
    1. **Feature Engineering** - Extracts 37+ signals from transaction data
    2. **Anomaly Detection** - Uses Isolation Forest to identify suspicious patterns
    3. **Risk Scoring** - Each transaction gets a 0-1 risk score
    4. **Visualization** - Interactive charts and tables for ops teams
    5. **Monitoring** - Real-time transaction monitoring capabilities
    """)
    st.stop()

if not st.session_state.model_trained:
    st.warning("⚠️ Please train the model using the sidebar before viewing the dashboard.")
    st.stop()

# ============================================
# DASHBOARD
# ============================================

df = st.session_state.df
engineer = st.session_state.feature_engineer

# Header
st.markdown('<div class="main-header">🛡️ Transaction Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">ISO 20022 Compliant · Isolation Forest Powered · {len(df):,} transactions loaded</div>', unsafe_allow_html=True)

# ============================================
# KPI ROW
# ============================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total = len(df)
    st.metric("📊 Total Transactions", f"{total:,}")

with col2:
    if 'risk_level' in df.columns:
        flagged = df[df['risk_level'] != 'Low'].shape[0]
        st.metric("🚨 Flagged", f"{flagged:,}", delta=f"{flagged/total*100:.1f}%")
    else:
        st.metric("🚨 Flagged", "0")

with col3:
    if 'risk_level' in df.columns:
        high_risk = df[df['risk_level'] == 'High'].shape[0]
        st.metric("🔴 High Risk", f"{high_risk:,}")
    else:
        st.metric("🔴 High Risk", "0")

with col4:
    if 'anomaly_score' in df.columns:
        avg_score = df['anomaly_score'].mean()
        st.metric("📈 Avg Risk Score", f"{avg_score:.3f}")
    else:
        st.metric("📈 Avg Risk Score", "0")

with col5:
    if 'risk_level' in df.columns:
        blocked = df[df['risk_level'] == 'High']['amount'].sum()
        st.metric("💰 Blocked Value", f"€{blocked:,.0f}")
    else:
        st.metric("💰 Blocked Value", "€0")

# ============================================
# CHARTS ROW 1
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Time Series: Volume vs Anomalies")
    if 'timestamp' in df.columns:
        # Group by hour
        df['hour_bin'] = df['timestamp'].dt.floor('H')
        time_series = df.groupby('hour_bin').agg({
            'transaction_id': 'count',
            'is_anomaly': 'sum' if 'is_anomaly' in df.columns else lambda x: 0
        }).reset_index()
        time_series.columns = ['hour', 'volume', 'anomalies']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=time_series['hour'],
            y=time_series['volume'],
            name='Volume',
            marker_color='#1a73e8',
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=time_series['hour'],
            y=time_series['anomalies'],
            name='Anomalies',
            marker_color='#ff6b6b',
            line=dict(width=3)
        ))
        fig.update_layout(
            template='plotly_dark',
            height=350,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Risk Distribution")
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        colors = {'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcb77'}
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[colors.get(k, '#8892b0') for k in risk_counts.index]),
            hole=0.4,
            textinfo='label+percent'
        )])
        fig.update_layout(
            template='plotly_dark',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# CHARTS ROW 2
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌍 Top Sender Countries")
    if 'sender_country' in df.columns:
        country_data = df['sender_country'].value_counts().head(10)
        fig = px.bar(
            x=country_data.values,
            y=country_data.index,
            orientation='h',
            color=country_data.values,
            color_continuous_scale='Blues',
            template='plotly_dark',
            height=300
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💳 Merchant Categories")
    if 'merchant_category' in df.columns:
        merchant_data = df['merchant_category'].value_counts().head(8)
        fig = px.pie(
            values=merchant_data.values,
            names=merchant_data.index,
            template='plotly_dark',
            height=300,
            hole=0.3
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("### 📊 Anomaly Score Distribution")
    if 'anomaly_score' in df.columns:
        fig = px.histogram(
            df,
            x='anomaly_score',
            nbins=30,
            color='risk_level' if 'risk_level' in df.columns else None,
            template='plotly_dark',
            height=300
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ff6b6b")
        fig.add_vline(x=0.4, line_dash="dash", line_color="#ffd93d")
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TRANSACTION TABLE
# ============================================

st.markdown("---")
st.markdown("### 📋 Transaction Explorer")

# Filters
col1, col2, col3, col4 = st.columns(4)
with col1:
    risk_filter = st.multiselect(
        "Risk Level",
        options=['All', 'High', 'Medium', 'Low'],
        default=['All']
    )
with col2:
    min_amount = st.number_input("Min Amount (€)", value=0.0, step=100.0)
with col3:
    max_amount = st.number_input("Max Amount (€)", value=1000000.0, step=1000.0)
with col4:
    if 'sender_country' in df.columns:
        countries = ['All'] + sorted(df['sender_country'].unique().tolist())
        country_filter = st.selectbox("Sender Country", countries)

# Apply filters
filtered_df = df.copy()
if 'All' not in risk_filter:
    filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
filtered_df = filtered_df[filtered_df['amount'].between(min_amount, max_amount)]
if country_filter != 'All':
    filtered_df = filtered_df[filtered_df['sender_country'] == country_filter]

# Display table
display_cols = ['transaction_id', 'timestamp', 'amount', 'sender_country', 
                'receiver_country', 'merchant_category', 'risk_level', 'anomaly_score']

# Only show columns that exist
display_cols = [c for c in display_cols if c in filtered_df.columns]

# Format amount
if 'amount' in filtered_df.columns:
    filtered_df['amount_formatted'] = filtered_df['amount'].apply(lambda x: f"€{x:,.2f}")

st.dataframe(
    filtered_df[display_cols].sort_values('timestamp', ascending=False).head(100),
    use_container_width=True,
    height=400
)

st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df)} transactions")

# ============================================
# EXPORT SECTION
# ============================================

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    # Export data
    csv = filtered_df[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"transactions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    st.caption(f"🔬 Feature engineering: {len(engineer.get_feature_names())} signals extracted")
    st.caption("🤖 Model: Isolation Forest (scikit-learn)")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📊 {len(df):,} total transactions")
with col2:
    st.caption(f"🔬 {len(engineer.get_feature_names())} engineered features")
with col3:
    if st.session_state.model:
        st.caption("🤖 Isolation Forest model active")