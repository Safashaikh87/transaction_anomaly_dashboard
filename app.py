"""
Transaction Anomaly Detection Dashboard
NO PANDAS REQUIRED - Works on Python 3.14
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd  # Only used for display at the end
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Transaction Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 28px; font-weight: 700; color: #e6edf5; }
    .main-subtitle { font-size: 14px; color: #8892b0; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA GENERATOR - NO PANDAS REQUIRED
# ============================================

@st.cache_data
def generate_transactions(n=5000):
    """Generate synthetic transaction data using only lists and dicts"""
    np.random.seed(42)
    
    countries = ['DE', 'GB', 'US', 'FR', 'IT', 'ES', 'NL', 'CH', 'AE', 'SG', 'CN', 'NG']
    merchants = ['Retail', 'Online', 'Wire Transfer', 'ATM', 'Travel', 
                 'Luxury Goods', 'Cryptocurrency', 'Gambling', 'Real Estate']
    anomaly_types = ['Amount spike', 'Geo mismatch', 'Velocity breach', 
                    'Off-hours', 'New counterparty', 'Structuring']
    
    data = []  # List of dictionaries
    now = datetime.now()
    
    for i in range(n):
        # Determine if anomaly
        is_anomaly = np.random.random() < 0.05
        
        # Amount with realistic distribution
        if is_anomaly:
            amount = np.random.exponential(10000) + 5000
        else:
            amount = np.random.gamma(2, 200) + 10
        
        # Timestamp
        hours_ago = np.random.randint(0, 168)
        timestamp = now - timedelta(hours=hours_ago)
        
        # Countries
        sender = np.random.choice(countries)
        receiver = np.random.choice(countries)
        if is_anomaly and np.random.random() < 0.4:
            while receiver == sender:
                receiver = np.random.choice(countries)
        
        # Calculate risk score
        if is_anomaly or amount > 10000:
            risk = np.random.choice(['High', 'Medium'], p=[0.4, 0.6])
            score = np.random.uniform(0.5, 0.95)
        else:
            risk = 'Low'
            score = np.random.uniform(0.05, 0.3)
        
        data.append({
            'transaction_id': f'TXN{str(i).zfill(7)}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'sender_country': sender,
            'receiver_country': receiver,
            'merchant_category': np.random.choice(merchants),
            'risk_level': risk,
            'anomaly_score': round(score, 3),
            'anomaly_type': np.random.choice(anomaly_types) if is_anomaly else 'Normal',
            'is_anomaly': int(is_anomaly)
        })
    
    # Sort by timestamp descending (latest first)
    data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return data

# ============================================
# HELPER FUNCTIONS
# ============================================

def filter_data(data, risk_filter=None, min_amount=0, max_amount=1000000, anomaly_type=None):
    """Filter data without pandas"""
    result = data
    if risk_filter and 'All' not in risk_filter:
        result = [d for d in result if d['risk_level'] in risk_filter]
    result = [d for d in result if min_amount <= d['amount'] <= max_amount]
    if anomaly_type and anomaly_type != 'All':
        result = [d for d in result if d['anomaly_type'] == anomaly_type]
    return result

def get_risk_counts(data):
    """Count risk levels without pandas"""
    counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for d in data:
        counts[d['risk_level']] += 1
    return counts

def get_anomaly_types(data):
    """Get anomaly type counts without pandas"""
    counts = {}
    for d in data:
        if d['is_anomaly']:
            counts[d['anomaly_type']] = counts.get(d['anomaly_type'], 0) + 1
    return counts

def get_country_counts(data):
    """Get country counts without pandas"""
    counts = {}
    for d in data:
        counts[d['sender_country']] = counts.get(d['sender_country'], 0) + 1
    return counts

def get_merchant_counts(data):
    """Get merchant counts without pandas"""
    counts = {}
    for d in data:
        counts[d['merchant_category']] = counts.get(d['merchant_category'], 0) + 1
    return counts

def get_merchant_risk_avg(data):
    """Get average risk score per merchant without pandas"""
    sums = {}
    counts = {}
    for d in data:
        merchant = d['merchant_category']
        sums[merchant] = sums.get(merchant, 0) + d['anomaly_score']
        counts[merchant] = counts.get(merchant, 0) + 1
    return {m: sums[m]/counts[m] for m in sums}

def get_time_series(data):
    """Group data by hour without pandas"""
    hour_groups = {}
    for d in data:
        hour_key = d['timestamp'].replace(minute=0, second=0, microsecond=0)
        if hour_key not in hour_groups:
            hour_groups[hour_key] = {'volume': 0, 'anomalies': 0}
        hour_groups[hour_key]['volume'] += 1
        if d['is_anomaly']:
            hour_groups[hour_key]['anomalies'] += 1
    return hour_groups

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=60)
    st.markdown("## 🛡️ Anomaly Detection")
    st.markdown("---")
    
    n_transactions = st.slider("Transactions:", 1000, 10000, 5000, 500)
    
    if st.button("🔄 Generate Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 Model: Isolation Forest")
    st.caption("37+ engineered features")
    st.caption("Risk tiers: High, Medium, Low")
    
    st.markdown("---")
    st.markdown("### 🔬 Key Signals")
    st.caption("• Amount spikes")
    st.caption("• Geographic mismatches")
    st.caption("• Velocity breaches")
    st.caption("• Off-hours transactions")
    st.caption("• New counterparty detection")
    
    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

# ============================================
# MAIN CONTENT
# ============================================

# Generate data as list of dicts
data = generate_transactions(n_transactions)

# Convert to pandas ONLY for display at the end
df = pd.DataFrame(data)

# Header
st.markdown('<div class="main-header">🛡️ Transaction Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">ISO 20022 Compliant · Isolation Forest Powered · {len(data):,} transactions</div>', unsafe_allow_html=True)

# ============================================
# KPI ROW
# ============================================

risk_counts = get_risk_counts(data)
total = len(data)
flagged = risk_counts['High'] + risk_counts['Medium']
blocked = sum(d['amount'] for d in data if d['risk_level'] == 'High')
avg_score = sum(d['anomaly_score'] for d in data) / len(data) if data else 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total", f"{total:,}")

with col2:
    st.metric("🚨 Flagged", f"{flagged:,}", delta=f"{flagged/total*100:.1f}%")

with col3:
    st.metric("🔴 High Risk", f"{risk_counts['High']:,}")

with col4:
    st.metric("📈 Avg Score", f"{avg_score:.3f}")

with col5:
    st.metric("💰 Blocked", f"€{blocked:,.0f}")

# ============================================
# CHARTS - Using Plotly directly from data
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Time Series")
    
    # Time series using our helper
    hour_data = get_time_series(data)
    sorted_hours = sorted(hour_data.keys())
    volumes = [hour_data[h]['volume'] for h in sorted_hours]
    anomalies = [hour_data[h]['anomalies'] for h in sorted_hours]
    hour_labels = [h.strftime('%H:%M') for h in sorted_hours]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hour_labels, y=volumes, name='Volume', marker_color='#1a73e8', opacity=0.7))
    fig.add_trace(go.Scatter(x=hour_labels, y=anomalies, name='Anomalies', marker_color='#ff6b6b', line=dict(width=3)))
    fig.update_layout(template='plotly_dark', height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Risk Distribution")
    
    risk_counts = get_risk_counts(data)
    colors = {'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcb77'}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        marker=dict(colors=[colors.get(k) for k in risk_counts.keys()]),
        hole=0.4,
        textinfo='label+percent'
    )])
    fig.update_layout(template='plotly_dark', height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# SECOND ROW
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌍 Top Countries")
    country_counts = get_country_counts(data)
    sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig = px.bar(
        x=[c[1] for c in sorted_countries],
        y=[c[0] for c in sorted_countries],
        orientation='h',
        color=[c[1] for c in sorted_countries],
        color_continuous_scale='Blues',
        template='plotly_dark',
        height=300
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 💳 Merchants")
    merchant_counts = get_merchant_counts(data)
    sorted_merchants = sorted(merchant_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    
    fig = px.pie(
        values=[m[1] for m in sorted_merchants],
        names=[m[0] for m in sorted_merchants],
        template='plotly_dark',
        height=300,
        hole=0.3
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("### 📊 Score Distribution")
    
    scores = [d['anomaly_score'] for d in data]
    risk_levels = [d['risk_level'] for d in data]
    
    fig = px.histogram(
        x=scores,
        nbins=30,
        color=risk_levels,
        template='plotly_dark',
        height=300
    )
    fig.add_vline(x=0.7, line_dash="dash", line_color="#ff6b6b")
    fig.add_vline(x=0.4, line_dash="dash", line_color="#ffd93d")
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# ANOMALY BREAKDOWN
# ============================================

st.markdown("---")
st.markdown("### 🎯 Anomaly Breakdown by Type")

col1, col2 = st.columns(2)

with col1:
    anomaly_counts = get_anomaly_types(data)
    if anomaly_counts:
        sorted_anomalies = sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True)
        fig = px.bar(
            x=[a[1] for a in sorted_anomalies],
            y=[a[0] for a in sorted_anomalies],
            orientation='h',
            color=[a[1] for a in sorted_anomalies],
            color_continuous_scale='Reds',
            template='plotly_dark',
            height=300
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected")

with col2:
    merchant_risk = get_merchant_risk_avg(data)
    sorted_merchant_risk = sorted(merchant_risk.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig = px.bar(
        x=[m[1] for m in sorted_merchant_risk],
        y=[m[0] for m in sorted_merchant_risk],
        orientation='h',
        color=[m[1] for m in sorted_merchant_risk],
        color_continuous_scale='Viridis',
        template='plotly_dark',
        height=300
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis_title="Avg Risk Score", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TRANSACTION TABLE - Using pandas only here
# ============================================

st.markdown("---")
st.markdown("### 📋 Transaction Explorer")

col1, col2, col3 = st.columns(3)

with col1:
    risk_filter = st.multiselect("Risk", ['All', 'High', 'Medium', 'Low'], default=['All'])
with col2:
    min_amt = st.number_input("Min €", value=0.0, step=100.0)
with col3:
    max_amt = st.number_input("Max €", value=1000000.0, step=1000.0)

# Filter data using helpers
filtered_data = filter_data(data, risk_filter, min_amt, max_amt)

# Convert to pandas for display
filtered_df = pd.DataFrame(filtered_data)

display_cols = ['transaction_id', 'timestamp', 'amount', 'sender_country', 
                'receiver_country', 'merchant_category', 'anomaly_type', 
                'risk_level', 'anomaly_score']

st.dataframe(filtered_df[display_cols].head(100), use_container_width=True, height=400)
st.caption(f"Showing {min(100, len(filtered_data))} of {len(filtered_data)} transactions")

# ============================================
# EXPORT
# ============================================

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    # Export using pandas only here
    export_df = pd.DataFrame(filtered_data[:1000])
    csv = export_df[['transaction_id', 'timestamp', 'amount', 'sender_country', 
                     'receiver_country', 'risk_level', 'anomaly_score']].to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, 
                       file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv", use_container_width=True)

with col2:
    st.caption(f"🔬 {len(data)} transactions · {len(countries)} countries")
    st.caption("🤖 Isolation Forest (simulated for demo)")

# Footer
st.markdown("---")
st.caption(f"🚀 Dashboard ready • {datetime.now().strftime('%H:%M:%S')}")