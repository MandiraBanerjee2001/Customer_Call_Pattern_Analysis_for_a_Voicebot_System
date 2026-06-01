"""
Customer Call Pattern Analysis — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Call Pattern Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #0f1117; --card: #1a1d27; --accent: #00d4ff;
        --success: #00ff88; --danger: #ff4757; --warn: #ffa502;
    }
    .main { background: var(--bg); }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: var(--card); border-radius: 12px; padding: 20px 24px;
        border-left: 4px solid var(--accent); margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-card.success { border-left-color: var(--success); }
    .metric-card.danger  { border-left-color: var(--danger); }
    .metric-card.warn    { border-left-color: var(--warn); }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: var(--accent); }
    .metric-card.success .metric-value { color: var(--success); }
    .metric-card.danger  .metric-value { color: var(--danger); }
    .metric-card.warn    .metric-value { color: var(--warn); }
    .metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase;
                    letter-spacing: 1px; margin-top: 2px; }
    .metric-sub   { font-size: 0.9rem; color: #ccc; margin-top: 4px; }
    .section-title { font-size: 1.3rem; font-weight: 700; color: #fff;
                     border-bottom: 2px solid #00d4ff33; padding-bottom: 8px;
                     margin: 24px 0 16px; }
    div[data-testid="stSidebarNav"] { background: #111320; }
    .stSelectbox > div, .stMultiSelect > div { background: #1a1d27 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27',
    font=dict(color='#e0e0e0'),
    xaxis=dict(gridcolor='#2d3142', linecolor='#2d3142'),
    yaxis=dict(gridcolor='#2d3142', linecolor='#2d3142'),
)

# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)
    path = os.path.join(root, 'data', 'call_records_cleaned.csv')
    df = pd.read_csv(path, parse_dates=['call_date'])
    df['month_str'] = df['call_date'].dt.to_period('M').astype(str)
    return df

df_full = load_data()

# ══════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/phone-office.png", width=60)
    st.title("📞 Call Analytics")
    st.markdown("---")
    st.subheader("🔎 Filters")

    date_min = df_full['call_date'].min().date()
    date_max = df_full['call_date'].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

    cities_all = sorted(df_full['city'].unique())
    sel_cities = st.multiselect("City", cities_all, default=[])

    statuses_all = sorted(df_full['call_status'].unique())
    sel_status   = st.multiselect("Call Status", statuses_all, default=[])

    call_types = sorted(df_full['call_type'].unique())
    sel_types  = st.multiselect("Call Type", call_types, default=[])

    langs_all = sorted(df_full['language'].dropna().unique())
    sel_langs = st.multiselect("Language", langs_all, default=[])

    st.markdown("---")
    st.caption("Customer Call Pattern Analysis v1.0")

# ── Apply Filters ─────────────────────────────────────────
df = df_full.copy()
if len(date_range) == 2:
    df = df[(df['call_date'].dt.date >= date_range[0]) &
            (df['call_date'].dt.date <= date_range[1])]
if sel_cities:  df = df[df['city'].isin(sel_cities)]
if sel_status:  df = df[df['call_status'].isin(sel_status)]
if sel_types:   df = df[df['call_type'].isin(sel_types)]
if sel_langs:   df = df[df['language'].isin(sel_langs)]

if len(df) == 0:
    st.warning("No data matches the current filters. Please adjust.")
    st.stop()

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("# 📞 Customer Call Pattern Analytics")
st.markdown(f"**Showing `{len(df):,}` calls** | {df['call_date'].min().date()} → {df['call_date'].max().date()}")
st.markdown("---")

# ══════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════
total      = len(df)
completed  = (df['call_status'] == 'Completed').sum()
failed     = (df['call_status'] == 'Failed').sum()
avg_dur    = df['duration_min'].mean()
success_rt = completed / total * 100
fail_rt    = failed / total * 100
unique_c   = df['customer_id'].nunique()
peak_h     = df.groupby('hour').size().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class="metric-card">
    <div class="metric-value">{total:,}</div>
    <div class="metric-label">Total Calls</div>
    <div class="metric-sub">{unique_c:,} unique customers</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""<div class="metric-card success">
    <div class="metric-value">{success_rt:.1f}%</div>
    <div class="metric-label">Success Rate</div>
    <div class="metric-sub">{completed:,} completed calls</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""<div class="metric-card danger">
    <div class="metric-value">{fail_rt:.1f}%</div>
    <div class="metric-label">Failure Rate</div>
    <div class="metric-sub">{failed:,} failed calls</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""<div class="metric-card warn">
    <div class="metric-value">{avg_dur:.1f} min</div>
    <div class="metric-label">Avg Duration</div>
    <div class="metric-sub">Peak hour: {peak_h}:00</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ROW 1: Volume Trend + Hourly Bar
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Call Volume Trends</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    monthly = df.groupby('month_str').agg(
        total=('call_id', 'count'),
        completed=('is_success', 'sum')
    ).reset_index()
    monthly['failed'] = monthly['total'] - monthly['completed']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['month_str'], y=monthly['total'],
                             mode='lines+markers', name='Total', line=dict(color='#00d4ff', width=2.5),
                             fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'))
    fig.add_trace(go.Scatter(x=monthly['month_str'], y=monthly['completed'],
                             mode='lines+markers', name='Completed', line=dict(color='#00ff88', width=2)))
    fig.update_layout(**PLOTLY_DARK, title='Monthly Call Volume', height=320,
                      legend=dict(orientation='h', y=1.1),
                      xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    hourly = df.groupby('hour').size().reset_index(name='calls')
    hourly['color'] = hourly['calls'].apply(lambda x: '#ffa502' if x == hourly['calls'].max() else '#00d4ff')
    fig = px.bar(hourly, x='hour', y='calls', color='color',
                 color_discrete_map='identity', title='Calls by Hour')
    fig.update_layout(**PLOTLY_DARK, height=320, showlegend=False,
                      xaxis_title='Hour', yaxis_title='Calls')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ROW 2: Status Donut + Heatmap
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Call Status & Patterns</div>', unsafe_allow_html=True)
col3, col4 = st.columns([1, 2])

with col3:
    status_c = df['call_status'].value_counts().reset_index()
    status_c.columns = ['status', 'count']
    fig = px.pie(status_c, names='status', values='count', hole=0.55,
                 color_discrete_sequence=['#00ff88','#ff4757','#ffa502','#00d4ff','#a29bfe'],
                 title='Call Status Distribution')
    fig.update_layout(**PLOTLY_DARK, height=340,
                      legend=dict(orientation='h', y=-0.15))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col4:
    DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = df.groupby(['day_of_week','hour']).size().unstack(fill_value=0)
    heat = heat.reindex([d for d in DOW_ORDER if d in heat.index])
    fig = px.imshow(heat, color_continuous_scale='YlOrRd',
                    title='Call Volume Heatmap (Hour × Day of Week)',
                    labels=dict(x='Hour of Day', y='Day', color='Calls'))
    fig.update_layout(**PLOTLY_DARK, height=340)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ROW 3: City Analysis
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🏙️ City-wise Analysis</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    city_agg = df.groupby('city').agg(
        total=('call_id','count'),
        sr=('is_success','mean')
    ).sort_values('total', ascending=False).head(15).reset_index()
    city_agg['sr_pct'] = (city_agg['sr'] * 100).round(1)
    fig = px.bar(city_agg, x='total', y='city', orientation='h',
                 color='sr_pct', color_continuous_scale='RdYlGn',
                 title='Top 15 Cities — Volume & Success Rate',
                 labels={'total':'Calls','sr_pct':'Success%'})
    fig.update_layout(**PLOTLY_DARK, height=420, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col6:
    state_agg = df.groupby('state').size().reset_index(name='calls').sort_values('calls', ascending=False)
    fig = px.bar(state_agg, x='state', y='calls', color='calls',
                 color_continuous_scale='Blues', title='State-wise Call Distribution')
    fig.update_layout(**PLOTLY_DARK, height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ROW 4: Flow + Language
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🤖 Voicebot Flow & Language</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)

with col7:
    flow_agg = df.groupby('voicebot_flow').agg(
        total=('call_id','count'),
        avg_dur=('duration_min','mean'),
        sr=('is_success','mean')
    ).sort_values('total', ascending=False).reset_index()
    fig = px.scatter(flow_agg, x='avg_dur', y='sr', size='total',
                     color='total', text='voicebot_flow',
                     color_continuous_scale='Viridis',
                     title='Flow: Avg Duration vs Success Rate (size=volume)',
                     labels={'avg_dur':'Avg Duration (min)','sr':'Success Rate'})
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(**PLOTLY_DARK, height=380)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    lang_agg = df.groupby('language').agg(
        total=('call_id','count'),
        sr=('is_success','mean')
    ).sort_values('total', ascending=False).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=lang_agg['language'], y=lang_agg['total'],
                         name='Volume', marker_color='#00d4ff', opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=lang_agg['language'], y=lang_agg['sr']*100,
                             name='Success Rate %', mode='lines+markers',
                             line=dict(color='#ffa502', width=2.5)), secondary_y=True)
    fig.update_layout(**PLOTLY_DARK, title='Language: Volume & Success Rate', height=380,
                      legend=dict(orientation='h', y=1.12))
    fig.update_yaxes(title_text='Calls', secondary_y=False)
    fig.update_yaxes(title_text='Success Rate (%)', secondary_y=True, range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# ROW 5: Duration + Repeat Callers
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">⏱️ Duration & Customer Behaviour</div>', unsafe_allow_html=True)
col9, col10 = st.columns(2)

with col9:
    fig = px.histogram(df, x='duration_min', nbins=60, color='call_status',
                       color_discrete_sequence=['#00ff88','#ff4757','#ffa502','#00d4ff','#a29bfe'],
                       title='Call Duration Distribution by Status',
                       labels={'duration_min':'Duration (min)'})
    fig.update_layout(**PLOTLY_DARK, height=360, barmode='overlay',
                      legend=dict(orientation='h', y=1.12))
    fig.update_traces(opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

with col10:
    if 'call_count' in df.columns:
        repeat_dist = df.groupby('customer_id')['call_count'].first().clip(upper=10)
        freq = repeat_dist.value_counts().sort_index().reset_index()
        freq.columns = ['calls', 'customers']
        fig = px.bar(freq, x='calls', y='customers', color='customers',
                     color_continuous_scale='Plasma',
                     title='Customer Call Frequency (capped at 10)',
                     labels={'calls':'Number of Calls','customers':'Customers'})
        fig.update_layout(**PLOTLY_DARK, height=360)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# RAW DATA TABLE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🗃️ Raw Data Explorer</div>', unsafe_allow_html=True)
cols_show = ['call_id','customer_id','call_date','call_time','call_duration',
             'call_status','city','state','voicebot_flow','call_type','language','call_disposition']
st.dataframe(
    df[cols_show].head(500).style.applymap(
        lambda v: 'color: #00ff88' if v == 'Completed' else (
                  'color: #ff4757' if v == 'Failed' else ''),
        subset=['call_status']
    ),
    use_container_width=True, height=350
)
st.caption(f"Showing first 500 of {len(df):,} filtered records")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555; font-size:0.8rem'>Customer Call Pattern Analysis • Built with Streamlit & Plotly</center>",
    unsafe_allow_html=True
)
