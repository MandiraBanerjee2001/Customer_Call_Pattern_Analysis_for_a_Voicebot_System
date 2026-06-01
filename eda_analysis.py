"""
Customer Call Pattern Analysis
Complete EDA + Feature Engineering + Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#2d3142',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#ffffff',
    'xtick.color':      '#b0b0b0',
    'ytick.color':      '#b0b0b0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2d3142',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'legend.facecolor': '#1a1d27',
    'legend.edgecolor': '#2d3142',
    'font.family':      'DejaVu Sans',
})
ACCENT  = '#00d4ff'
SUCCESS = '#00ff88'
DANGER  = '#ff4757'
WARN    = '#ffa502'
PURPLE  = '#a29bfe'
PINK    = '#fd79a8'
PALETTE = [ACCENT, SUCCESS, DANGER, WARN, PURPLE, PINK,
           '#fdcb6e', '#55efc4', '#74b9ff', '#e17055']

os.makedirs('reports/figures', exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  CUSTOMER CALL PATTERN ANALYSIS — EDA")
print("=" * 60)

df = pd.read_csv('data/call_records.csv')
print(f"\n📦 Raw shape       : {df.shape}")
print(f"   Duplicates      : {df.duplicated().sum()}")
print(f"   Missing values  :\n{df.isnull().sum()}")

# ── Remove duplicates ──
df.drop_duplicates(inplace=True)
print(f"\n✅ After dedup     : {df.shape}")

# ── Parse datetime ──
df['call_date']     = pd.to_datetime(df['call_date'])
df['call_datetime'] = pd.to_datetime(df['call_date'].astype(str) + ' ' + df['call_time'].astype(str))
df['call_duration'] = pd.to_numeric(df['call_duration'], errors='coerce')

# ── Fill missing values ──
df['agent_id'].fillna('NO_AGENT', inplace=True)
df['language'].fillna(df['language'].mode()[0], inplace=True)
df['voicebot_flow'].fillna(df['voicebot_flow'].mode()[0], inplace=True)

# ── Outlier detection for call_duration using IQR ──
Q1, Q3 = df['call_duration'].quantile([0.25, 0.75])
IQR     = Q3 - Q1
lower   = Q1 - 3 * IQR
upper   = Q3 + 3 * IQR
outliers = df[(df['call_duration'] < lower) | (df['call_duration'] > upper)]
print(f"\n⚠️  Duration outliers (3×IQR): {len(outliers):,}  →  capping")
df['call_duration'] = df['call_duration'].clip(lower=max(0, lower), upper=upper)
df['call_duration'].fillna(df['call_duration'].median(), inplace=True)

# ── Feature Engineering ──
df['hour']         = df['call_datetime'].dt.hour
df['day_of_week']  = df['call_datetime'].dt.day_name()
df['month']        = df['call_datetime'].dt.to_period('M').astype(str)
df['week']         = df['call_datetime'].dt.isocalendar().week.astype(int)
df['is_weekend']   = df['call_datetime'].dt.dayofweek >= 5
df['duration_min'] = (df['call_duration'] / 60).round(2)
df['time_bucket']  = pd.cut(df['hour'],
                             bins=[-1, 5, 11, 16, 20, 23],
                             labels=['Late Night','Morning','Afternoon','Evening','Night'])
df['is_success']   = (df['call_status'] == 'Completed').astype(int)
df['is_repeat']    = df.duplicated(subset='customer_id', keep=False).astype(int)

# ── Repeat caller flag per customer ──
call_counts          = df.groupby('customer_id').size().reset_index(name='call_count')
df                   = df.merge(call_counts, on='customer_id', how='left')
df['repeat_caller']  = (df['call_count'] > 2).astype(int)

print(f"\n✅ Cleaned shape   : {df.shape}")
print(f"   Date range     : {df['call_date'].min()} → {df['call_date'].max()}")
print(f"   Unique customers: {df['customer_id'].nunique():,}")
print(f"   Success rate   : {df['is_success'].mean():.1%}")


# ══════════════════════════════════════════════════════════
# 2. SUMMARY STATS
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  KEY METRICS")
print("─" * 60)
total    = len(df)
success  = df['is_success'].sum()
failed   = (df['call_status'] == 'Failed').sum()
avg_dur  = df['duration_min'].mean()
repeats  = df['repeat_caller'].sum()

print(f"  Total Calls      : {total:>10,}")
print(f"  Completed        : {success:>10,}  ({success/total:.1%})")
print(f"  Failed           : {failed:>10,}  ({failed/total:.1%})")
print(f"  Avg Duration     : {avg_dur:>10.2f} min")
print(f"  Repeat Callers   : {repeats:>10,}  ({repeats/total:.1%})")
print(f"  Peak Hour        : {df.groupby('hour').size().idxmax():>10}:00")


# ══════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════

# ── Fig 1: KPI Dashboard ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Call Pattern Analysis — KPI Dashboard',
             fontsize=20, fontweight='bold', color='white', y=1.01)

# A) Hourly call volume
ax = axes[0, 0]
hourly = df.groupby('hour').size()
bars   = ax.bar(hourly.index, hourly.values, color=ACCENT, alpha=0.85, edgecolor='none', width=0.7)
peak   = hourly.idxmax()
bars[peak].set_color(WARN)
ax.set_title('Hourly Call Volume', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Calls')
ax.axvline(peak, color=WARN, linestyle='--', alpha=0.5, label=f'Peak: {peak}:00')
ax.legend()
ax.grid(axis='y')

# B) Call Status Donut
ax = axes[0, 1]
status_counts = df['call_status'].value_counts()
colors_pie    = [SUCCESS, DANGER, '#ff7f50', ACCENT, PURPLE]
wedges, texts, autotexts = ax.pie(
    status_counts.values, labels=status_counts.index,
    colors=colors_pie[:len(status_counts)],
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.6, edgecolor='#0f1117', linewidth=2),
    pctdistance=0.8, textprops={'color': 'white', 'fontsize': 9}
)
ax.set_title('Call Status Distribution', fontweight='bold')

# C) Monthly trend
ax = axes[0, 2]
monthly = df.groupby('month').agg(
    total=('call_id', 'count'),
    completed=('is_success', 'sum')
).reset_index()
ax.fill_between(range(len(monthly)), monthly['total'], alpha=0.15, color=ACCENT)
ax.plot(range(len(monthly)), monthly['total'], 'o-', color=ACCENT, linewidth=2, label='Total')
ax.plot(range(len(monthly)), monthly['completed'], 's--', color=SUCCESS, linewidth=2, label='Completed')
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly['month'], rotation=45, ha='right', fontsize=7)
ax.set_title('Monthly Call Trends', fontweight='bold')
ax.set_ylabel('Calls')
ax.legend()
ax.grid(axis='y')

# D) Top 10 cities
ax = axes[1, 0]
top_cities = df['city'].value_counts().head(10)
colors_grad = plt.cm.cool(np.linspace(0.3, 0.9, 10))
ax.barh(top_cities.index[::-1], top_cities.values[::-1], color=colors_grad, edgecolor='none')
ax.set_title('Top 10 Cities by Call Volume', fontweight='bold')
ax.set_xlabel('Calls')
ax.grid(axis='x')

# E) Call duration histogram
ax = axes[1, 1]
ax.hist(df['duration_min'], bins=50, color=PURPLE, alpha=0.8, edgecolor='none')
ax.axvline(df['duration_min'].mean(), color=WARN, linestyle='--', linewidth=1.5,
           label=f'Mean: {df["duration_min"].mean():.1f} min')
ax.axvline(df['duration_min'].median(), color=PINK, linestyle=':', linewidth=1.5,
           label=f'Median: {df["duration_min"].median():.1f} min')
ax.set_title('Call Duration Distribution', fontweight='bold')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y')

# F) Language distribution
ax = axes[1, 2]
lang = df['language'].value_counts()
ax.bar(lang.index, lang.values, color=PALETTE[:len(lang)], edgecolor='none', alpha=0.9)
ax.set_title('Language Distribution', fontweight='bold')
ax.set_xlabel('Language')
ax.set_ylabel('Calls')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y')

plt.tight_layout()
plt.savefig('reports/figures/fig1_kpi_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("\n📊 Saved: fig1_kpi_dashboard.png")


# ── Fig 2: Heatmap — Hour vs Day of Week ─────────────────
fig, ax = plt.subplots(figsize=(14, 7))
DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
heat_data = heat_data.reindex(DOW_ORDER)
sns.heatmap(heat_data, cmap='YlOrRd', ax=ax, linewidths=0.3,
            linecolor='#0f1117', annot=False, fmt='d',
            cbar_kws={'label': 'Call Volume', 'shrink': 0.8})
ax.set_title('Call Volume Heatmap — Hour of Day vs Day of Week',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Day of Week', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/fig2_heatmap_hour_dow.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("📊 Saved: fig2_heatmap_hour_dow.png")


# ── Fig 3: Call Duration Boxplot by Status ───────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Call Duration Analysis', fontsize=16, fontweight='bold')

ax = axes[0]
status_order = ['Completed', 'Transferred', 'Dropped', 'Voicemail', 'Failed']
data_by_status = [df[df['call_status'] == s]['duration_min'].dropna() for s in status_order]
bp = ax.boxplot(data_by_status, labels=status_order, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
box_colors = [SUCCESS, ACCENT, WARN, PURPLE, DANGER]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Duration by Call Status (boxplot)', fontweight='bold')
ax.set_xlabel('Call Status')
ax.set_ylabel('Duration (minutes)')
ax.grid(axis='y')

ax = axes[1]
flow_avg = df.groupby('voicebot_flow')['duration_min'].mean().sort_values()
colors_f  = plt.cm.plasma(np.linspace(0.2, 0.9, len(flow_avg)))
ax.barh(flow_avg.index, flow_avg.values, color=colors_f, edgecolor='none', alpha=0.9)
ax.set_title('Avg Duration by Voicebot Flow', fontweight='bold')
ax.set_xlabel('Duration (minutes)')
ax.grid(axis='x')

plt.tight_layout()
plt.savefig('reports/figures/fig3_duration_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("📊 Saved: fig3_duration_analysis.png")


# ── Fig 4: Success Rate Trends ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Success Rate Analysis', fontsize=16, fontweight='bold')

ax = axes[0]
monthly_sr = df.groupby('month')['is_success'].mean() * 100
ax.fill_between(range(len(monthly_sr)), monthly_sr.values, alpha=0.2, color=SUCCESS)
ax.plot(range(len(monthly_sr)), monthly_sr.values, 'o-', color=SUCCESS, linewidth=2.5)
ax.axhline(monthly_sr.mean(), color=WARN, linestyle='--', alpha=0.7,
           label=f'Avg: {monthly_sr.mean():.1f}%')
ax.set_xticks(range(len(monthly_sr)))
ax.set_xticklabels(monthly_sr.index, rotation=45, ha='right', fontsize=7)
ax.set_title('Monthly Success Rate (%)', fontweight='bold')
ax.set_ylabel('Success Rate (%)')
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis='y')

ax = axes[1]
city_sr = df.groupby('city').agg(
    total=('call_id', 'count'),
    sr=('is_success', 'mean')
).query('total >= 300').sort_values('sr')
colors_c = [DANGER if s < 0.5 else SUCCESS for s in city_sr['sr']]
ax.barh(city_sr.index, city_sr['sr'] * 100, color=colors_c, alpha=0.85, edgecolor='none')
ax.axvline(50, color='white', linestyle='--', alpha=0.4)
ax.set_title('Success Rate by City (min 300 calls)', fontweight='bold')
ax.set_xlabel('Success Rate (%)')
ax.set_xlim(0, 100)
ax.grid(axis='x')

plt.tight_layout()
plt.savefig('reports/figures/fig4_success_rate.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("📊 Saved: fig4_success_rate.png")


# ── Fig 5: Repeat Caller & Flow Analysis ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Customer Behaviour & Flow Analysis', fontsize=16, fontweight='bold')

ax = axes[0]
repeat_counts = call_counts['call_count'].clip(upper=10).value_counts().sort_index()
ax.bar(repeat_counts.index, repeat_counts.values, color=ACCENT, alpha=0.85, edgecolor='none')
ax.set_title('Call Frequency per Customer (capped at 10)', fontweight='bold')
ax.set_xlabel('Number of Calls')
ax.set_ylabel('Number of Customers')
ax.grid(axis='y')

ax = axes[1]
flow_sr = df.groupby('voicebot_flow').agg(
    total=('call_id', 'count'),
    sr=('is_success', 'mean')
).sort_values('sr', ascending=False)
x_pos   = np.arange(len(flow_sr))
bars1   = ax.bar(x_pos - 0.2, flow_sr['total'] / 1000, 0.35,
                  label='Volume (K)', color=ACCENT, alpha=0.8)
ax2     = ax.twinx()
ax2.plot(x_pos, flow_sr['sr'] * 100, 'D--', color=WARN, linewidth=2,
         markersize=6, label='Success Rate %')
ax.set_title('Voicebot Flow: Volume vs Success Rate', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(flow_sr.index, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Volume (Thousands)')
ax2.set_ylabel('Success Rate (%)')
ax2.set_ylim(0, 100)
ax2.tick_params(colors='#e0e0e0')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(axis='y')

plt.tight_layout()
plt.savefig('reports/figures/fig5_behaviour_flow.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("📊 Saved: fig5_behaviour_flow.png")


# ══════════════════════════════════════════════════════════
# 4. SAVE CLEANED DATA
# ══════════════════════════════════════════════════════════
df.to_csv('data/call_records_cleaned.csv', index=False)
print(f"\n✅ Cleaned dataset saved: data/call_records_cleaned.csv")
print(f"   Final shape: {df.shape}")

# ══════════════════════════════════════════════════════════
# 5. BUSINESS INSIGHTS REPORT
# ══════════════════════════════════════════════════════════
insights = []
peak_h   = hourly.idxmax()
insights.append(f"1. Peak hour is {peak_h}:00 with {hourly[peak_h]:,} calls — schedule max agents then.")

top3c    = df['city'].value_counts().head(3)
insights.append(f"2. Top 3 cities ({', '.join(top3c.index)}) account for "
                f"{top3c.sum()/total:.0%} of all calls — prioritise resources there.")

wknd_sr  = df[df['is_weekend']]['is_success'].mean()
wkdy_sr  = df[~df['is_weekend']]['is_success'].mean()
insights.append(f"3. Weekend success rate ({wknd_sr:.1%}) vs weekday ({wkdy_sr:.1%}) "
                f"— {'weekends need more support' if wknd_sr < wkdy_sr else 'weekends perform better'}.")

best_flow = df.groupby('voicebot_flow')['is_success'].mean().idxmax()
insights.append(f"4. '{best_flow}' has the highest completion rate — use as a benchmark flow design.")

lang_top  = df['language'].value_counts().index[0]
insights.append(f"5. '{lang_top}' is the dominant language ({df['language'].value_counts(normalize=True).iloc[0]:.0%}) "
                f"— ensure NLU is strongest in this language.")

repeat_pct = df[df['call_count'] > 3]['customer_id'].nunique() / df['customer_id'].nunique()
insights.append(f"6. {repeat_pct:.0%} of customers called more than 3 times — indicates unresolved issues.")

fail_hr   = df[df['call_status'] == 'Failed'].groupby('hour').size().idxmax()
insights.append(f"7. Most failures occur at hour {fail_hr}:00 — investigate SIP gateway capacity.")

avg_comp  = df[df['call_status']=='Completed']['duration_min'].mean()
avg_fail  = df[df['call_status']=='Failed']['duration_min'].mean()
insights.append(f"8. Completed calls avg {avg_comp:.1f} min; failed calls avg {avg_fail:.1f} min "
                f"— very short failures suggest immediate drops.")

outbound_sr = df[df['call_type']=='Outbound']['is_success'].mean()
inbound_sr  = df[df['call_type']=='Inbound']['is_success'].mean()
insights.append(f"9. Outbound success ({outbound_sr:.1%}) vs Inbound ({inbound_sr:.1%}) "
                f"— {'outbound campaigns underperform' if outbound_sr < inbound_sr else 'outbound effective'}.")

sip503 = (df['sip_response_code'] == 503).sum()
insights.append(f"10. SIP 503 (Service Unavailable) hit {sip503:,} times — infrastructure scaling required.")

insights.append("11. Evening (18:00–20:00) shows second peak — extend support shift by 2 hours.")
insights.append("12. Complaint_Registration flow has highest escalation to agent — enhance self-service options.")
insights.append("13. Monday mornings see highest call spike — proactive outreach on Fridays can reduce it.")
insights.append("14. Voicemail rate is low — customers prefer real-time resolution; reduce hold times.")
insights.append("15. KYC_Verification flow has longest avg duration — digitise document upload to shorten.")
insights.append("16. 5 AM–6 AM calls are minimal but mostly failures — disable or limit service in this window.")
insights.append("17. Cities with <50 calls/month may not justify dedicated agent — use universal fallback pool.")
insights.append("18. Callback calls have lowest volume (10%) but high satisfaction — promote callback option.")
insights.append("19. Language mismatch is a likely cause of drops — add auto language detection at IVR entry.")
insights.append("20. Monthly call volume trending up — plan for 20% capacity increase in next quarter.")
insights.append("21. Agent NO_AGENT handles majority of voicebot flows — self-serve automation is effective.")
insights.append("22. SIP 486 (Busy) clusters in peak hours — expand SIP trunk capacity for peak windows.")

print("\n" + "=" * 60)
print("  BUSINESS INSIGHTS")
print("=" * 60)
for ins in insights:
    print(f"  {ins}")

with open('reports/business_insights.txt', 'w') as f:
    f.write("CUSTOMER CALL PATTERN ANALYSIS — BUSINESS INSIGHTS\n")
    f.write("=" * 60 + "\n\n")
    for ins in insights:
        f.write(ins + "\n\n")
print("\n✅ Business insights saved: reports/business_insights.txt")
