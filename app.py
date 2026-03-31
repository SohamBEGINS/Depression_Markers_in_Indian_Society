import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# Set page config
st.set_page_config(layout="wide", page_title="Indian Distress Galaxy")

@st.cache_data
def load_data():
    df = pd.read_csv("distress_final_reduced.csv")
    df['date'] = pd.to_datetime(df['date'])
    # Ensure purity is calculated for marker sizing
    if 'purity' not in df.columns:
        df['purity'] = 1.0 - df['fuzziness']
    return df

df = load_data()

def compute_linguistic_markers(text):
    """
    Computes validated linguistic markers for a given post.
    Based on Al-Mosaiwi & Johnstone (2018) absolutism methodology
    and Pennebaker (1997) pronoun analysis.
    """
    
    # Validated word dictionaries
    absolutist_words = {
        "always", "never", "completely", "nothing",
        "everything", "everyone", "forever", "impossible",
        "entire", "only", "must", "certain", "totally",
        "absolutely", "every", "all", "no one", "nobody",
        "nowhere"
    }
    
    negative_self_words = {
        "failure", "worthless", "useless", "burden",
        "stupid", "idiot", "loser", "pathetic",
        "disgusting", "broken", "damaged", "ugly",
        "weak", "incapable", "incompetent", "waste",
        "hopeless", "helpless", "meaningless", "pointless",
        "terrible", "awful", "horrible", "worst"
    }
    
    positive_self_words = {
        "capable", "strong", "worthy", "enough",
        "good", "deserving", "valuable", "able",
        "confident", "proud", "happy", "better",
        "improving", "growing", "healing", "recovering"
    }
    
    first_person_singular = {
        "i", "me", "my", "myself", "mine"
    }
    
    third_person_external = {
        "they", "them", "their", "he", "she",
        "parents", "family", "society", "system",
        "everyone", "people", "others", "friends",
        "mother", "father", "mom", "dad", "boss",
        "college", "school", "government", "india"
    }
    
    # Tokenize
    words = text.lower().split()
    # Remove punctuation from words
    words = [re.sub(r'[^\w\s]', '', w) for w in words]
    words = [w for w in words if w]  # remove empty strings
    total = max(len(words), 1)
    
    # Compute frequencies
    absolutist_found = [w for w in words if w in absolutist_words]
    neg_self_found = [w for w in words if w in negative_self_words]
    pos_self_found = [w for w in words if w in positive_self_words]
    first_person_found = [w for w in words if w in first_person_singular]
    third_person_found = [w for w in words if w in third_person_external]
    
    absolutist_freq = len(absolutist_found) / total
    neg_self_freq = len(neg_self_found) / total
    pos_self_freq = len(pos_self_found) / total
    first_freq = len(first_person_found) / total
    third_freq = len(third_person_found) / total
    
    # Attribution ratio
    attribution_ratio = (first_freq + 0.001) / (third_freq + 0.001)
    
    if attribution_ratio > 1.5:
        attribution_label = "Internal (self-focused)"
        attribution_color = "#EF5350"  # red
    elif attribution_ratio < 0.7:
        attribution_label = "External (blames circumstances)"
        attribution_color = "#42A5F5"  # blue
    else:
        attribution_label = "Mixed"
        attribution_color = "#FFA726"  # orange
    
    # Cognitive distress score
    cognitive_distress = min(
        (absolutist_freq * 15) * 0.5 + 
        (neg_self_freq * 20) * 0.5,
        1.0
    )
    
    # Resilience signal
    resilience_signal = min(pos_self_freq * 20, 1.0)
    
    return {
        "absolutist_freq": absolutist_freq,
        "absolutist_found": absolutist_found,
        "neg_self_freq": neg_self_freq,
        "neg_self_found": neg_self_found,
        "pos_self_found": pos_self_found,
        "attribution_ratio": attribution_ratio,
        "attribution_label": attribution_label,
        "attribution_color": attribution_color,
        "cognitive_distress": cognitive_distress,
        "resilience_signal": resilience_signal,
        "word_count": total,
        "first_person_freq": first_freq,
        "third_person_freq": third_freq,
    }

# Initialize session state for the selected point if not already set
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0

st.title("🌌 Sociological Galaxy: Reddit India (2010-2026)")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Galaxy")

categories = df['super_cluster'].unique()
selected_cat = st.sidebar.multiselect("Select Stressors", categories, default=categories)

# Fuzziness Slider
fuzz_limit = st.sidebar.slider(
    "Fuzziness >=Threshold (Filter out posts)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.0001,
    format="%.4f"
)

# Filter the data for the main plot
# Note: Using <= for a 'Max' threshold to filter OUT high-fuzziness (vague) posts
filtered_df = df[
    (df['super_cluster'].isin(selected_cat)) & 
    (df['fuzziness'] >= fuzz_limit)
].copy()

# --- FEATURE C: PROBABILITY RADAR CHART & MANUAL AUDIT (SIDEBAR) ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Fuzzy Membership Audit")
st.sidebar.write("Click a point in the Galaxy OR enter an index below.")

# NEW: Manual Index Input box
# We use the session_state value as the starting point
manual_idx = st.sidebar.number_input(
    "Enter Post Index to Audit:",
    min_value=0,
    max_value=len(df)-1,
    value=int(st.session_state.selected_index),
    step=1
)

# If the manual input is changed, update the session state and rerun
if manual_idx != st.session_state.selected_index:
    st.session_state.selected_index = manual_idx
    st.rerun()

current_idx = st.session_state.selected_index

# Safety check for index range
if current_idx >= len(df): current_idx = 0

# Extract values for Radar Chart
prob_cols = [c for c in df.columns if c.startswith('prob_')]
radar_labels = [c.replace('prob_', '').replace('_', ' ').title() for c in prob_cols]
radar_values = df.loc[current_idx, prob_cols].values

# Create Radar Chart
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values,
    theta=radar_labels,
    fill='toself',
    name='Membership Degree',
    line_color='#00d4ff'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="gray"),
        bgcolor="rgba(0,0,0,0)"
    ),
    showlegend=False,
    template="plotly_dark",
    margin=dict(l=40, r=40, t=40, b=40),
    height=300
)

st.sidebar.plotly_chart(fig_radar, use_container_width=True)

# Show post details in sidebar
st.sidebar.markdown(f"**Selected Post Index: {current_idx}**")
st.sidebar.caption(f"**Content:** {df.loc[current_idx, 'full_text']}")

# --- LINGUISTIC MARKER ANALYSIS ---
post_content = str(df.loc[current_idx, 'full_text'])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Linguistic Marker Analysis")
st.sidebar.caption("Based on validated NLP methodology (Al-Mosaiwi & Johnstone, 2018)")

if len(post_content.split()) >= 20:
    markers = compute_linguistic_markers(post_content)
    
    # Row 1: Three metric cards
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        st.metric(
            label="Absolutist Thinking",
            value=f"{markers['absolutist_freq']*100:.1f}%",
            delta="of words",
            help="Words like 'never', 'always', 'nothing', 'everyone'. "
                 "Higher = more all-or-nothing thinking. "
                 "Western depression baseline: ~1.5% (Al-Mosaiwi 2018)"
        )
    
    with col2:
        st.metric(
            label="Negative Self-Ref",
            value=f"{markers['neg_self_freq']*100:.1f}%",
            delta="of words",
            help="Words like 'failure', 'worthless', 'burden', 'useless'. "
                 "Higher = stronger negative self-concept."
        )
    
    with col3:
        st.metric(
            label="Cognitive Distress",
            value=f"{markers['cognitive_distress']:.2f}",
            delta="/ 1.0 max",
            help="Combined score from absolutism + negative self-reference. "
                 "Weighted formula derived from ATQ literature."
        )
    
    # Row 2: Attribution analysis
    st.sidebar.markdown("**Attribution Style**")
    
    attr_col1, attr_col2 = st.sidebar.columns([2, 1])
    
    with attr_col1:
        st.progress(
            min(markers['attribution_ratio'] / 3.0, 1.0),
            text=f"{markers['attribution_label']}"
        )
        st.caption(
            f"First-person pronouns: {markers['first_person_freq']*100:.1f}% | "
            f"External references: {markers['third_person_freq']*100:.1f}% | "
            f"Ratio: {markers['attribution_ratio']:.2f}"
        )
    
    with attr_col2:
        st.markdown(
            f"<div style='background-color: {markers['attribution_color']}22; "
            f"border-left: 4px solid {markers['attribution_color']}; "
            f"padding: 8px 12px; border-radius: 4px; "
            f"color: {markers['attribution_color']}; font-weight: 600;'>"
            f"{markers['attribution_label']}</div>",
            unsafe_allow_html=True
        )
    
    # Row 3: Flagged words
    if markers['absolutist_found'] or markers['neg_self_found']:
        st.sidebar.markdown("**Flagged Words**")
        
        flag_col1, flag_col2 = st.sidebar.columns(2)
        
        with flag_col1:
            if markers['absolutist_found']:
                unique_abs = list(set(markers['absolutist_found']))[:8]
                st.markdown("🔴 **Absolutist language**")
                pills_html = " ".join([
                    f"<span style='background: #EF535022; color: #EF5350; "
                    f"padding: 2px 8px; border-radius: 12px; "
                    f"font-size: 12px; margin: 2px; display: inline-block;'>"
                    f"{word}</span>"
                    for word in unique_abs
                ])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown("🟢 No absolutist language detected")
        
        with flag_col2:
            if markers['neg_self_found']:
                unique_neg = list(set(markers['neg_self_found']))[:8]
                st.markdown("🔴 **Negative self-reference**")
                pills_html = " ".join([
                    f"<span style='background: #FF980022; color: #FF9800; "
                    f"padding: 2px 8px; border-radius: 12px; "
                    f"font-size: 12px; margin: 2px; display: inline-block;'>"
                    f"{word}</span>"
                    for word in unique_neg
                ])
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown("🟢 No negative self-reference detected")
    
    if markers['pos_self_found']:
        unique_pos = list(set(markers['pos_self_found']))[:8]
        st.sidebar.markdown("🟢 **Resilience signals detected**")
        pills_html = " ".join([
            f"<span style='background: #4CAF5022; color: #4CAF50; "
            f"padding: 2px 8px; border-radius: 12px; "
            f"font-size: 12px; margin: 2px; display: inline-block;'>"
            f"{word}</span>"
            for word in unique_pos
        ])
        st.sidebar.markdown(pills_html, unsafe_allow_html=True)
    
    # Row 4: Word count context
    st.sidebar.caption(f"Analysis based on {markers['word_count']} words. "
                       f"Minimum 20 words required for reliable analysis.")

else:
    st.sidebar.info("Post too short for linguistic analysis (minimum 20 words required)")

# --- MAIN GALAXY PLOT ---
# Pass the original index as custom_data to handle clicks accurately
fig = px.scatter(
    filtered_df, x='x', y='y',
    color='super_cluster',
    size='purity',
    hover_data=['umbrella_marker', 'fuzziness'],
    custom_data=[filtered_df.index], 
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Prism,
    render_mode='webgl' 
)

fig.update_layout(
    height=700, 
    clickmode='event+select'
)

# Capture the click event using Streamlit's new interactive plotly component
event_data = st.plotly_chart(
    fig, 
    use_container_width=True, 
    on_select="rerun", 
    selection_mode="points"
)

# Update session state if a point is clicked in the plot
if event_data and "selection" in event_data:
    points = event_data["selection"]["points"]
    if points:
        new_idx = points[0]["customdata"][0]
        if new_idx != st.session_state.selected_index:
            st.session_state.selected_index = new_idx
            st.rerun()

# Data Table for quick reference
if st.checkbox("Show Data Table (Filtered)"):
    st.dataframe(filtered_df[['date', 'super_cluster', 'umbrella_marker', 'fuzziness', 'full_text']].head(100))