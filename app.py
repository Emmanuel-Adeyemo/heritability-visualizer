
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import simulate as sim

st.set_page_config(page_title='Parent-Offspring Regression Simulator', layout='wide')

PRESETS = {
    '-- choose a preset --': None,
    'High h2, tight scatter': {'VA': 20.0, 'VE': 5.0, 'n_families': 200, 'offspring_per_family': 1},
    'Low h2, loose scatter': {'VA': 5.0, 'VE': 20.0, 'n_families': 200, 'offspring_per_family': 1},
    'Small sample, high noise': {'VA': 10.0, 'VE': 15.0, 'n_families': 15, 'offspring_per_family': 1},
    'Averaging tightens the fit': {'VA': 10.0, 'VE': 15.0, 'n_families': 200, 'offspring_per_family': 8},
}

st.title('Parent-Offspring Regression Simulator')
st.latex(r'\text{offspring phenotype} = a + h^2 \times \text{midparent phenotype} + \text{noise}')

for key, default in [
    ('VA_slider', 10.0),
    ('VE_slider', 15.0),
    ('n_families_slider', 100),
    ('offspring_k_slider', 1),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def apply_preset():
    preset = PRESETS[st.session_state.preset_choice]
    if preset is None:
        return
    if 'VA' in preset:
        st.session_state.VA_slider = preset['VA']
    if 'VE' in preset:
        st.session_state.VE_slider = preset['VE']
    if 'n_families' in preset:
        st.session_state.n_families_slider = preset['n_families']
    if 'offspring_per_family' in preset:
        st.session_state.offspring_k_slider = preset['offspring_per_family']


st.selectbox('Preset scenario', list(PRESETS.keys()), key='preset_choice', on_change=apply_preset)

st.divider()


# top panel
st.header('Panel A: Variance components')

col_a1, col_a2 = st.columns(2)
with col_a1:
    VA = st.slider('Additive genetic variance (VA)', min_value=0.0, max_value=40.0,
                    step=0.5, key='VA_slider')
with col_a2:
    VE = st.slider('Environmental variance (VE)', min_value=0.1, max_value=40.0,
                    step=0.5, key='VE_slider')

VP = VA + VE
true_h2 = sim.calc_heritability(VA, VE)

fig_a = go.Figure()
fig_a.add_trace(go.Bar(
    x=['Phenotypic variance'], y=[VA], name='VA (additive genetic)',
    marker_color='#4C72B0', orientation='v',
))
fig_a.add_trace(go.Bar(
    x=['Phenotypic variance'], y=[VE], name='VE (environmental)',
    marker_color='#C44E52', orientation="v",
))
fig_a.update_layout(
    barmode="stack", height=280, showlegend=True,
    yaxis_title='Variance',
    margin=dict(t=20, b=20),
)
col_chart, col_metric = st.columns([2, 1])
with col_chart:
    st.plotly_chart(fig_a, use_container_width=True)
with col_metric:
    st.metric('VP = VA + VE', f'{VP:.1f}')
    st.metric('h² = VA / VP', f'{true_h2:.3f}')
    st.caption('This is the true heritability of the simulated population. '
               'Watch for this exact number to reappear as the regression slope in Panel B.')

st.divider()

# parent offspring scatter plot
st.header('Panel B: Parent-offspring regression')

col_b1, col_b2 = st.columns(2)
with col_b1:
    n_families = st.slider('Number of families (parent pairs)', min_value=5, max_value=500,
                            step=5, key='n_families_slider')
with col_b2:
    offspring_k = st.slider('Offspring per family (averaged)', min_value=1, max_value=10,
                             step=1, key='offspring_k_slider',
                             help="Averaging more offspring per family reduces scatter from "
                                  "Mendelian sampling and environmental noise -- it does NOT "
                                  "change the true h², only how precisely you can see it.")

if 'sim_seed' not in st.session_state:
    st.session_state.sim_seed = 785

col_resim, _ = st.columns([1, 3])
with col_resim:
    if st.button('Resimulate (new random families)'):
        st.session_state.sim_seed += 1

rng = np.random.default_rng(st.session_state.sim_seed)
data = sim.simulate_parent_offspring(VA, VE, n_families=n_families,
                                  offspring_per_family=offspring_k, rng=rng)
fit = sim.fit_regression(data['midparent_pheno'], data['offspring_pheno'])

mu_ref = 100.0
x_line_true = np.array([data['midparent_pheno'].min(), data['midparent_pheno'].max()])
y_line_true = mu_ref + true_h2 * (x_line_true - mu_ref)

fig_b = go.Figure()
fig_b.add_trace(go.Scatter(
    x=data['midparent_pheno'], y=data['offspring_pheno'], mode='markers',
    name='Families', marker=dict(color='#4C72B0', size=6, opacity=0.55),
))
fig_b.add_trace(go.Scatter(
    x=fit['x_fit'], y=fit['y_fit'], mode='lines',
    name=f"Fitted' regression (slope = {fit['slope']:.3f})",
    line=dict(color='#DD8452', width=3),
))
fig_b.add_trace(go.Scatter(
    x=x_line_true, y=y_line_true, mode='lines',
    name=f'True h² reference (slope = {true_h2:.3f})',
    line=dict(color='#55A868', width=2, dash='dash'),
))
fig_b.update_layout(
    height=480,
    xaxis_title='Midparent phenotype',
    yaxis_title='Offspring phenotype',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(t=60),
)
st.plotly_chart(fig_b, use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric('True h²', f'{true_h2:.3f}')
m2.metric('Fitted slope (estimated h²)', f"{fit['slope']:.3f}")
m3.metric('R² of fit', f"{(fit['r_value']**2):.3f}")

gap = fit['slope'] - true_h2
st.caption(
    f"The fitted slope differs from the true h² by {gap:+.3f} in this particular sample of "
    f"{n_families} families. Click 'Resimulate' to draw a new random sample at the same settings "
    "and watch the fitted line jitter around the true (green dashed) line -- this is sampling "
    "noise, not model error. Smaller family counts and fewer offspring per family produce more "
    "jitter; increasing either tightens the estimate toward the true value."
)