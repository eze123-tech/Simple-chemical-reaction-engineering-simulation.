"""
Batch Reactor Yield Optimization using Bayesian Optimization
============================================================
Consecutive reaction: A -> B -> C
Maximize yield of intermediate product B by tuning Temperature (T) and Time (t).

Deploy: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from skopt import Optimizer
from skopt.space import Real
from skopt.utils import use_named_args

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Batch Reactor BO",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants / kinetics
# ---------------------------------------------------------------------------
R = 8.314  # J/(mol·K)

# Default kinetic parameters (user-editable)
DEFAULT_KINETICS = {
    "A1": 1.0e6,    # pre-exponential A->B [1/min]
    "Ea1": 50_000.0,  # activation energy A->B [J/mol]
    "A2": 2.0e8,    # pre-exponential B->C [1/min]
    "Ea2": 70_000.0,  # activation energy B->C [J/mol]
}


def batch_reactor_yield(T, t, kinetics):
    """Yield of intermediate B in batch reactor A -> B -> C (first-order)."""
    A1, Ea1 = kinetics["A1"], kinetics["Ea1"]
    A2, Ea2 = kinetics["A2"], kinetics["Ea2"]

    k1 = A1 * np.exp(-Ea1 / (R * T))
    k2 = A2 * np.exp(-Ea2 / (R * T))

    # Safe evaluation
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_B = (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))

    yield_B = np.where(np.isnan(yield_B), k1 * t * np.exp(-k1 * t), yield_B)
    return np.clip(yield_B, 0.0, 1.0)


def batch_reactor_purity(T, t, kinetics):
    """Selectivity of B among products formed (B / (B + C))."""
    A1, Ea1 = kinetics["A1"], kinetics["Ea1"]
    A2, Ea2 = kinetics["A2"], kinetics["Ea2"]
    k1 = A1 * np.exp(-Ea1 / (R * T))
    k2 = A2 * np.exp(-Ea2 / (R * T))

    CA = np.exp(-k1 * t)
    CB = (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    CC = 1.0 - CA - CB

    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.where((CB + CC) > 0, CB / (CB + CC), 0.0)
    return np.clip(purity, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "opt_history": [],       # list of dicts: T, t, yield, iteration
        "optimizer": None,
        "best_T": None,
        "best_t": None,
        "best_yield": None,
        "running": False,
        "kinetics": DEFAULT_KINETICS.copy(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Optimization Settings")

T_min = st.sidebar.slider("Temperature min [K]", 280.0, 340.0, 300.0, 1.0)
T_max = st.sidebar.slider("Temperature max [K]", 340.0, 450.0, 400.0, 1.0)
t_min = st.sidebar.slider("Time min [min]", 1.0, 30.0, 10.0, 1.0)
t_max = st.sidebar.slider("Time max [min]", 60.0, 240.0, 120.0, 10.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Bayesian Optimization")
n_initial = st.sidebar.number_input("Initial random points", 2, 20, 5, 1)
n_calls = st.sidebar.number_input("Total BO evaluations", 10, 100, 25, 1)
acq_func = st.sidebar.selectbox(
    "Acquisition function", ["EI", "PI", "LCB", "gp_hedge"], index=0
)
random_seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Constraints (optional)")
min_purity = st.sidebar.slider(
    "Minimum purity B/(B+C)", 0.0, 1.0, 0.90, 0.01,
    help="Constraint applied as a large penalty in the objective."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_home, tab_run, tab_results, tab_surface, tab_about = st.tabs(
    ["🏠 Overview", "🚀 Run Optimization", "📊 Results", "🗺️ Yield Surface", "ℹ️ About"]
)

# --- Overview ---------------------------------------------------------------
with tab_home:
    st.title("⚗️ Batch Reactor Yield Optimization")
    st.markdown(
        """
        ### Problem
        A batch reactor carries out **consecutive first-order reactions**:

        $$A \\xrightarrow{k_1} B \\xrightarrow{k_2} C$$

        - **B** is the desired product.
        - **C** is an unwanted byproduct.
        - Kinetics: $k_i = A_i \\exp(-E_{a,i}/(RT))$

        ### The Trade-off
        - **High T** → faster reactions, but **more C** formed (higher selectivity loss).
        - **Low T** → high selectivity to B, but **slow conversion** (long time needed).

        ### Objective
        Maximize **yield of B** over the search space:
        - Temperature: 300–400 K
        - Time: 10–120 min

        ### Why Bayesian Optimization?
        The yield surface is **non-convex** with an interior maximum.
        BO finds the optimum in ~20–25 evaluations — dramatically cheaper
        than a brute-force grid for realistic (expensive) experiments.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision variables", "2")
    col2.metric("Typical optimum", "T ≈ 350 K, t ≈ 30 min")
    col3.metric("Typical yield", "~65 %")

# --- Run optimization -------------------------------------------------------
with tab_run:
    st.title("🚀 Run Bayesian Optimization")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        start = st.button("▶️ Start / Continue Optimization", type="primary", use_container_width=True)
    with col_b:
        reset = st.button("🔄 Reset", use_container_width=True)

    if reset:
        st.session_state.opt_history = []
        st.session_state.optimizer = None
        st.session_state.best_T = None
        st.session_state.best_t = None
        st.session_state.best_yield = None
        st.success("Optimization reset.")
        st.rerun()

    # Initialize optimizer if needed
    if start:
        if st.session_state.optimizer is None:
            search_space = [Real(T_min, T_max, name="T"), Real(t_min, t_max, name="t")]
            st.session_state.optimizer = Optimizer(
                dimensions=search_space,
                base_estimator="GP",
                acq_func=acq_func,
                random_state=int(random_seed),
                n_initial_points=n_initial,
            )

        # Continue until we've done n_calls total
        history = st.session_state.opt_history
        n_done = len(history)
        n_target = int(n_calls)

        progress = st.progress(0.0, text="Running Bayesian Optimization...")
        placeholder = st.empty()

        while n_done < n_target:
            x = st.session_state.optimizer.ask()
            T, t = float(x[0]), float(x[1])

            y = batch_reactor_yield(T, t, st.session_state.kinetics)
            purity = batch_reactor_purity(T, t, st.session_state.kinetics)

            # Penalty if purity constraint violated
            y_obs = y if purity >= min_purity else y - 1.0

            st.session_state.optimizer.tell(x, -y_obs)  # minimize negative
            history.append(
                {
                    "iteration": n_done,
                    "T": T,
                    "t": t,
                    "yield": y,
                    "purity": purity,
                    "y_obs": y_obs,
                }
            )
            n_done += 1

            # Update best
            best = max(history, key=lambda r: r["yield"] if r["purity"] >= min_purity else -1)
            st.session_state.best_T = best["T"]
            st.session_state.best_t = best["t"]
            st.session_state.best_yield = best["yield"]

            progress.progress(
                min(n_done / n_target, 1.0),
                text=f"Iteration {n_done}/{n_target} — best yield = {st.session_state.best_yield*100:.2f}%",
            )

        progress.progress(1.0, text="✅ Optimization complete!")
        st.success("Optimization finished.")

    # Live display
    if st.session_state.opt_history:
        hist_df = pd.DataFrame(st.session_state.opt_history)
        st.subheader("Convergence")
        hist_df["running_best"] = hist_df["yield"].cummax()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist_df["iteration"],
                y=hist_df["yield"] * 100,
                mode="markers+lines",
                name="Yield (observed)",
                marker=dict(size=8, opacity=0.6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df["iteration"],
                y=hist_df["running_best"] * 100,
                mode="lines",
                name="Best so far",
                line=dict(width=3, color="crimson"),
            )
        )
        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Yield of B [%]",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Click **Start Optimization** to begin. Initial points are sampled randomly, then BO takes over.")

# --- Results ----------------------------------------------------------------
with tab_results:
    st.title("📊 Results")

    if not st.session_state.opt_history:
        st.warning("No optimization has been run yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.opt_history)

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Yield", f"{st.session_state.best_yield*100:.2f} %")
        c2.metric("Optimal T", f"{st.session_state.best_T:.1f} K")
        c3.metric("Optimal t", f"{st.session_state.best_t:.1f} min")

        best_row = hist_df.loc[hist_df["yield"].idxmax()]
        st.markdown(
            f"**Purity at optimum:** {best_row['purity']*100:.2f} %  "
            f"| **Total evaluations:** {len(hist_df)}"
        )

        st.subheader("All Evaluated Points")
        display_df = hist_df.copy()
        display_df["yield"] = (display_df["yield"] * 100).round(2)
        display_df["purity"] = (display_df["purity"] * 100).round(2)
        display_df = display_df.rename(
            columns={"yield": "Yield [%]", "purity": "Purity [%]", "y_obs": "Penalized score"}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results (CSV)",
            data=csv,
            file_name="batch_reactor_bo_results.csv",
            mime="text/csv",
        )

        # Scatter of all evaluations in parameter space
        st.subheader("Search Space Coverage")
        fig2 = px.scatter(
            hist_df,
            x="T",
            y="t",
            color="yield",
            color_continuous_scale="Viridis",
            size=hist_df["yield"] * 20 + 5,
            hover_data=["iteration", "purity"],
            labels={"T": "Temperature [K]", "t": "Time [min]", "yield": "Yield"},
        )
        fig2.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig2, use_container_width=True)

# --- Yield surface ----------------------------------------------------------
with tab_surface:
    st.title("🗺️ Yield Surface & Optimum")

    T_grid = np.linspace(T_min, T_max, 100)
    t_grid = np.linspace(t_min, t_max, 100)
    TT, tt = np.meshgrid(T_grid, t_grid)
    ZZ = batch_reactor_yield(TT, tt, st.session_state.kinetics) * 100

    fig = go.Figure(
        data=go.Contour(
            x=T_grid,
            y=t_grid,
            z=ZZ,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="Yield [%]"),
        )
    )

    if st.session_state.best_T is not None:
        fig.add_trace(
            go.Scatter(
                x=[st.session_state.best_T],
                y=[st.session_state.best_t],
                mode="markers",
                marker=dict(symbol="star", size=18, color="red", line=dict(color="white", width=2)),
                name="BO optimum",
            )
        )

    if st.session_state.opt_history:
        hist_df = pd.DataFrame(st.session_state.opt_history)
        fig.add_trace(
            go.Scatter(
                x=hist_df["T"],
                y=hist_df["t"],
                mode="markers",
                marker=dict(size=6, color="white", line=dict(color="black", width=1)),
                name="Evaluated points",
            )
        )

    fig.update_layout(
        xaxis_title="Temperature [K]",
        yaxis_title="Time [min]",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Contour of the analytical yield. The red star is the best point found by BO. "
        "White dots are evaluated points — note how BO clusters near the optimum after exploration."
    )

# --- About ------------------------------------------------------------------
with tab_about:
    st.title("ℹ️ About this App")
    st.markdown(
        r"""
        ### Model
        First-order consecutive reactions in an isothermal batch reactor:

        $$
        \frac{dC_A}{dt} = -k_1 C_A \qquad
        \frac{dC_B}{dt} = k_1 C_A - k_2 C_B \qquad
        \frac{dC_C}{dt} = k_2 C_B
        $$

        With $k_i = A_i \exp(-E_{a,i}/(RT))$, the analytical solution gives:

        $$
        Y_B = \frac{k_1}{k_2 - k_1}\left(e^{-k_1 t} - e^{-k_2 t}\right)
        $$

        ### Default kinetics
        | Reaction | A [1/min] | Ea [J/mol] |
        |----------|-----------|-------------|
        | A → B | 1.0 × 10⁶ | 50 000 |
        | B → C | 2.0 × 10⁸ | 70 000 |

        Since $E_{a,2} > E_{a,1}$, higher temperatures **accelerate the undesired
        reaction more strongly**, producing the trade-off that makes this an
        interesting optimization problem.

        ### Bayesian Optimization
        - **Surrogate model**: Gaussian Process (`skopt` GP)
        - **Acquisition**: EI / PI / LCB / gp_hedge (user selectable)
        - **Initialization**: random points (`n_initial_points`)
        - **Constraint**: purity B/(B+C) enforced via penalty

        ### References
        - Brochu, E., Cora, V. M., & de Freitas, N. (2010). *A Tutorial on Bayesian Optimization of Expensive Cost Functions*.
        - Fogler, H. S. *Elements of Chemical Reaction Engineering* — Chapter 8 (consecutive reactions).
        - `scikit-optimize` documentation: https://scikit-optimize.github.io/
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit + scikit-optimize · Batch reactor BO demo")