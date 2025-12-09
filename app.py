import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import enum
import streamlit as st
from models import BSHWModel, HestonHWModel, SZHWModel, DiversificationModel

# Set page config first
st.set_page_config(page_title="Hybrid Financial Models", layout="wide")

def main():
    st.title("Hybrid Financial Models Dashboard")
    st.markdown("Comprehensive analysis of BSHW, Heston-HW, SZHW, and Diversification Product Models")
    
    # Sidebar navigation
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Home", "BSHW Model", "Heston-HW Model", "SZHW Model", "Diversification Products"]
    )
    
    if model_choice == "Home":
        show_home_page()
    elif model_choice == "BSHW Model":
        show_bshw_analysis_page()
    elif model_choice == "Heston-HW Model":
        show_heston_hw_analysis_page()
    elif model_choice == "SZHW Model":
        show_szhw_analysis_page()
    elif model_choice == "Diversification Products":
        show_diversification_analysis_page()

def show_home_page():
    st.header("Hybrid Models Overview")
    st.write("""
    This dashboard provides interactive analysis of advanced hybrid financial models 
    combining stochastic volatility with stochastic interest rates.
    
    **Available Models:**
    - **BSHW**: Black-Scholes Hull-White model
    - **Heston-HW**: Heston stochastic volatility with Hull-White rates
    - **SZHW**: Schöbel-Zhu Hull-White model  
    - **Diversification Products**: Structured products with hybrid dynamics
    """)
    
    # Quick model comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Characteristics")
        st.write("""
        - Stochastic Volatility: Heston, Schöbel-Zhu
        - Stochastic Rates: Hull-White
        - Pricing Methods: COS, Monte Carlo, Analytical
        - Applications: Exotics, Structured Products
        """)
    
    with col2:
        st.subheader("Key Features")
        st.write("""
        - Implied volatility surfaces
        - Parameter sensitivity analysis
        - Monte Carlo path generation
        - Real-time visualization
        - Model validation tools
        """)

def show_bshw_analysis_page():
    st.header("Black-Scholes Hull-White Model Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        S0 = st.number_input("Spot Price", value=100.0, min_value=1.0, max_value=1000.0, step=1.0)
        T = st.number_input("Maturity (years)", value=5.0, min_value=0.1, max_value=30.0, step=0.1)
        lambd = st.number_input("Mean Reversion (λ)", value=0.1, min_value=0.01, max_value=2.0, step=0.01)
        eta = st.number_input("Rate Vol (η)", value=0.05, min_value=0.01, max_value=1.0, step=0.01)
        sigma = st.number_input("Equity Vol (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
        rho = st.slider("Correlation (ρ)", -1.0, 1.0, 0.3, step=0.1)
        
        if st.button("Run BSHW Analysis", type="primary"):
            with st.spinner("Calculating BSHW prices and volatilities..."):
                try:
                    # Initialize BSHW model
                    bshw_model = BSHWModel()
                    
                    # Run analysis
                    results = bshw_model.run_analysis(S0, T, lambd, eta, sigma, rho)
                    
                    with col2:
                        st.subheader("BSHW Results")
                        
                        # Display key metrics
                        st.metric("BSHW Exact Volatility", f"{results['iv_exact']*100:.2f}%")
                        
                        # Create plots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Option prices plot
                        ax1.plot(results['K'], results['exact_prices'], '--r', linewidth=2)
                        ax1.grid(True, alpha=0.3)
                        ax1.set_xlabel("Strike Price")
                        ax1.set_ylabel("Option Price")
                        ax1.set_title("BSHW Option Prices")
                        ax1.legend(["Exact Solution"])
                        
                        # Implied volatility plot
                        ax2.plot(results['K'], np.ones([len(results['K']), 1]) * results['iv_exact'] * 100.0, '--r', linewidth=2)
                        ax2.grid(True, alpha=0.3)
                        ax2.set_xlabel("Strike Price")
                        ax2.set_ylabel("Implied Volatility [%]")
                        ax2.set_title("BSHW Implied Volatility")
                        ax2.set_ylim([0, 100])
                        ax2.legend(["Exact Solution"])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Additional information
                        st.info(f"""
                        **Analysis Summary:**
                        - Spot Price: ${S0:.2f}
                        - Maturity: {T} years
                        - Mean Reversion: {lambd:.2f}
                        - Rate Volatility: {eta:.3f}
                        - Equity Volatility: {sigma:.3f}
                        - Correlation: {rho:.2f}
                        """)
                        
                except Exception as e:
                    st.error(f"Error in BSHW calculation: {str(e)}")
    
    # Show placeholder when no analysis has been run
    if not st.session_state.get('bshw_analysis_run', False):
        with col2:
            st.subheader("BSHW Results")
            st.info("Click 'Run BSHW Analysis' to see the results here.")

def show_heston_hw_analysis_page():
    st.header("Heston Hull-White Model Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        
        # Basic parameters
        S0 = st.number_input("Spot Price", value=100.0, min_value=1.0, max_value=1000.0, step=1.0, key="heston_s0")
        T = st.number_input("Maturity (years)", value=15.0, min_value=0.1, max_value=30.0, step=0.1, key="heston_t")
        r = st.number_input("Risk-free Rate", value=0.1, min_value=0.0, max_value=0.5, step=0.01, key="heston_r")
        
        # Heston parameters
        st.subheader("Heston Parameters")
        v0 = st.number_input("Initial Variance", value=0.02, min_value=0.001, max_value=1.0, step=0.01)
        vbar = st.number_input("Long-run Variance", value=0.05, min_value=0.001, max_value=1.0, step=0.01)
        kappa = st.number_input("Mean Reversion", value=0.5, min_value=0.01, max_value=5.0, step=0.01)
        gamma = st.number_input("Vol of Vol", value=0.3, min_value=0.01, max_value=2.0, step=0.01)
        
        # Hull-White parameters
        st.subheader("Hull-White Parameters")
        lambd = st.number_input("Rate Mean Reversion", value=1.12, min_value=0.01, max_value=5.0, step=0.01, key="heston_lambd")
        eta = st.number_input("Rate Volatility", value=0.01, min_value=0.001, max_value=1.0, step=0.01, key="heston_eta")
        
        # Correlation parameters
        st.subheader("Correlation Parameters")
        rhoxv = st.slider("Stock-Vol Correlation", -1.0, 1.0, -0.8, step=0.1)
        rhoxr = st.slider("Stock-Rate Correlation", -1.0, 1.0, 0.5, step=0.1)
        
        # Simulation settings
        st.subheader("Simulation Settings")
        NoOfPaths = st.slider("Number of Paths", min_value=100, max_value=5000, value=1000, step=100)
        NoOfSteps = st.slider("Number of Steps", min_value=10, max_value=1000, value=100, step=10)
        
        if st.button("Run Heston-HW Analysis", type="primary"):
            with st.spinner("Running Heston-HW Monte Carlo simulation..."):
                try:
                    # Initialize Heston-HW model
                    heston_model = HestonHWModel()
                    
                    # Run analysis
                    results = heston_model.run_analysis(
                        S0=S0, T=T, r=r, v0=v0, vbar=vbar, kappa=kappa, 
                        gamma=gamma, lambd=lambd, eta=eta, rhoxv=rhoxv, 
                        rhoxr=rhoxr, NoOfPaths=NoOfPaths, NoOfSteps=NoOfSteps
                    )
                    
                    with col2:
                        st.subheader("Heston-HW Results")
                        
                        # Display martingale checks
                        col2a, col2b = st.columns(2)
                        with col2a:
                            st.metric("Euler Martingale Check", f"{results['euler_martingale']:.6f}")
                        with col2b:
                            st.metric("AES Martingale Check", f"{results['aes_martingale']:.6f}")
                        
                        # Create plots
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(results['K'], results['euler_prices'], label='Euler Method')
                        ax.plot(results['K'], results['aes_prices'], '.k', label='AES Method')
                        ax.plot(results['K'], results['cos_prices'], '--r', label='COS Method')
                        ax.set_ylim([0.0, 110.0])
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('Strike Price, K')
                        ax.set_ylabel('Option Value')
                        ax.set_title('Heston-HW Model Comparison')
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display COS method values
                        st.subheader("COS Method Values")
                        st.dataframe(results['cos_values_df'])
                        
                except Exception as e:
                    st.error(f"Error in Heston-HW calculation: {str(e)}")
    
    # Show placeholder when no analysis has been run
    if not st.session_state.get('heston_analysis_run', False):
        with col2:
            st.subheader("Heston-HW Results")
            st.info("Click 'Run Heston-HW Analysis' to see the results here.")

def show_szhw_analysis_page():
    st.header("Schöbel-Zhu Hull-White Model Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Parameters")
        
        # Basic parameters
        S0 = st.number_input("Spot Price", value=100.0, min_value=1.0, max_value=1000.0, step=1.0, key="szhw_s0")
        T = st.number_input("Maturity (years)", value=5.0, min_value=0.1, max_value=30.0, step=0.1, key="szhw_t")
        
        # Hull-White parameters
        st.subheader("Hull-White Parameters")
        lambd = st.number_input("Mean Reversion (λ)", value=0.425, min_value=0.01, max_value=2.0, step=0.01, key="szhw_lambd")
        eta = st.number_input("Rate Vol (η)", value=0.1, min_value=0.01, max_value=1.0, step=0.01, key="szhw_eta")
        
        # SZHW parameters
        st.subheader("SZHW Parameters")
        sigma0 = st.number_input("Initial Volatility", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        gamma = st.number_input("Vol of Vol", value=0.11, min_value=0.01, max_value=1.0, step=0.01)
        kappa = st.number_input("Vol Mean Reversion", value=0.4, min_value=0.01, max_value=2.0, step=0.01)
        sigmabar = st.number_input("Long-run Volatility", value=0.05, min_value=0.01, max_value=1.0, step=0.01)
        
        # Correlation parameters
        st.subheader("Correlation Parameters")
        Rxsigma = st.slider("Stock-Vol Correlation", -1.0, 1.0, -0.42, step=0.01)
        Rrsigma = st.slider("Rate-Vol Correlation", -1.0, 1.0, 0.32, step=0.01)
        Rxr = st.slider("Stock-Rate Correlation", -1.0, 1.0, 0.3, step=0.01)
        
        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        param_to_vary = st.selectbox(
            "Parameter to Analyze",
            ["gamma", "kappa", "Rxsigma", "sigmabar"]
        )
        
        if st.button("Run SZHW Analysis", type="primary"):
            with st.spinner("Calculating SZHW implied volatilities..."):
                try:
                    # Initialize SZHW model
                    szhw_model = SZHWModel()
                    
                    # Run analysis
                    results = szhw_model.run_sensitivity_analysis(
                        S0=S0, T=T, lambd=lambd, eta=eta, sigma0=sigma0,
                        gamma=gamma, kappa=kappa, sigmabar=sigmabar,
                        Rxsigma=Rxsigma, Rrsigma=Rrsigma, Rxr=Rxr,
                        param_to_vary=param_to_vary
                    )
                    
                    with col2:
                        st.subheader("SZHW Results")
                        
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        for i, (param_val, iv_data) in enumerate(results['sensitivity_data']):
                            ax.plot(results['K'], iv_data * 100.0, label=f'{param_to_vary}={param_val}')
                        
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('Strike Price, K')
                        ax.set_ylabel('Implied Volatility [%]')
                        ax.set_title(f'SZHW Implied Volatility - {param_to_vary.capitalize()} Variation')
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display parameter info
                        st.info(f"""
                        **Analysis Parameters:**
                        - Spot Price: ${S0:.2f}
                        - Maturity: {T} years
                        - Initial Volatility: {sigma0:.3f}
                        - Vol of Vol: {gamma:.3f}
                        - Analyzing: {param_to_vary} variation
                        """)
                        
                except Exception as e:
                    st.error(f"Error in SZHW calculation: {str(e)}")
    
    # Show placeholder when no analysis has been run
    if not st.session_state.get('szhw_analysis_run', False):
        with col2:
            st.subheader("SZHW Results")
            st.info("Click 'Run SZHW Analysis' to see the results here.")

def show_diversification_analysis_page():
    st.header("Diversification Product Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Product Parameters")
        
        # Basic parameters
        S0 = st.number_input("Spot Price", value=100.0, min_value=1.0, max_value=1000.0, step=1.0, key="div_s0")
        
        # Product specifications
        T = st.number_input("Product Maturity (years)", value=9.0, min_value=0.1, max_value=30.0, step=0.1)
        T1 = st.number_input("ZCB Maturity (years)", value=10.0, min_value=0.1, max_value=30.0, step=0.1)
        
        # Hull-White parameters
        st.subheader("Hull-White Parameters")
        lambd = st.number_input("Mean Reversion (λ)", value=1.12, min_value=0.01, max_value=5.0, step=0.01, key="div_lambd")
        eta = st.number_input("Rate Vol (η)", value=0.02, min_value=0.001, max_value=1.0, step=0.01, key="div_eta")
        
        # Allocation range
        st.subheader("Allocation Analysis")
        omega_min = st.number_input("Min Allocation (ω)", value=-3.0, min_value=-5.0, max_value=0.0, step=0.1)
        omega_max = st.number_input("Max Allocation (ω)", value=3.0, min_value=0.0, max_value=5.0, step=0.1)
        omega_points = st.slider("Number of Points", min_value=10, max_value=100, value=50)
        
        # Correlation scenarios
        st.subheader("Correlation Scenarios")
        scenario = st.selectbox(
            "Correlation Scenario",
            ["Low Correlation (Rxr=0.0)", "Negative Correlation (Rxr=-0.7)", "Positive Correlation (Rxr=0.7)"]
        )
        
        if st.button("Run Diversification Analysis", type="primary"):
            with st.spinner("Running diversification product analysis..."):
                try:
                    # Initialize diversification model
                    div_model = DiversificationModel()
                    
                    # Map scenario to parameters
                    scenario_params = {
                        "Low Correlation (Rxr=0.0)": {
                            "Rxr": 0.0, "sigmabar": 0.167, "gamma": 0.2, 
                            "Rxsigma": -0.850, "Rrsigma": -0.008, "sigma0": 0.035
                        },
                        "Negative Correlation (Rxr=-0.7)": {
                            "Rxr": -0.7, "sigmabar": 0.137, "gamma": 0.236,
                            "Rxsigma": -0.381, "Rrsigma": -0.339, "sigma0": 0.084
                        },
                        "Positive Correlation (Rxr=0.7)": {
                            "Rxr": 0.7, "sigmabar": 0.102, "gamma": 0.211,
                            "Rxsigma": -0.850, "Rrsigma": -0.340, "sigma0": 0.01
                        }
                    }
                    
                    # Run analysis
                    results = div_model.run_analysis(
                        S0=S0, T=T, T1=T1, lambd=lambd, eta=eta,
                        omega_min=omega_min, omega_max=omega_max, omega_points=omega_points,
                        scenario_params=scenario_params[scenario]
                    )
                    
                    with col2:
                        st.subheader("Diversification Product Results")
                        
                        # Create plots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Product prices
                        ax1.plot(results['omegaV'], results['prices'])
                        ax1.grid(True, alpha=0.3)
                        ax1.set_xlabel('Allocation Weight (ω)')
                        ax1.set_ylabel('Product Price')
                        ax1.set_title('Diversification Product Prices')
                        
                        # Relative correlation effect (if reference available)
                        if results.get('relative_prices') is not None:
                            ax2.plot(results['omegaV'], results['relative_prices'])
                            ax2.grid(True, alpha=0.3)
                            ax2.set_xlabel('Allocation Weight (ω)')
                            ax2.set_ylabel('Relative Price')
                            ax2.set_title('Relative Correlation Effect')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display scenario info
                        st.info(f"""
                        **Scenario: {scenario}**
                        - Stock Allocation Range: {omega_min} to {omega_max}
                        - Product Maturity: {T} years
                        - ZCB Maturity: {T1} years
                        - Correlation Rxr: {scenario_params[scenario]['Rxr']}
                        """)
                        
                except Exception as e:
                    st.error(f"Error in diversification analysis: {str(e)}")
    
    # Show placeholder when no analysis has been run
    if not st.session_state.get('div_analysis_run', False):
        with col2:
            st.subheader("Diversification Results")
            st.info("Click 'Run Diversification Analysis' to see the results here.")

# Initialize session state variables
if 'bshw_analysis_run' not in st.session_state:
    st.session_state.bshw_analysis_run = False
if 'heston_analysis_run' not in st.session_state:
    st.session_state.heston_analysis_run = False
if 'szhw_analysis_run' not in st.session_state:
    st.session_state.szhw_analysis_run = False
if 'div_analysis_run' not in st.session_state:
    st.session_state.div_analysis_run = False

if __name__ == "__main__":
    main()