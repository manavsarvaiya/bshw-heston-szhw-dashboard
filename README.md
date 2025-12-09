# Bshw-Heston-Szhw-Dashboard

## Overview

Interactive Streamlit dashboard for analyzing advanced hybrid financial models that combine stochastic volatility with stochastic interest rates. The app provides real-time pricing, risk analysis, and visualization tools built around four core model families implemented in `models.py.`

--------------------------------------------------------------

## Features

### Model Coverage

- BSHW Model: Black–Scholes with Hull–White stochastic rates
- Heston–HW Model: Heston stochastic volatility with Hull–White rates
- SZHW Model: Schöbel–Zhu stochastic volatility with Hull–White rates
- Diversification Products: Stylized structured products with hybrid dynamics and correlation scenarios

### Technical Capabilities

- Real-time parameter controls with immediate visualization (via Streamlit UI)
- Monte Carlo path simulation with CIR-based variance dynamics
- Implied volatility–style sensitivity curves for SZHW
- Parameter sensitivity analysis across key volatility and correlation parameters
- Martingale checks for model validation (Heston–HW)
- Diversification payoff and relative price profiles under different correlations

## Installation

### Requirements

Install dependencies with:

````
pip install streamlit numpy matplotlib scipy pandas
````

### Launch Application

Install dependencies with:

```
streamlit run app.py
```

--------------------------------------------------------------

## Project Structure

### Core Files

```
Project/
|- app.py
|- models.py
|- README.me
```

### Model Classes

```
# models.py contains:
class BaseModel()             # Shared BS + implied vol utilities
class BSHWModel(BaseModel)    # Black–Scholes Hull–White
class HestonHWModel(BaseModel)# Heston with Hull–White rates
class SZHWModel(BaseModel)    # Schöbel–Zhu Hull–White
class DiversificationModel(BaseModel)  # Stylized structured products
```

--------------------------------------------------------------

## Model Specifications

### BaseModel Utilities

- Black–Scholes Pricing

```
BS_Call_Option_Price(CP, S0, K, σ, τ, r)
```
Implements standard BS call/put pricing with careful handling of input shapes.

- Implied Volatility (Black-76 Style)
```
ImpliedVolatilityBlack76(CP, marketPrice, K, T, S0)
```
Uses Newton–Raphson with robust fallback to bisection search and bounded volatility.

--------------------------------------------------------------

## BSHW Model Dynamics

### Effective stock dynamics under stochastic short rate:
```
dS(t) = r(t) S(t) dt + σ S(t) dW₁(t)
```
### Hull–White short-rate process:
```
dr(t) = λ (θ(t) - r(t)) dt + η dW₂(t)
⟨dW₁, dW₂⟩ = ρ dt
```
### The implementation:
- Computes an effective BSHW volatility via time integration of a forward volatility term.
- Prices options using Black–Scholes on the forward S0/P0T with this composite volatility.

### Key methods:
- `BSHWVolatility(T, eta, sigma, rho, lambd)`
- `BSHWOptionPrice(CP, S0, K, P0T, T, eta, sigma, rho, lambd)`
- `run_analysis(S0, T, lambd, eta, sigma, rho)`
  
--------------------------------------------------------------

## Heston–HW Framework

### Heston stochastic volatility with Hull–White rates:

```
dS(t) = r(t) S(t) dt + √v(t) S(t) dW₁(t)
dv(t) = κ (v̄ - v(t)) dt + γ √v(t) dW₂(t)
dr(t) = λ (θ(t) - r(t)) dt + η dW₃(t)
```
### Correlations:

```
⟨dW₁, dW₂⟩ = ρₓᵥ dt   (stock–vol correlation)
⟨dW₁, dW₃⟩ = ρₓʳ dt   (stock–rate correlation)
```
## Implementation highlights:

- CIR_Sample: Exact (noncentral chi-square) sampling for CIR variance, with Euler fallback.
- GeneratePathsHestonHW_AES: "Almost exact" path simulation for (S, v, r) and money-market account $\( M_t = e^{\int_0^t r(s)\,ds} \)$
- EUOptionPriceFromMCPathsGeneralizedStochIR: Monte Carlo pricing using path-dependent discounting via $M_T$
- run_analysis: Orchestrates path generation, pricing across strikes, and martingale checks.

--------------------------------------------------------------

## SZHW Model (Schöbel–Zhu with HW Rates)

### Schöbel–Zhu volatility plus Hull–White rates (conceptual form):

```
dS(t)   = r(t) S(t) dt + σ(t) S(t) dW₁(t)
dσ(t)   = κ (σ̄ - σ(t)) dt + γ dW₂(t)
dr(t)   = λ (θ(t) - r(t)) dt + η dW₃(t)
```
### The current implementation focuses on parameter sensitivity of implied volatility curves, using stylized (closed-form-like) IV patterns to highlight qualitative effects:

- `run_sensitivity_analysis(...)` returns IV curves across strikes for:
  - γ (vol-of-vol)
  - κ (volatility mean reversion)
  - Rxsigma (stock–volatility correlation)
  - sigmabar (long-run volatility level)

The Streamlit page plots these curves for visual comparison.

--------------------------------------------------------------

## Diversification Product Model

### Stylized structured product model that illustrates diversification benefits under different stock–rate correlations:

- Allocation parameter ω (e.g., equity vs fixed-income weight).
- Correlation scenario:
  - Rxr = 0.0: Low correlation baseline
  - Rxr < 0: Diversification benefit (risks offset)
  - Rxr > 0: Less diversification (risks reinforce)

### The implementation:

- `run_analysis(S0, T, T1, lambd, eta, omega_min, omega_max, omega_points, scenario_params)`

### Returns:

- `omegaV` : grid of allocation weights
- `prices` : product price profile vs ω
- `relative_prices` : price relative to baseline (Rxr = 0), when applicable

--------------------------------------------------------------

## Usage Guide

### Step-by-Step

### 1. Install Dependencies
  
  Run `pip install streamlit numpy matplotlib scipy pandas.`

### 2. Launch Dashboard

```
streamlit run app.py
```

### 3. Select Model

Use the sidebar navigation:
- Home
- BSHW Model
- Heston–HW Model
- SZHW Model
- Diversification Products

### 4. Adjust Parameters

Use sliders and numeric inputs for:
- Spot price, maturities
- Volatility parameters (σ, v₀, v̄, γ, κ)
- Rate parameters (λ, η)
- Correlation parameters (ρ, ρₓᵥ, ρₓʳ)
- Monte Carlo settings (number of paths/steps)
- Allocation ranges for diversification products

### 5. Run Analysis

Click the corresponding Run Analysis button for each model page.

### 6. View Results

Inspect:
- Option price vs strike plots
- Effective volatility / IV curves
- Martingale metrics
- Sensitivity charts and diversification profiles

--------------------------------------------------------------

## Application Scenarios

### Use Cases

- Long-dated option pricing under stochastic interest rates
- Hybrid derivative and structured product prototyping
- Studying volatility smiles/skews in hybrid frameworks
- Testing diversification strategies under different correlation regimes
- Educational demonstrations of stochastic volatility + stochastic rates

### Target Users

- Quantitative Analysts & Quants
- Risk Managers
- Financial Engineers / Structurers
- Researchers & Graduate Students in Quant Finance

--------------------------------------------------------------

## Technical Implementation

### Numerical Methods

### - Black–Scholes Analytics
  Closed-form European option pricing and implied volatility inversion.

### - CIR Exact Sampling
  Use of noncentral chi-square distribution for variance paths, with safe fallbacks.

### - Monte Carlo Simulation
  - Heston–HW: joint simulation of (S, v, r, $M_t$)
  - Path-wise discounting using $M_T$
  - Strike-wise payoff aggregation for calls/puts

### - Effective Volatility (BSHW)
  Time integration of forward volatility to obtain a composite volatility per maturity.

### - Sensitivity & Stylized IV Curves (SZHW)
  Synthetic IV “smiles” to visualize parametric effects (γ, κ, ρ, σ̄).

### Key Functions

```
# Base analytics
BS_Call_Option_Price()
ImpliedVolatilityBlack76()

# BSHW
BSHWVolatility()
BSHWOptionPrice()
run_analysis()  # per model

# Heston–HW
CIR_Sample()
GeneratePathsHestonHW_AES()
EUOptionPriceFromMCPathsGeneralizedStochIR()
run_analysis()

# SZHW
run_sensitivity_analysis()

# Diversification
run_analysis()
```

--------------------------------------------------------------

## Performance & Validation

### Model Validation

#### - Martingale Check (Heston–HW):
 $ \( \mathbb{E}[S_T / M_T] \approx S_0 \)$ for risk-neutral correctness.

#### - Basic error handling with fallbacks:
  - CIR sampling fallback to Euler
  - BSHW volatility fallback approximation
  - Safe clipping of volatilities and variances

#### - Computational Efficiency
  - Vectorized NumPy operations for path and payoff calculations
  - Adjustable path and step counts for speed/accuracy tradeoffs
  - Lightweight visualizations with Matplotlib embedded in Streamlit

--------------------------------------------------------------























































