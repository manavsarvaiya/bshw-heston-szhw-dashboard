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
- GeneratePathsHestonHW_AES: "Almost exact" path simulation for (S, v, r) and money-market account \( M_t = e^{\int_0^t r(s)\,ds} \).


EUOptionPriceFromMCPathsGeneralizedStochIR: Monte Carlo pricing using path-dependent discounting via 
𝑀
𝑇
M
T
	​

.

run_analysis: Orchestrates path generation, pricing across strikes, and martingale checks.




























































