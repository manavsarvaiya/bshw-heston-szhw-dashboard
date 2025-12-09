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

## Model Specifications

### BaseModel Utilities

































































