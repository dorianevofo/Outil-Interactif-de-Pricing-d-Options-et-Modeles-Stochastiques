import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline

# 1. Black-Scholes Call
def bs_call(S0, K, T, r, sigma):
    if T <= 0:
        return max(S0 - K, 0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 2. Surface de volatilité implicite
K_grid = np.linspace(80, 120, 9)
T_grid = np.linspace(0.1, 2, 9)
sigma_imp = 0.2 + 0.1 * np.outer((K_grid - 100)**2 / 10000, T_grid)
interp_sigma = RectBivariateSpline(K_grid, T_grid, sigma_imp)

# 3. Volatilité locale (Dupire)
def local_volatility(S0, r, interp_sigma, K, T):
    sigma = interp_sigma(K, T)[0][0]
    C = bs_call(S0, K, T, r, sigma)
    dK = 1e-2 * K
    dT = 1e-4

    C_K_plus  = bs_call(S0, K + dK, T, r, interp_sigma(K + dK, T)[0][0])
    C_K_minus = bs_call(S0, K - dK, T, r, interp_sigma(K - dK, T)[0][0])
    C_T_plus  = bs_call(S0, K, T + dT, r, interp_sigma(K, T + dT)[0][0])

    dC_dK  = (C_K_plus - C_K_minus) / (2 * dK)
    d2C_dK2 = (C_K_plus - 2 * C + C_K_minus) / (dK**2)
    dC_dT = (C_T_plus - C) / dT

    sigma_loc_sq = (dC_dT + r * K * dC_dK) / (0.5 * K**2 * d2C_dK2)
    return np.sqrt(abs(sigma_loc_sq))

# 4. Simulation Monte Carlo sous modèle LV
def monte_carlo_lv(S0, K, r, T, n_paths=1000, n_steps=50):
    dt = T / n_steps
    S = np.full(n_paths, S0)

    for i in range(n_steps):
        t = (i+1) * dt
        # Calcul de la vol locale point par point
        sigma_loc = np.array([local_volatility(S0, r, interp_sigma, s, t) for s in S])
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        S = S * np.exp((r - 0.5 * sigma_loc**2) * dt + sigma_loc * dW)

    payoff = np.maximum(S - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# 5. Exemple d’utilisation
S0, K, r, T = 200, 100, 0.02, 1.0
prix_call = monte_carlo_lv(S0, K, r, T)
sigma_loc = local_volatility(S0, r, interp_sigma, K, T)
print(f"\n Volatilité locale en K={K}, T={T} : {sigma_loc:.4f}")
print("----------------------------------------------------------------")
print(f"Prix du Call (modèle de volatilité locale) = {prix_call:.4f}\n")


