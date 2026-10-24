import numpy as np
import matplotlib.pyplot as plt

def simulate_merton(S0=100, mu=0.06, sigma=0.25, 
                    lam=0.8, muJ=-0.05, sigmaJ=0.20,
                    T=1.0, N=252, n_paths=50, seed=42):
    np.random.seed(seed)
    dt = T / N
    times = np.linspace(0, T, N+1)
    kappa = np.exp(muJ + 0.5*sigmaJ**2) - 1.0
    drift = (mu - lam*kappa - 0.5*sigma**2) * dt
    S = np.zeros((n_paths, N+1))
    S[:,0] = S0
    lnS = np.log(S0) * np.ones(n_paths)
    for i in range(N):
        Z = np.random.normal(size=n_paths)
        # nombre de sauts par path dans dt (vecteur)
        K = np.random.poisson(lam*dt, size=n_paths)
        # contribution totale des sauts en log for each path
        jump_contrib = np.zeros(n_paths)
        nonzero = (K > 0)
        if nonzero.any():
            k_nz = K[nonzero]
            # somme de k normales => Normal(k*muJ, k*sigmaJ^2)
            ZJ = np.random.normal(size=k_nz.size)
            jump_contrib[nonzero] = k_nz*muJ + np.sqrt(k_nz)*sigmaJ*ZJ
        lnS = lnS + drift + sigma*np.sqrt(dt)*Z + jump_contrib
        S[:, i+1] = np.exp(lnS)
    return times, S

# Exemple d'utilisation
times, S = simulate_merton(n_paths=10)
plt.figure(figsize=(10,5))
for i in range(S.shape[0]):
    plt.plot(times, S[i], lw=0.9)
plt.title("Simulations Merton jump-diffusion (quelques trajectoires)")
plt.xlabel("Temps (années)")
plt.ylabel("Prix")
plt.grid(True)
plt.show()

def price_european_call_mc(S0=100, K=100, r=0.03, sigma=0.25,
                           lam=0.8, muJ=-0.05, sigmaJ=0.20,
                           T=1.0, N=252, n_paths=20000, seed=1):
    # Simuler sous Q (mu = r)
    times, S = simulate_merton(S0=S0, mu=r, sigma=sigma,
                               lam=lam, muJ=muJ, sigmaJ=sigmaJ,
                               T=T, N=N, n_paths=n_paths, seed=seed)
    ST = S[:, -1]
    payoffs = np.maximum(ST - K, 0.0)
    price = np.exp(-r*T) * payoffs.mean()
    stderr = np.exp(-r*T) * payoffs.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr

price, se = price_european_call_mc(n_paths=20000)
print(f"Call MC price ≈ {price:.4f} ± {1.96*se:.4f} (95% CI)")
