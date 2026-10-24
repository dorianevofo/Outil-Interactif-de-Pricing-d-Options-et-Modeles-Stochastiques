import numpy as np
from scipy.stats import norm

# ------------------------
# Paramètres du modèle
# ------------------------
r0 = 0.03          # Taux d'intérêt initial
a = 0.1            # Vitesse de réversion
sigma = 0.01       # Volatilité
T_option = 1.0     # Maturité option (en années)
T_bond = 2.0       # Maturité obligation (en années)
K = 0.95           # Prix d'exercice de l'option (en % de nominal)
nominal = 1000     # Valeur nominale de l'obligation en euros

# ------------------------
# Prix de l'obligation zéro-coupon sous Hull-White
# ------------------------
def P_HW(r, t, T, a, sigma):
    """Calcule le prix de l'obligation zéro-coupon P(t,T) selon Hull-White."""
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp((B - (T - t)) * (a**2 * 0.5 * sigma**2) - sigma**2 / (4 * a) * B**2)
    return A * np.exp(-B * r)

# ------------------------
# Prix d'une option de call sur l'obligation
# ------------------------
def call_option_price_HW(r0, a, sigma, T_option, T_bond, K, nominal=1.0):
    """
    Prix d'un call sur obligation zéro-coupon avec formule analytique 
    de Jamshidian dans Hull-White (simplifié pour zéro-coupon bond)
    """
    # Prix de l'obligation sous-jacente à t=0
    P0T = P_HW(r0, 0, T_bond, a, sigma)
    
    # Écart type du prix de l'obligation
    tau = T_bond - T_option
    sigma_P = sigma / a * (1 - np.exp(-a * tau)) * np.sqrt((1 - np.exp(-2 * a * T_option)) / (2 * a))
    
    # Prix de l'option (formule de Black sur le prix de l'obligation)
    d1 = (np.log(P0T / K) + 0.5 * sigma_P**2) / sigma_P
    d2 = d1 - sigma_P
    call_price_unit = P0T * norm.cdf(d1) - K * norm.cdf(d2)
    
    # Conversion en euros
    call_price_euro = call_price_unit * nominal
    return call_price_euro

# ------------------------
# Calcul et exemple d'utilisation
# ------------------------
call_price_euro = call_option_price_HW(r0, a, sigma, T_option, T_bond, K, nominal)
print(f"Prix de l'option de call sur obligation: {call_price_euro:.2f} €")