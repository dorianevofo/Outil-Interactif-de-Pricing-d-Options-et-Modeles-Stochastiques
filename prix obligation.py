import numpy as np

# Fonction qui calcule le facteur d'actualisation pour le modèle Hull-White
def P_HW(t, T, r0, a, sigma):
    """
    Prix d'une obligation zéro-coupon sous Hull-White.
    t : date actuelle
    T : date de maturité
    r0: taux instantané actuel
    a : taux de réversion
    sigma : volatilité du taux courte
    """
    B = (1 - np.exp(-a*(T-t)))/a
    A = np.exp((B - (T-t))*(a**2*0/(2*sigma**2)) - (sigma**2 * B**2)/(4*a))
    return A * np.exp(-B*r0)

# Fonction pour calculer le prix d'une obligation couponnée
def bond_price_hw(face_value, coupon_rate, maturity, coupon_freq, r0, a, sigma):
    """
    face_value : valeur nominale de l'obligation
    coupon_rate : taux du coupon (en décimal)
    maturity : échéance en années
    coupon_freq : fréquence des coupons par an
    r0, a, sigma : paramètres du modèle Hull-White
    """
    price = 0.0
    periods = int(maturity * coupon_freq)
    dt = 1 / coupon_freq
    
    for i in range(1, periods + 1):
        t = 0
        T = i*dt
        cf = face_value*coupon_rate*dt
        price += cf * P_HW(t, T, r0, a, sigma)
    
    # Ajouter la valeur nominale au dernier paiement
    price += face_value * P_HW(0, maturity, r0, a, sigma)
    return price

# Exemple d'utilisation
face_value = 1000        # valeur nominale
coupon_rate = 0.05       # 5% de coupon annuel
maturity = 5             # échéance 5 ans
coupon_freq = 2          # semi-annuel
r0 = 0.03                # taux initial 3%
a = 0.1                  # paramètre de réversion
sigma = 0.01             # volatilité du taux court

prix_obligation = bond_price_hw(face_value, coupon_rate, maturity, coupon_freq, r0, a, sigma)
print(f"Prix de l'obligation (Hull-White) : {prix_obligation:.2f} €")