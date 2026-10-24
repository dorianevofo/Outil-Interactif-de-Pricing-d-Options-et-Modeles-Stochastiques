import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
st.title("GROUPE 8 CALCUL STOCHASTIQUE")




def page_accueil():
    st.subheader("PAGE D'ACCEUIL")
    col1,col2,col3,col4,col5=st.columns([1,10,1,1,1])
    with col2:
        if st.button("MODÈLE DE VOLATILITÉ LOCALE",use_container_width=True, key="page1"):
            st.session_state.page="page1"
        if st.button("VARIATION QUADRATIQUE",use_container_width=True, key="page2"):
            st.session_state.page="page2"
        if st.button("MODÈLE DE DIFFUSION PAR SAUT",use_container_width=True, key="page3"):
            st.session_state.page="page3"
        if st.button("MODÈLE DE HULL-WHITE",use_container_width=True, key="page4"):
            st.session_state.page="page4"
def page1():
    st.subheader("MODÈLE DE VOLATILITÉ LOCALE")
    st.write("entrez les paramètres:")
    S0=st.number_input("entrez le prix initial S_0:",min_value=0.0, icon='\U0001F4B8',step=0.1)
    K=st.number_input("entrer le prix d'exercice (strike) K:",min_value=0.0, icon='\U0001F4B0',step=0.1)
    r=st.number_input("entrer le taux sans risque r: ",min_value=0.0,max_value=100.0,icon='\U0001F4C8',step=0.01)
    T=st.number_input("entrer la durée de l'échéance(maturité) T: ",min_value=0.0,icon='\U000023F3',step=0.1)
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
    def monte_carlo_lv(S0, K, r, T, n_paths=100, n_steps=50):
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
    if st.button("calculer"):
        try:
            prix_call = monte_carlo_lv(S0, K, r, T)
            sigma_loc = local_volatility(S0, r, interp_sigma, K, T)
            st.text_area("\n La valeur de la volatilité locale: ", sigma_loc,disabled=True)
            st.text_area("\n Le prix du Call est de ", prix_call,disabled=True)
        except:
            st.write("vos paramètres ne sont pas valides!!")

    if st.button("retour a l'accueil"):
        st.session_state.page="accueil"

def page2():
    st.subheader("VARIATION QUADRATIQUE")
    st.write("simulation de processus brownien")
    st.write("entrez les paramètres:")
    n_steps=st.number_input("entrez le nombre de pas n_steps:",min_value=100, icon='\U0001F463',step=10)
    T=st.number_input("entrer le temps total:",min_value=0.0, icon='\U000023F0',step=0.1)
    # Simulation d'un mouvement brownien
    def simulate_brownian_motion(n_steps, T):
        dt = T / n_steps  # Pas de temps
        increments = np.random.normal(0, np.sqrt(dt), n_steps)  # Incréments normaux
        brownian_path = np.cumsum(increments)  # Mouvement brownien
        return brownian_path, dt

    # Calcul de la variation quadratique
    def quadratic_variation(path):
        increments = np.diff(path)  # Différences successives
        qv = np.sum(increments**2)  # Somme des carrés des incréments
        return qv

    # Simulation et calcul
    
    if st.button("simuler"):
        try:
            brownian_path, dt = simulate_brownian_motion(n_steps, T)
            qv = quadratic_variation(brownian_path)

            brownian_path2, dt2=simulate_brownian_motion(n_steps, T)
            qv2 = quadratic_variation(brownian_path2)

            brownian_path3, dt3=simulate_brownian_motion(n_steps, T)
            qv3 = quadratic_variation(brownian_path3)

            st.write(f"la valeur de la variation quadratique théorique est de : {T:.4f}")
            st.write("-------------------------------------------------------------------------------------------")
            st.write(f"la valeur de la variation quadratique calculée de la première trajectoire est de : {qv:.4f}  ≈ {T:.4f}")
            st.write(f"la valeur de la variation quadratique calculée de la deuxième trajectoire est de : {qv2:.4f} ≈ {T:.4f}")
            st.write(f"la valeur de la variation quadratique calculée de la troisième trajectoire est de : {qv3:.4f} ≈ {T:.4f}")
            
            # Visualisation


        except:
            st.write("vos paramètres ne sont pas valides!!")
        T = 1.0
        n_sim = 50000
        n_steps_list = [2**k for k in range(4, 13)]

        vq_means = []

        for n in n_steps_list:
            dt = T / n
            vq = np.sum(np.random.randn(n_sim, n)**2 * dt, axis=1)
            vq_means.append(vq.mean())

        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(n_steps_list,vq_means,linestyle="-.",color="b", linewidth=1.2)
        ax.axhline(y=T,color='r',linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlabel('Nombre de pas (échelle log)')
        ax.set_ylabel('Variation quadratique moyenne')
        ax.set_title("Convergence de [B,B]_T vers T")
        
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    if st.button("retour a l'accueil"):
        st.session_state.page="accueil"



def page3():
    st.subheader("MODÈLE DE DIFFUSION PAR SAUT")
    st.write("entrez les paramètres:")
    S0=st.number_input("entrez le prix initial S_0:",min_value=0.0, icon='\U0001F4B8',step=0.1)
    K=st.number_input("entrer le prix d'exercice (strike) K:",min_value=0.0, icon='\U0001F4B0',step=0.1)
    r=st.number_input("entrer le taux sans risque r: ",min_value=0.0,max_value=100.0,icon='\U0001F4C8',step=0.01)
    sig=st.number_input("Volatilité de la composante continue σ:",min_value=0.0,max_value=100.0,icon='\U0001F4CA',step=0.01)
    lamb=st.number_input(" Intensité des sauts λ",min_value=0.0,max_value=1.0,icon='\U0001F4DD')
    sigma=st.number_input(" Ecart-type des sauts σJ",min_value=0.0,max_value=1.0,icon='\U0001F4CA')
    T_f=st.number_input("entrer la durée de l'échéance(maturité) T: ",min_value=0.0,icon='\U000023F3',step=0.1)
    muJ=st.number_input("entrer la Moyenne logarithmique des sauts µJ",icon='\U0001F4C9',step=0.01)
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
    
    if st.button("simuler"):
        times, S = simulate_merton(S0=S0, mu=0.06, sigma=sig, 
                        lam=lamb, muJ=muJ, sigmaJ=sigma,
                        T=T_f,n_paths=10)
        
        fig,ax=plt.subplots(figsize=(6,4))
        for i in range(S.shape[0]):
            ax.plot(times,S[i],lw=0.9)
        ax.set_title("Simulations Merton jump-diffusion (quelques trajectoires)")
        ax.set_xlabel("temps")
        ax.set_ylabel("prix")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig) 
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

        price, se = price_european_call_mc(S0, K, r, sig,
                                lamb, muJ, sigma,
                                T_f,n_paths=20000)
        st.text_area("\n La valeur du prix le l'option de Call: ", f"{price:.4f} ± {1.96*se:.4f}",disabled=True)
    if st.button("retour a l'accueil"):
        st.session_state.page="accueil"


def page4():
    st.subheader("MODÈLE DE HULL-WHITE")
    coll1,coll2=st.columns([1,10])
    with coll2:
        if st.button("tarification d'obligation",use_container_width=True,key="page6"):
            st.session_state.page="page6"
        if st.button("tarification d'option d'obligation",use_container_width=True,key="page7"):
            st.session_state.page="page7"
    if st.button("retour a l'accueil"):
        st.session_state.page="accueil"


def page6():
    def P_HW(t, T, r0, a, sigma):
        B = (1 - np.exp(-a*(T-t)))/a
        A = np.exp((B - (T-t))*(a**2*0/(2*sigma**2)) - (sigma**2 * B**2)/(4*a))
        return A * np.exp(-B*r0)

    # Fonction pour calculer le prix d'une obligation couponnée
    def bond_price_hw(face_value, coupon_rate, maturity, coupon_freq, r0, a, sigma):
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
    st.subheader("tarification d'obligation ")
    st.write("entrez les paramètres:")
    S0=st.number_input("entrez la valeur nominale S_0:",min_value=0.0, icon='\U0001F4B8',step=0.1)
    c=st.number_input("entrez le coupon annuel c",min_value=0.0,max_value=100.0,icon='\U0001F4CA',step=0.01)
    T_f=st.number_input("entrer la durée de l'échéance(maturité) T: ",min_value=0.0,icon='\U000023F3',step=0.1)
    c_f=st.number_input(" enterz la frequence du coupon",min_value=0,icon='\U0001F4DD')
    r=st.number_input("entrer le taux r: ",min_value=0.0,max_value=100.0,icon='\U0001F4C8',step=0.01)
    sigm=st.number_input("volatilité du taux court σ",min_value=0.0,max_value=1.0,icon='\U0001F4CA')
    if st.button("calculer"):
        prix_obligation=bond_price_hw(S0,c,T_f,c_f,r,a=0.1,sigma=sigm)
        st.text_area("\n La valeur du prix le l'obligation est : ", f"{prix_obligation:.4f} €",disabled=True)

    if st.button("retour a la page precedente"):
        st.session_state.page="p_precedente"

def page7():
    st.subheader("tarification d'option d'obligation")

    def P_HW(r, t, T, a, sigma):
        """Calcule le prix de l'obligation zéro-coupon P(t,T) selon Hull-White."""
        B = (1 - np.exp(-a * (T - t))) / a
        A = np.exp((B - (T - t)) * (a**2 * 0.5 * sigma**2) - sigma**2 / (4 * a) * B**2)
        return A * np.exp(-B * r)
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

    st.write("entrez les paramètres:")
    r=st.number_input("entrer le taux d'intérêt initial r: ",min_value=0.0,max_value=100.0,icon='\U0001F4C8',step=0.01)
    T_op=st.number_input("entrer la maturité de l'option T_option: ",min_value=0.0,icon='\U000023F3',step=0.1)
    T_ob=st.number_input("entrer la maturité de l'obligation T_obligation: ",min_value=0.0,icon='\U000023F3',step=0.1)
    K=st.number_input("entrer le prix d'exercice de l'option(strike) K:",min_value=0.0, icon='\U0001F4B0',step=0.1)
    N0=st.number_input("entrez la valeur nominale N_0:",min_value=0.0, icon='\U0001F4B8',step=0.1)
    sigm=st.number_input("volatilité  σ",min_value=0.0,max_value=1.0,icon='\U0001F4CA')

    if st.button("calculer"):
        call_price_euro = call_option_price_HW(r, a=0.1, sigma=sigm, T_option=T_op, T_bond=T_ob, K=K, nominal=N0)
        st.text_area("\n La valeur du prix le l'option est : ", f"{call_price_euro:.4f} €",disabled=True)

    if st.button("retour a la page precedente"):
        st.session_state.page="p_precedente"


if 'page' not in st.session_state:
    st.session_state.page="accueil"
if st.session_state.page=="accueil":
    page_accueil()
elif st.session_state.page=="page1":
    page1()
elif st.session_state.page=="page2":
    page2()
elif st.session_state.page=="page3":
    page3()
elif st.session_state.page=="page4":
    page4()
elif st.session_state.page=="page6":
    page6()
elif st.session_state.page=="page7":
    page7()
elif st.session_state.page=="p_precedente":
    page4()
