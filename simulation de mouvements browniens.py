import numpy as np
import matplotlib.pyplot as plt

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

# Paramètres
n_steps = 1000  # Nombre de pas
T = 1.0  # Temps total (1 unité de temps)

# Simulation et calcul
brownian_path, dt = simulate_brownian_motion(n_steps, T)
qv = quadratic_variation(brownian_path)

brownian_path2, dt2=simulate_brownian_motion(n_steps, T)
qv2 = quadratic_variation(brownian_path2)
# Affichage des résultats
print(f"Variation quadratique calculée : {qv:.4f}")
print(f"Théorique (T pour un mouvement brownien standard) : {T:.4f}")
print("-------------------------------------------------------------")
print(f"Variation quadratique calculée : {qv2:.4f}")
print(f"Théorique (T pour un mouvement brownien standard) : {T:.4f}")
print("-------------------------------------------------------------")
brownian_path3, dt3=simulate_brownian_motion(n_steps, T)
qv3 = quadratic_variation(brownian_path3)
# Affichage des résultats
print(f"Variation quadratique calculée : {qv3:.4f}")
print(f"Théorique (T pour un mouvement brownien standard) : {T:.4f}")
# Visualisation
time = np.linspace(0, T, n_steps)

plt.plot(time, brownian_path,color="blue", label="Mouvement Brownien 1")
plt.plot(time, brownian_path2,color="red", label="Mouvement Brownien 2")
plt.plot(time, brownian_path3,color="orange", label="Mouvement Brownien 3")
plt.title("Mouvement Brownien")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.legend()
plt.grid()
plt.show()