import numpy as np
import pandas as pd # Correction : 'pandas' est plus courant sous le nom 'pd'
import matplotlib.pyplot as plt

# ... (tes fonctions mse, rmse, quadratic_regression restent les mêmes)

def backpropagation_quadratic(a, b, c, x, y, learning_rate):
    n = len(x)
    y_pred = a * x**2 + b * x + c
    error = y_pred - y
    
    # Tes formules étaient déjà excellentes !
    dL_da = 2/n * np.sum(error * x**2)
    dL_db = 2/n * np.sum(error * x)
    dL_dc = 2/n * np.sum(error)
    
    a -= learning_rate * dL_da
    b -= learning_rate * dL_db
    c -= learning_rate * dL_dc
    
    current_rmse = np.sqrt(np.mean(error**2))
    return a, b, c, current_rmse

def gradient_descent_quadratic(x, y, learning_rate= 10**(-3) , epochs=500):
    # Initialisation
    a, b, c = np.random.random (), np.random.random (), np.random.random ()
    rmse_history = []

    for i in range(epochs):
        a, b, c, current_rmse = backpropagation_quadratic(a, b, c, x, y, learning_rate)
        rmse_history.append(current_rmse)

    # Visualisation finale
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color="red", label="Données")
    
    # Tracer la courbe apprise
    x_range = np.linspace(min(x), max(x), 100)
    y_range = a * x_range**2 + b * x_range + c
    plt.plot(x_range, y_range, color="blue", label="Modèle Quadratique")
    
    plt.legend()
    plt.title(f"Résultat après {epochs} époques")
    plt.show()
    
    return a, b, c, rmse_history

if __name__ == "__main__":
    # Chargement
    df = pd.read_csv("prix_maisons.csv")
    
    # Standardisation
    x = (df["surface"] - df["surface"].mean()) / df["surface"].std()
    y = (df["prix"] - df["prix"].mean()) / df["prix"].std()
    
    # LANCEMENT DE L'ENTRAÎNEMENT (ce qui manquait)
    a_fin, b_fin, c_fin, history = gradient_descent_quadratic(x.values, y.values)
    
    print(f"Coefficients finaux : a={a_fin:.4f}, b={b_fin:.4f}, c={c_fin:.4f}")
    print(f"RMSE finale : {history[-1]:.4f}")