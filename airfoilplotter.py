import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,1,500)

y =  0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 
                        0.3516 * x**2 + 0.2843 * x**3 - 
                        0.1015 * x**4)

x_cor = np.concatenate((x, np.flip(x)))
y_cor = np.concatenate((y, np.flip(-y)))

def plot_airfoil(angle_of_attack, scalefactor, xloc, yloc, ax=None):

    x_qc_target, y_qc_target = xloc, yloc

    # Quarter chord as rotation center
    x_qc_local = 0.25 * scalefactor
    y_qc_local = 0.0

    alpha_rad = -np.radians(angle_of_attack)

    # Scale
    x_scaled = x_cor * scalefactor
    y_scaled = y_cor * scalefactor

    # Rotate about quarter chord
    x_rot = np.cos(alpha_rad) * (x_scaled - x_qc_local) - np.sin(alpha_rad) * (y_scaled - y_qc_local)
    y_rot = np.sin(alpha_rad) * (x_scaled - x_qc_local) + np.cos(alpha_rad) * (y_scaled - y_qc_local)

    x_final = x_rot + x_qc_target
    y_final = y_rot + y_qc_target

    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False

    ax.fill(x_final, y_final, color='silver', alpha=1)
    ax.plot(x_final, y_final, color='black', linewidth=0.5)

    if standalone:
        ax.set_aspect('equal')


if __name__ == "__main__":
    angle_of_attack = 0  # Example angle of attack in degrees
    plot_airfoil(0, 100, 25, 0, ax=None)
    plt.show()


