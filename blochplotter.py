import numpy as np
import matplotlib.pyplot as plt


def plot_bloch_trajectory(x, y, z, show_points=True):
    """
    Plot a trajectory on the Bloch sphere.

    Parameters
    ----------
    x, y, z : array-like
        Expectation values of the Pauli operators at each time step.
        Must be equal-length sequences.
    show_points : bool
        If True, marks start (green) and end (red) of the trajectory.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length")

    # Create sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)

    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere surface (light transparency)
    ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.1, linewidth=0)

    # Plot axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='black', lw=1)
    ax.plot([0, 0], [-1, 1], [0, 0], color='black', lw=1)
    ax.plot([0, 0], [0, 0], [-1, 1], color='black', lw=1)

    # Plot trajectory
    ax.plot(x, y, z, color='blue', lw=2, label='Trajectory')

    # Start/end markers
    if show_points:
        ax.scatter(x[0], y[0], z[0], color='green', s=60, label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=60, label='End')

    # Aesthetics
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('⟨σx⟩')
    ax.set_ylabel('⟨σy⟩')
    ax.set_zlabel('⟨σz⟩')

    ax.set_title("Bloch Sphere Trajectory")
    ax.legend()

    # Remove grid for cleaner Bloch look
    ax.grid(False)

    plt.tight_layout()
    plt.show()
    