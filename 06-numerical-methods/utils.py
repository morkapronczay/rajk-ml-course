import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


dlblue = "#0096ff"
dlorange = "#FF9300"
dldarkred = "#C00000"
dlmagenta = "#FF40FF"
dlpurple = "#7030A0"
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
n_bin = 5
dlcm = LinearSegmentedColormap.from_list("dl_map", dlcolors, N=n_bin)


# Loop version of multi-variable compute_cost
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,)=scalar
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def inbounds(a, b, xlim, ylim):
    xlow, xhigh = xlim
    ylow, yhigh = ylim
    ax, ay = a
    bx, by = b
    if (
        (ax > xlow and ax < xhigh)
        and (bx > xlow and bx < xhigh)
        and (ay > ylow and ay < yhigh)
        and (by > ylow and by < yhigh)
    ):
        return True
    return False


def plt_contour_wgrad(
    x,
    y,
    hist,
    ax,
    w_range=[-100, 500, 5],
    b_range=[-500, 500, 5],
    contours=[0.1, 50, 1000, 5000, 10000, 25000, 50000],
    resolution=5,
    w_final=200,
    b_final=100,
    step=10,
):
    b0, w0 = np.meshgrid(np.arange(*b_range), np.arange(*w_range))
    z = np.zeros_like(b0)
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            z[i][j] = compute_cost(x, y, w0[i][j], b0[i][j])

    CS = ax.contour(
        w0,
        b0,
        z,
        contours,
        linewidths=2,
        colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple],
    )
    ax.clabel(CS, inline=1, fmt="%1.0f", fontsize=10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Contour plot of cost J(w,b), vs b,w with path of gradient descent")
    w = w_final
    b = b_final
    ax.hlines(b, ax.get_xlim()[0], w, lw=2, color=dlpurple, ls="dotted")
    ax.vlines(w, ax.get_ylim()[0], b, lw=2, color=dlpurple, ls="dotted")

    base = hist[0]
    for point in hist[0::step]:
        edist = np.sqrt((base[0] - point[0]) ** 2 + (base[1] - point[1]) ** 2)
        if edist > resolution or point == hist[-1]:
            if inbounds(point, base, ax.get_xlim(), ax.get_ylim()):
                plt.annotate(
                    "",
                    xy=point,
                    xytext=base,
                    xycoords="data",
                    arrowprops={"arrowstyle": "->", "color": "r", "lw": 3},
                    va="center",
                    ha="center",
                )
            base = point
    return


def plt_divergence(p_hist, J_hist, x_train, y_train):
    """
    Plot cost surface with respect to w,b and the path of gradient descent
    """
    # Initialize paths
    x = np.zeros(len(p_hist))
    y = np.zeros(len(p_hist))
    v = np.zeros(len(p_hist))
    for i in range(len(p_hist)):
        x[i] = p_hist[i][0]
        y[i] = p_hist[i][1]
        v[i] = min(J_hist[i], 1e10)  # Clip extremely large values

    fig = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.5)  # Increase spacing between subplots
    gs = fig.add_gridspec(1, 5)
    fig.suptitle("Cost escalates when learning rate is too large")

    # ===============
    #  First subplot
    # ===============
    ax = fig.add_subplot(gs[:2])

    # Reduce range and increase step size
    w_array = np.arange(-1000, 1000, 20)  # Reduced range
    cost = np.zeros_like(w_array)
    fix_b = 100

    # Compute cost with safety checks
    for i in range(len(w_array)):
        try:
            cost[i] = min(compute_cost(x_train, y_train, w_array[i], fix_b), 1e10)
        except OverflowError:
            cost[i] = 1e10

    # Plot first subplot
    ax.plot(w_array, cost)
    ax.plot(x, v, c=dlmagenta)
    ax.set_title("Cost vs w, b set to 100")
    ax.set_ylabel("Cost")
    ax.set_xlabel("w")
    ax.set_ylim([0, min(max(cost), 1e5)])  # Limit y-axis range
    ax.xaxis.set_major_locator(MaxNLocator(2))

    # ===============
    # Second Subplot
    # ===============
    # Reduce mesh grid size and range
    tmp_b, tmp_w = np.meshgrid(
        np.arange(-1000, 1000, 50),  # Reduced range
        np.arange(-1000, 1000, 50),  # Reduced range
    )
    z = np.zeros_like(tmp_b)

    # Compute surface with safety checks
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            try:
                z[i][j] = min(
                    compute_cost(x_train, y_train, tmp_w[i][j], tmp_b[i][j]), 1e10
                )
            except OverflowError:
                z[i][j] = 1e10

    # Create 3D subplot
    ax = fig.add_subplot(gs[2:], projection="3d")
    ax.plot_surface(tmp_w, tmp_b, z, alpha=0.3, color=dlblue)
    ax.xaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(2))

    # Set labels and customize view
    ax.set_xlabel("w", fontsize=16)
    ax.set_ylabel("b", fontsize=16)
    ax.set_zlabel("\ncost", fontsize=16)
    ax.set_title("Cost vs (b, w)")
    ax.view_init(elev=20.0, azim=-65)

    # Plot the path
    ax.plot(x, y, v, c=dlmagenta)

    plt.tight_layout()
    return


# draw derivative line
# y = m*(x - x1) + y1
def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1 - d, x1 + d, 50)
    y = dj_dx * (x - x1) + y1
    ax.scatter(x1, y1, color=dlblue, s=50)
    ax.plot(x, y, "--", c=dldarkred, zorder=10, linewidth=1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(
        r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx,
        fontsize=14,
        xy=(x1, y1),
        xycoords="data",
        xytext=(xoff, 10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment="left",
        verticalalignment="top",
    )


def plt_gradients(x_train, y_train, f_compute_cost, f_compute_gradient):
    # ===============
    #  First subplot
    # ===============
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Print w vs cost to see minimum
    fix_b = 100
    w_array = np.linspace(-100, 500, 50)
    w_array = np.linspace(0, 400, 50)
    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = f_compute_cost(x_train, y_train, tmp_w, fix_b)
    ax[0].plot(w_array, cost, linewidth=1)
    ax[0].set_title("Cost vs w, with gradient; b set to 100")
    ax[0].set_ylabel("Cost")
    ax[0].set_xlabel("w")

    # plot lines for fixed b=100
    for tmp_w in [100, 200, 300]:
        fix_b = 100
        dj_dw, dj_db = f_compute_gradient(x_train, y_train, tmp_w, fix_b)
        j = f_compute_cost(x_train, y_train, tmp_w, fix_b)
        add_line(dj_dw, tmp_w, j, 30, ax[0])

    # ===============
    # Second Subplot
    # ===============

    tmp_b, tmp_w = np.meshgrid(np.linspace(-200, 200, 10), np.linspace(-100, 600, 10))
    U = np.zeros_like(tmp_w)
    V = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            U[i][j], V[i][j] = f_compute_gradient(
                x_train, y_train, tmp_w[i][j], tmp_b[i][j]
            )
    X = tmp_w
    Y = tmp_b
    n = -2
    color_array = np.sqrt(((V - n) / 2) ** 2 + ((U - n) / 2) ** 2)

    ax[1].set_title("Gradient shown in quiver plot")
    Q = ax[1].quiver(
        X,
        Y,
        U,
        V,
        color_array,
        units="width",
    )
    ax[1].quiverkey(
        Q, 0.9, 0.9, 2, r"$2 \frac{m}{s}$", labelpos="E", coordinates="figure"
    )
    ax[1].set_xlabel("w")
    ax[1].set_ylabel("b")
