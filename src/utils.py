import matplotlib.pyplot as plt


def plot_signal(sparse, smooth, signal):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis("off")
    ax1.imshow(sparse)
    ax2.axis("off")
    ax2.imshow(smooth)
    ax3.axis("off")
    ax3.imshow(signal)
