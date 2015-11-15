import matplotlib.pyplot as plt


def plot_pictures(lImg, rImg):
    # --- PLOTTING ---
    plt.ion()
    fig = plt.figure()

    ax_left = fig.add_subplot(121)
    mapp_left = ax_left.pcolor(lImg)
    fig.colorbar(mapp_left)
    ax_left.set_title("left_Img = A = Ksi")


    ax_right = fig.add_subplot(122)
    mapp_right = ax_right.pcolor(rImg)
    fig.colorbar(mapp_right)
    ax_right.set_title("right_Img = B = Eta")

    plt.show(block=False)

