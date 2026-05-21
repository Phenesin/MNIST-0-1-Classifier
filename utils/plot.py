import matplotlib.pyplot as plt

def plot_metric(
        sgd_values,
        momentum_values,
        adam_values,
        title,
        y_label
):
    epochs = range(1, len(sgd_values) + 1)

    plt.figure(figsize = (8,5))

    plt.plot(
        epochs,
        sgd_values,
        marker = 'o',
        label = "SGD"
    )

    plt.plot(
        epochs,
        momentum_values,
        marker = "o",
        label = "Momentum"
    )

    plt.plot(
        epochs,
        adam_values,
        marker = "o",
        label = "Adam"
    )


    plt.xlabel("Epoch")
    plt.ylabel(y_label)

    plt.title(title)
    plt.legend()
    plt.show()