import numpy as np
from engine import Value
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

from training.core import train_with_sgd
from nn import MLP


def main():
    features, labels = make_moons(n_samples=100, noise=0.1)

    # make y be -1 or 1
    labels = labels * 2 - 1

    # visualize in 2D
    # plt.figure(figsize=(5, 5))
    # plt.scatter(features[:, 0], features[:, 1], c=labels, s=20, cmap='jet')
    # plt.show()

    mlp = MLP(2, [16, 16], 1)
    mlp, final_loss, final_predictions, final_accuracy = train_with_sgd(
        mlp, features, labels
    )
    print(f"Final loss: {final_loss}")
    print(f"Final predictions: {final_predictions}")
    print(f"Final accuracy: {final_accuracy}")

    # How to visualise the decision boundary?
    # Use a grid
    padding_factor = 0.05
    x_min, x_max = features[:, 0].min(), features[:, 0].max()
    x_padding = (x_max - x_min) * padding_factor
    y_min, y_max = features[:, 1].min(), features[:, 1].max()
    y_padding = (y_max - y_min) * padding_factor
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_padding, x_max + x_padding, 20),
        np.linspace(y_min - y_padding, y_max + y_padding, 20),
    )

    predictions = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        output = mlp([Value(x), Value(y)])[0]
        predictions.append(output.data)

    predictions_np = np.array(predictions).reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, predictions_np, alpha=0.3, cmap="jet")
    plt.scatter(features[:, 0], features[:, 1], c=labels, s=20, cmap=plt.cm.coolwarm)
    plt.show()


if __name__ == "__main__":
    main()
