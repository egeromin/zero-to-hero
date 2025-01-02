from nn import MLP
from training.core import train_with_sgd


def main():
    # First, generate some training data.
    features = [
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [-1, 1, -1, 1],
        [-1, -1, -1, 1],
    ]
    labels = [-1, 1, -1, 1]

    mlp = MLP(4, [2, 3, 2], 1)

    mlp, final_loss, final_predictions, final_accuracy = train_with_sgd(
        mlp, features, labels
    )

    print(f"Final loss: {final_loss}")
    print(f"Final predictions: {final_predictions}")
    print(f"Final accuracy: {final_accuracy}")


if __name__ == "__main__":
    main()
