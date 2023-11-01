import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_accuracy(arrays, labels, save_path="accuracy_plot.svg"):
    """
    Plot multiple accuracy arrays against epoch numbers.

    :param arrays: List of arrays containing accuracy values.
    :param labels: List of labels for the arrays.
    :param save_path: Path to save the SVG plot.
    """
    # Ensure the number of arrays matches the number of labels
    assert len(arrays) == len(labels), "Number of arrays must match number of labels."

    # Plot each array
    for i, arr in enumerate(arrays):
        plt.plot(arr, label=labels[i])

    # Set plot title and labels
    plt.title("Accuracy R@1 vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy R@1")
    plt.legend()

    # Save the plot as SVG and show it
    plt.savefig(save_path, format='svg')
    plt.show()


def plot_loss(arrays, labels, save_path="accuracy_plot.svg"):
    """
    Plot multiple accuracy arrays against epoch numbers.

    :param arrays: List of arrays containing accuracy values.
    :param labels: List of labels for the arrays.
    :param save_path: Path to save the SVG plot.
    """
    # Ensure the number of arrays matches the number of labels
    assert len(arrays) == len(labels), "Number of arrays must match number of labels."

    # Plot each array
    for i, arr in enumerate(arrays):
        plt.plot(arr, label=labels[i])

    # Set plot title and labels
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot as SVG and show it
    plt.savefig(save_path, format='svg')
    #plt.show()


# Example usage:
# arrays = [array1, array2, array3]
# labels = ["Model A", "Model B", "Model C"]
# plot_accuracy(arrays, labels)


from sklearn.preprocessing import StandardScaler


def visualize_embeddings(embeddings, targets, class_names=None, perplexity=30, learning_rate=200):
    """
    Visualize embeddings using t-SNE in 2D space.

    :param embeddings: Tensor of shape [B, 64, 21, 21]
    :param targets: Tensor of shape [B]
    :param class_names: List of class names
    :param perplexity: t-SNE perplexity
    :param learning_rate: t-SNE learning rate
    """
    # Flatten and normalize the embeddings
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).numpy()
    flattened_embeddings = StandardScaler().fit_transform(flattened_embeddings)
    targets_np = targets.numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(flattened_embeddings)

    # Plot the results
    plt.figure(figsize=(10, 8))

    unique_targets = np.unique(targets_np)
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_targets]

    for target, class_name in zip(unique_targets, class_names):
        indices = np.where(targets_np == target)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=class_name, alpha=0.6)

    plt.title("t-SNE visualization of CNN embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()
    plt.savefig("tsne_dn4avq.svg", format='svg')


# Example usage:
# visualize_embeddings(embeddings_tensor, targets_tensor)

if __name__ == "__main__":
    # Load the embeddings and targets
    # embeddings = np.load("embeddings.npy")
    # targets = np.load("targets.npy")
    #
    # # Visualize the embeddings
    # visualize_embeddings(embeddings, targets)
    # Extract all Prec@1 values
    out = []
    for ix, file in zip([11, 18], ["./miniImageNet_s5_av2.txt", "./miniImageNet_s5_av1.txt"]):
        with open(file, "r") as f:
            text = f.read()
        prec_values = re.findall(r"Loss (\d+\.\d+) \((\d+\.\d+)\)", text)
        prec_values = [x[1] for x in prec_values]

        # Take every 2nd value, skipping the first one
        ix += 98
        selected_values = prec_values[ix::ix]
        print(selected_values)
        # Convert to float and print
        selected_float_values = [float(value) for value in selected_values]
        out.append(selected_float_values)
    dn4avq, dn4 = out
    print(len(dn4avq), len(dn4))
    labels = ["DN4", "DN4AVQ"]
    plot_loss([dn4, dn4avq], labels, save_path="loss_plot_s5.svg")
