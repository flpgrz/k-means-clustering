import numpy as np
from numpy.matlib import repmat
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2

plt.ion()

def kmeans(data, n_cl, verbose=False):
    """
    Kmeans algorithm.
    
    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    verbose: bool
        whether or not to plot assignment at each iteration (default is True).

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    n_samples, dim = data.shape

    # initialize centers
    centers = data[np.random.choice(range(n_samples), size=n_cl)]
    old_labels = np.zeros(shape=n_samples)

    while True:  # stopping criterion

        # assign
        distances = np.zeros(shape=(n_samples, n_cl))
        for c_idx, c in enumerate(centers):
            distances[:, c_idx] = np.sum(np.square(data - repmat(c, n_samples, 1)), axis=1)

        new_labels = np.argmin(distances, axis=1)

        # re-estimate
        for l in range(0, n_cl):
            centers[l] = np.mean(data[new_labels == l], axis=0)

        if verbose:
            fig, ax = plt.subplots()
            ax.scatter(data[:, 0], data[:, 1], c=new_labels, s=40)
            ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
            plt.waitforbuttonpress()
            plt.close()

        if np.all(new_labels == old_labels):
            break

        # update
        old_labels = new_labels

    return new_labels


def main_kmeans_img(img_path, n_cl):
    """
    Main function to run kmeans for image segmentation.
    
    Parameters
    ----------
    img_path: str
        Path of the image to load and segment.

    Returns
    -------
    None
    """

    # load the image
    img = np.float32(cv2.imread(img_path))
    h, w, c = img.shape

    # add coordinates
    row_indexes = np.arange(0, h)
    col_indexes = np.arange(0, w)
    coordinates = np.zeros(shape=(h, w, 2))
    coordinates[..., 0] = normalize(repmat(row_indexes, w, 1).T)
    coordinates[..., 1] = normalize(repmat(col_indexes, h, 1))
    print(coordinates[..., 0])

    data = np.concatenate((img, coordinates), axis=-1)
    data = np.reshape(data, newshape=(w * h, 5))


    # solve kmeans optimization
    labels = kmeans(data, n_cl=n_cl, verbose=False)

    # visualize image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))  # axis of first graph
    ax[0].axis('off')
    ax[1].imshow(np.reshape(labels, (h, w)), cmap='hot')  #axis of second graph
    ax[1].axis('off')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_kmeans_img('img/road.jpg', n_cl=3)
    main_kmeans_img('img/dog.jpg', n_cl=2)
    main_kmeans_img('img/emma.png', n_cl=2)
