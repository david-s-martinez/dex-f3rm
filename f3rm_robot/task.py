from dataclasses import dataclass
from typing import List, Tuple

import torch
from jaxtyping import Float

from f3rm_robot.assets import get_asset_path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA

def plot_embeddings_2d(query_emb, other_embs, labels, title, filename):
    """
    Plots a 2D plot where all embeddings are plotted as vectors. 
    The most similar embedding to the query has the smallest angle and is highlighted.

    :param query_emb: Query embedding tensor of shape (num_channels,)
    :param other_embs: Embedding tensors of shape (num_items, num_channels)
    :param labels: Labels corresponding to other embeddings
    :param title: Title of the plot
    :param filename: Filename to save the plot
    """
    # Normalize the query embedding and other embeddings to unit vectors
    query_emb_normalized = query_emb / query_emb.norm()
    other_embs_normalized = other_embs / other_embs.norm(dim=1, keepdim=True)

    # Check the shape of embeddings before applying PCA
    print(f"Shape of other_embs_normalized before PCA: {other_embs_normalized.shape}")

    # Reduce dimensionality to 2D using PCA if the embeddings have more than 2 dimensions
    if other_embs_normalized.shape[1] > 2:
        pca = PCA(n_components=2)
        other_embs_normalized = pca.fit_transform(other_embs_normalized)

    print(f"Shape of other_embs_normalized after PCA: {other_embs_normalized.shape}")

    # Compute cosine similarities using torch.cosine_similarity (1D vectors)
    cosine_similarities = torch.cosine_similarity(query_emb_normalized.unsqueeze(0), torch.tensor(other_embs_normalized)).cpu().numpy()

    # Find index of the most similar embedding (the one with the largest cosine similarity)
    most_similar_idx = np.argmax(cosine_similarities)

    # Set up the plot
    plt.figure(figsize=(10, 10))
    
    # Plot all embeddings as vectors
    for i in range(other_embs_normalized.shape[0]):
        plt.quiver(0, 0, other_embs_normalized[i, 0], other_embs_normalized[i, 1], angles='xy', scale_units='xy', scale=1, color='blue')
    
    # Highlight the most similar embedding with a red vector
    plt.quiver(0, 0, other_embs_normalized[most_similar_idx, 0], other_embs_normalized[most_similar_idx, 1], angles='xy', scale_units='xy', scale=1, color='red', linewidth=3)

    # Plot the query embedding as a black vector
    plt.quiver(0, 0, query_emb_normalized[0], query_emb_normalized[1], angles='xy', scale_units='xy', scale=1, color='black', linewidth=3)

    # Add labels for each embedding
    for i, label in enumerate(labels):
        plt.text(other_embs_normalized[i, 0] * 1.1, other_embs_normalized[i, 1] * 1.1, label, fontsize=10)

    # Set the limits and title
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title(title)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save the plot
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_cosine_similarity(query_emb, other_embs, labels, title, filename):
    """
    Plots a bar chart showing cosine similarity between the query embedding and other embeddings.
    Highlights the most similar bar in red.

    :param query_emb: Query embedding tensor of shape (num_channels,)
    :param other_embs: Embedding tensors of shape (num_items, num_channels)
    :param labels: Labels corresponding to other embeddings
    :param title: Title of the plot
    :param filename: Filename to save the plot
    """
    # Ensure other_embs is 2D
    if other_embs.dim() == 3:
        other_embs = other_embs.squeeze(1)  # Convert (num_items, 1, num_channels) -> (num_items, num_channels)

    # Compute cosine similarities
    cosine_similarities = torch.cosine_similarity(query_emb.unsqueeze(0), other_embs).cpu().numpy()

    # Ensure cosine_similarities is 1D
    if cosine_similarities.ndim > 1:
        cosine_similarities = cosine_similarities.squeeze()

    # Check if the number of labels matches
    if len(labels) != len(cosine_similarities):
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of similarities ({len(cosine_similarities)}).")

    # Find index of the most similar embedding
    max_sim_index = np.argmax(cosine_similarities)

    # Create a color list for the bars
    colors = ['red' if i == max_sim_index else 'blue' for i in range(len(cosine_similarities))]

    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.barh(labels, cosine_similarities, color=colors)
    plt.xlabel("Cosine Similarity")
    plt.title(title)
    plt.xlim(-1, 1)  # Cosine similarity ranges from -1 to 1

    # Add values to bars
    for i, v in enumerate(cosine_similarities):
        plt.text(v, i, f"{v:.5f}", color='black', va='center', fontsize=10)

    # Save plot
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_embeddings_3d(query_emb, other_embs, labels, title, filename):
    """
    Plots the query embedding and other embeddings in a 3D space.
    
    :param query_emb: Query embedding tensor of shape (1, num_channels)
    :param other_embs: List of embedding tensors of shape (num_items, num_channels)
    :param labels: Labels corresponding to other embeddings
    :param title: Title of the plot
    :param filename: Filename to save the plot
    """
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Ensure query_emb and other_embs are 2D
    query_emb = query_emb.squeeze(0)  # (1, num_channels) → (num_channels,)
    other_embs = other_embs.squeeze(1) if other_embs.dim() == 3 else other_embs  # (num_items, 1, num_channels) → (num_items, num_channels)

    # Reduce dimensionality to 3D using PCA if necessary
    if query_emb.shape[-1] > 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X = torch.cat([query_emb.unsqueeze(0), other_embs], dim=0).cpu().numpy()  # Ensure both are 2D
        reduced_data = pca.fit_transform(X)
        
        query_emb, other_embs = reduced_data[0], reduced_data[1:]

    # Ensure other_embs is at least 2D
    other_embs = np.atleast_2d(other_embs)

    # Plot embeddings
    ax.scatter(other_embs[:, 0], other_embs[:, 1], other_embs[:, 2], c='b', label="Other Embeddings")
    ax.scatter(query_emb[0], query_emb[1], query_emb[2], c='r', marker='*', s=200, label="Query Embedding")

    # Add labels
    for i, label in enumerate(labels):
        ax.text(other_embs[i, 0], other_embs[i, 1], other_embs[i, 2], label, fontsize=8)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)
    ax.legend()

    # Save plot
    plt.savefig(filename)
    plt.close()


def sample_query_points(
    num_samples: int, mean: Tuple[float, float, float] = (0, 0, 0), std_dev: float = 0.0075
) -> Float[torch.Tensor, "num_qps 3"]:
    """
    Sample query points from a 3D gaussian with specified mean and standard deviation.
    We use the same standard deviation for the 3 dimensions. Use this to generate query points for a new Task.
    """
    assert std_dev > 0, "std_dev must be positive."

    mean = torch.tensor(mean).float()
    variance = std_dev**2
    gaussian = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=torch.eye(3) * variance)
    query_points = gaussian.sample(torch.Size([num_samples]))
    return query_points


@dataclass(frozen=True)
class Task:
    """
    A Task which is defined by the query points, and demo embeddings. We store the demo query point features and
    density, as to get the alpha-weighted features we need the voxel size which may vary.

    The averaging of the alpha-weighted features is done upstream in the optimization script.
    """

    name: str
    query_points: Float[torch.Tensor, "num_qps 3"]
    link_points: Float[torch.Tensor, "num_qps 3"]

    # Features and density for each demo
    demo_features: Float[torch.Tensor, "num_demos num_qps num_channels"]
    demo_density: Float[torch.Tensor, "num_demos num_qps 1"]
    demo_joints: Float[torch.Tensor, "num_demos num_joints"]
    demo_torques: Float[torch.Tensor, "num_demos num_torques"]

    def __post_init__(self):
        assert len(self.query_points) > 0, f"Query points cannot be empty for task {self.name}"
        assert len(self.demo_features) > 0, f"Must have at least one demo for task {self.name}"
        assert (
            self.demo_features.shape[:2] == self.demo_density.shape[:2]
        ), f"Features and density must have same number of demos and query points."
        assert self.demo_density.ndim == 3 and self.demo_density.shape[-1] == 1, "Density must be 3D with 1 channel."
        assert self.demo_features.ndim == 3, "Features must be 3D."

    @property
    def num_demos(self) -> int:
        return len(self.demo_features)

    @property
    def num_query_points(self) -> int:
        return len(self.query_points)

    @property
    def num_channels(self) -> int:
        return self.demo_features.shape[-1]

grasp_primitives_dict = {
    "Cylinder grasp for larger objects like boxes or bottles or balls": [ "teddy_bear_head","white_mug_body","teddy_bear_legs"],
    "Hook grasp for objects like tools or bag straps": ["black_headphones_band", "gray_sweep_handle"],
    "Pinch grasp for small objects like candies or fruits and small parts": ["teddy_bear_ears", "white_mug_handle", "peach", "black_foam_cube"],
    "Lumbrical grasp for flat or boxy objects like books and packages": ["beige_bowl", "crackers_box"],
    "Tripod grasp for objects requiring precision like small toys or pc mouse": ["teddy_bear_arms", "mentos", "rubiks_cube"]
}

def get_tasks(task_names = None, tasks_folder = "hithand_tasks_og_fk") -> List[Task]:
    """Load all tasks from cache. Note these are for ClIP ViT-L/14@336px."""
    if task_names == None:
        task_names = ["teddy_bear_legs",
                    "teddy_bear_arms", 
                    "teddy_bear_ears", 
                    "teddy_bear_head",
                    "beige_bowl",
                    "white_mug_body",
                    "white_mug_handle",
                    "black_headphones_band",
                    "crackers_box",
                    "gray_sweep_handle",
                    "mentos",
                    "black_foam_cube",
                    "rubiks_cube",
                    "peach"] # all new corrected
    task_paths = [get_asset_path(f"{tasks_folder}/{task_name}.pt") for task_name in task_names]
    tasks = [torch.load(task_path) for task_path in task_paths]
    return tasks
