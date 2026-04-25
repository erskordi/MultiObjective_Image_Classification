import torch
from torch.utils.data import Dataset


class SparseImageGraphBuilder:
    """
    Builds sparse graph topology from image grids.

    Each pixel is one node.
    Only edge_index is stored.
    """

    def __init__(
        self,
        connectivity: int = 4,
        add_self_loops: bool = True,
        undirected: bool = True,
        device: str | torch.device | None = None,
    ):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be either 4 or 8.")

        self.connectivity = connectivity
        self.add_self_loops = add_self_loops
        self.undirected = undirected
        self.device = device

    @staticmethod
    def node_id(row: int, col: int, width: int) -> int:
        return row * width + col

    def grid_edge_index(self, height: int, width: int) -> torch.Tensor:
        edges = []

        for r in range(height):
            for c in range(width):
                u = self.node_id(r, c, width)

                neighbors = []
                if r > 0:
                    neighbors.append((r - 1, c))
                if r < height - 1:
                    neighbors.append((r + 1, c))
                if c > 0:
                    neighbors.append((r, c - 1))
                if c < width - 1:
                    neighbors.append((r, c + 1))

                if self.connectivity == 8:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < height and 0 <= cc < width:
                            neighbors.append((rr, cc))

                for rr, cc in neighbors:
                    v = self.node_id(rr, cc, width)
                    edges.append((u, v))

                if self.add_self_loops:
                    edges.append((u, u))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        if self.undirected:
            rev = edge_index[[1, 0], :]
            edge_index = torch.cat([edge_index, rev], dim=1)
            edge_index = torch.unique(edge_index, dim=1)

        if self.device is not None:
            edge_index = edge_index.to(self.device)

        return edge_index

    def build_graph(self, height: int, width: int) -> dict:
        num_nodes = height * width
        edge_index = self.grid_edge_index(height, width)
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
        }

    def build_graph_from_sample(self, sample: dict) -> dict:
        height = int(sample["height"])
        width = int(sample["width"])
        graph = self.build_graph(height, width)
        return {**sample, **graph}


class SparseGraphReadyDataset(Dataset):
    """
    Wraps preprocessed image samples and adds sparse graph topology.
    """

    def __init__(
        self,
        base_dataset,
        graph_builder: SparseImageGraphBuilder,
        cache_graphs_by_size: bool = True,
    ):
        self.base_dataset = base_dataset
        self.graph_builder = graph_builder
        self.cache_graphs_by_size = cache_graphs_by_size
        self._graph_cache = {}

    def __len__(self):
        return len(self.base_dataset)

    def _get_cached_graph(self, height: int, width: int) -> dict:
        key = (height, width)
        if key not in self._graph_cache:
            self._graph_cache[key] = self.graph_builder.build_graph(height, width)
        return self._graph_cache[key]

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        height = int(sample["height"])
        width = int(sample["width"])

        if self.cache_graphs_by_size:
            graph = self._get_cached_graph(height, width)
            return {**sample, **graph}

        return self.graph_builder.build_graph_from_sample(sample)


def sparse_graph_collate_fn(batch: list[dict]) -> dict:
    """
    Unified batch schema for PGNNNetSparse.

    Returns:
        {
            "x1":         (total_nodes, F1),
            "x2":         (total_nodes, F2),
            "edge_index": (2, total_edges),
            "batch":      (total_nodes,),
            "label":      (B,),
            "num_graphs": int,
            "ptr":        (B+1,),
            "sizes":      list[(H, W)],
            "image":      list[Tensor],   # optional debug key
        }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    x1_list = []
    x2_list = []
    edge_list = []
    batch_vec_list = []
    labels = []
    images = []
    sizes = []
    ptr = [0]

    node_offset = 0

    for graph_idx, item in enumerate(batch):
        x1 = item["x1"]
        x2 = item["x2"]
        edge_index = item["edge_index"]
        num_nodes = int(item["num_nodes"])

        x1_list.append(x1)
        x2_list.append(x2)
        edge_list.append(edge_index + node_offset)
        batch_vec_list.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        labels.append(item["label"])
        images.append(item["image"])
        sizes.append((int(item["height"]), int(item["width"])))

        node_offset += num_nodes
        ptr.append(node_offset)

    return {
        "x1": torch.cat(x1_list, dim=0),
        "x2": torch.cat(x2_list, dim=0),
        "edge_index": torch.cat(edge_list, dim=1),
        "batch": torch.cat(batch_vec_list, dim=0),
        "label": torch.stack(labels, dim=0),
        "num_graphs": len(batch),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "sizes": sizes,
        "image": images,
    }

class GraphReadyDataset(Dataset):
    """
    Wraps a preprocessed dataset and adds graph structures.

    Input dataset should return dicts from GenericGraphReadyImageDataset.
    Output dataset returns:
        - x1
        - x2
        - label
        - image
        - edge_index
        - adjacency
        - adjacency_normalized
        - laplacian
        - num_nodes
        - height
        - width
        - channels
    """

    def __init__(
        self,
        base_dataset,
        graph_builder: SparseImageGraphBuilder,
        cache_graphs_by_size: bool = True,
    ):
        """
        Args:
            base_dataset:
                dataset returning samples from preprocess.py
            graph_builder:
                ImageGraphBuilder instance
            cache_graphs_by_size:
                if True, graph topology/matrices are cached for each unique (H, W)
        """
        self.base_dataset = base_dataset
        self.graph_builder = graph_builder
        self.cache_graphs_by_size = cache_graphs_by_size
        self._graph_cache = {}

    def __len__(self):
        return len(self.base_dataset)

    def _get_cached_graph(self, height: int, width: int) -> dict:
        key = (height, width)

        if key not in self._graph_cache:
            self._graph_cache[key] = self.graph_builder.build_graph(height, width)

        return self._graph_cache[key]

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        height = int(sample["height"])
        width = int(sample["width"])

        if self.cache_graphs_by_size:
            graph = self._get_cached_graph(height, width)
            out = {
                **sample,
                **graph,
            }
        else:
            out = self.graph_builder.build_graph_from_sample(sample)

        return out


def graph_collate_fn(batch: list[dict]) -> dict:
    """
    Simple collate function for fixed-size graphs.

    This version assumes all items in the batch have the same image size
    and therefore the same graph size/topology.

    Returns:
        dict with:
            - x1: (B, N, F1)
            - x2: (B, N, F2)
            - label: (B,)
            - image: (B, C, H, W)
            - edge_index: (2, E)
            - adjacency: (N, N)
            - adjacency_normalized: (N, N)
            - laplacian: (N, N)
            - num_nodes: int
            - height: int
            - width: int
            - channels: int
    """
    if len(batch) == 0:
        raise ValueError("Empty batch provided to graph_collate_fn.")

    first = batch[0]
    height = int(first["height"])
    width = int(first["width"])
    num_nodes = int(first["num_nodes"])
    channels = int(first["channels"])

    for item in batch:
        if int(item["height"]) != height or int(item["width"]) != width:
            raise ValueError(
                "graph_collate_fn currently requires all samples in a batch "
                "to have the same height and width."
            )

    x1 = torch.stack([item["x1"] for item in batch], dim=0)
    x2 = torch.stack([item["x2"] for item in batch], dim=0)
    label = torch.stack([item["label"] for item in batch], dim=0)
    image = torch.stack([item["image"] for item in batch], dim=0)

    return {
        "x1": x1,
        "x2": x2,
        "label": label,
        "image": image,
        "edge_index": first["edge_index"],
        "adjacency": first["adjacency"],
        "adjacency_normalized": first["adjacency_normalized"],
        "laplacian": first["laplacian"],
        "num_nodes": num_nodes,
        "height": height,
        "width": width,
        "channels": channels,
    }


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from gnn_preprocess import GenericImagePreprocessor, GenericGraphReadyImageDataset

    # Example with FashionMNIST
    preprocessor = GenericImagePreprocessor(
        normalize=True,
        mean=(0.5,),
        std=(0.5,),
        add_spatial_coords=True,
        include_intensity=True,
    )

    base_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    pre_ds = GenericGraphReadyImageDataset(
        base_dataset=base_dataset,
        preprocessor=preprocessor,
        use_patches=False,
    )

    graph_builder = SparseImageGraphBuilder(
        connectivity=4,
        add_self_loops=True,
    )

    graph_ds = GraphReadyDataset(
        base_dataset=pre_ds,
        graph_builder=graph_builder,
        cache_graphs_by_size=True,
    )

    sample = graph_ds[0]


    print("x1:", sample["x1"].shape)
    print("x2:", sample["x2"].shape)
    print("label:", sample["label"].shape)
    print("image:", sample["image"].shape)
    print("edge_index:", sample["edge_index"].shape)
    print("num_nodes:", sample["num_nodes"])