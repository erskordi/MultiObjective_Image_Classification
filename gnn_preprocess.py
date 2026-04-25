import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class GenericImagePreprocessor:
    """
    Generic preprocessing utilities for image-to-graph pipelines.

    Supports:
    - grayscale images: (H, W) or (1, H, W)
    - color images: (H, W, C) or (C, H, W)
    - arbitrary spatial sizes
    - optional normalization
    - optional patch extraction
    - conversion of images to node-feature tensors

    Main idea:
    each pixel becomes one node, and this module prepares the node features.
    Edge construction is handled later in graph_build.py.
    """

    def __init__(
        self,
        normalize: bool = False,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        add_spatial_coords: bool = True,
        include_intensity: bool = True,
        smoothing_kernel_size: int = 3,
    ):
        """
        Args:
            normalize:
                If True, apply channel-wise normalization using mean/std.
            mean, std:
                Channel-wise normalization statistics.
                Required when normalize=True unless normalization is handled upstream.
            add_spatial_coords:
                If True, append normalized x,y coordinates to each node feature.
            include_intensity:
                If True, include pixel/channel values as part of node features.
            smoothing_kernel_size:
                Kernel size for the second feature view. Must be odd.
        """
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.add_spatial_coords = add_spatial_coords
        self.include_intensity = include_intensity
        self.smoothing_kernel_size = smoothing_kernel_size

        if self.smoothing_kernel_size % 2 == 0:
            raise ValueError("smoothing_kernel_size must be odd.")

        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("mean and std must be provided when normalize=True.")

    @staticmethod
    def to_channel_first(image: torch.Tensor) -> torch.Tensor:
        """
        Converts image to shape (C, H, W).

        Supported input formats:
        - (H, W)
        - (1, H, W)
        - (3, H, W)
        - (H, W, C)

        Returns:
            image tensor with shape (C, H, W)
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")

        if image.dim() == 2:
            # (H, W) -> (1, H, W)
            return image.unsqueeze(0)

        if image.dim() != 3:
            raise ValueError("Expected image with 2 or 3 dimensions.")

        # Already channel-first: (C, H, W)
        if image.shape[0] in (1, 3, 4):
            return image

        # Assume channel-last: (H, W, C)
        if image.shape[-1] in (1, 3, 4):
            return image.permute(2, 0, 1)

        raise ValueError(
            f"Could not infer channel order from shape {tuple(image.shape)}. "
            "Expected grayscale or color image."
        )

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies channel-wise normalization to image of shape (C, H, W).
        """
        image = self.to_channel_first(image).float()

        if not self.normalize:
            return image

        c = image.shape[0]
        if len(self.mean) != c or len(self.std) != c:
            raise ValueError(
                f"Normalization stats length mismatch: image has {c} channels, "
                f"but mean/std lengths are {len(self.mean)} and {len(self.std)}."
            )

        mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device).view(c, 1, 1)
        std = torch.tensor(self.std, dtype=image.dtype, device=image.device).view(c, 1, 1)
        return (image - mean) / std

    @staticmethod
    def image_shape(image: torch.Tensor) -> tuple[int, int]:
        """
        Returns spatial shape (H, W) for any supported image layout.
        """
        image = GenericImagePreprocessor.to_channel_first(image)
        _, h, w = image.shape
        return h, w

    @staticmethod
    def num_channels(image: torch.Tensor) -> int:
        """
        Returns number of channels.
        """
        image = GenericImagePreprocessor.to_channel_first(image)
        c, _, _ = image.shape
        return c

    @staticmethod
    def pad_image(image: torch.Tensor, pad: int, mode: str = "reflect") -> torch.Tensor:
        """
        Pads a channel-first image tensor.

        Args:
            image: shape (C, H, W), (H, W), or (H, W, C)
            pad: number of pixels per side
            mode: padding mode for torch.nn.functional.pad

        Returns:
            padded image tensor in shape (C, H, W)
        """
        image = GenericImagePreprocessor.to_channel_first(image)
        return F.pad(image, (pad, pad, pad, pad), mode=mode)

    @staticmethod
    def extract_patch(
        image: torch.Tensor,
        center_y: int,
        center_x: int,
        patch_size: int,
        pad_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Extracts a square patch centered at (center_y, center_x).

        Args:
            image: input image in any supported format
            center_y, center_x: center location
            patch_size: odd patch size
            pad_mode: padding mode

        Returns:
            patch tensor of shape (C, P, P)
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")

        image = GenericImagePreprocessor.to_channel_first(image)
        _, h, w = image.shape
        r = patch_size // 2

        padded = F.pad(image, (r, r, r, r), mode=pad_mode)
        y = center_y + r
        x = center_x + r

        patch = padded[:, y - r:y + r + 1, x - r:x + r + 1]
        return patch

    def image_to_node_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Converts an image into node features.

        Each pixel becomes one node.

        Feature construction:
        - if include_intensity=True:
            include all channel values at that pixel
        - if add_spatial_coords=True:
            append normalized x and y coordinates

        Args:
            image: input image in any supported format

        Returns:
            node_features: (N, F)
                N = H * W
                F = C (+2 if spatial coords are added)
        """
        image = self.normalize_image(image)
        c, h, w = image.shape

        features = []

        if self.include_intensity:
            # (C, H, W) -> (H, W, C) -> (N, C)
            pixel_values = image.permute(1, 2, 0).reshape(-1, c)
            features.append(pixel_values)

        if self.add_spatial_coords:
            ys, xs = torch.meshgrid(
                torch.linspace(0.0, 1.0, h, device=image.device),
                torch.linspace(0.0, 1.0, w, device=image.device),
                indexing="ij",
            )
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            features.extend([xs, ys])

        if not features:
            raise ValueError(
                "At least one of include_intensity or add_spatial_coords must be True."
            )

        return torch.cat(features, dim=1)

    def smooth_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Produces a lightly smoothed version of the image using average pooling.

        Args:
            image: input image in any supported format

        Returns:
            smoothed image tensor of shape (C, H, W)
        """
        image = self.normalize_image(image)
        k = self.smoothing_kernel_size
        p = k // 2

        # Use module-style pooling to avoid functional call-site issues.
        pool = torch.nn.AvgPool2d(kernel_size=k, stride=1, padding=p)
        smoothed = pool(image.unsqueeze(0)).squeeze(0)

        return smoothed

    def image_to_two_feature_views(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produces two feature views from the same image.

        View 1:
            node features from original image

        View 2:
            node features from smoothed image

        Args:
            image: input image in any supported format

        Returns:
            x1, x2:
                tensors of shape (N, F1) and (N, F2)
        """
        image_cf = self.normalize_image(image)
        x1 = self.image_to_node_features(image_cf)
        x2 = self.image_to_node_features(self.smooth_image(image_cf))
        return x1, x2

    @staticmethod
    def label_to_tensor(label: int | torch.Tensor) -> torch.Tensor:
        """
        Converts scalar label to torch.long tensor.
        """
        if isinstance(label, torch.Tensor):
            return label.long()
        return torch.tensor(label, dtype=torch.long)


class GenericGraphReadyImageDataset(Dataset):
    """
    Dataset wrapper that prepares images for later graph construction.

    This wrapper does NOT build graph edges.
    It returns:
        - x1: first node-feature view
        - x2: second node-feature view
        - label
        - spatial metadata
        - normalized channel-first image

    It can wrap datasets returning:
        (image, label)
    where image may be grayscale or color and may have arbitrary size.
    """

    def __init__(
        self,
        base_dataset,
        preprocessor: GenericImagePreprocessor,
        use_patches: bool = False,
        patch_size: int = 7,
        patch_center: tuple[int, int] | None = None,
        pad_mode: str = "reflect",
    ):
        """
        Args:
            base_dataset:
                Underlying dataset returning (image, label)
            preprocessor:
                GenericImagePreprocessor instance
            use_patches:
                If True, extract one patch from each image
            patch_size:
                Odd patch size if use_patches=True
            patch_center:
                Optional fixed patch center (y, x).
                If None, image center is used.
            pad_mode:
                Padding mode for patch extraction
        """
        self.base_dataset = base_dataset
        self.preprocessor = preprocessor
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.patch_center = patch_center
        self.pad_mode = pad_mode

        if self.use_patches and self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd when use_patches=True.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # Many torchvision datasets return PIL images.
        # Convert to tensor if needed.
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        image = self.preprocessor.to_channel_first(image).float()

        if self.use_patches:
            h, w = self.preprocessor.image_shape(image)
            if self.patch_center is None:
                cy, cx = h // 2, w // 2
            else:
                cy, cx = self.patch_center

            image = self.preprocessor.extract_patch(
                image=image,
                center_y=cy,
                center_x=cx,
                patch_size=self.patch_size,
                pad_mode=self.pad_mode,
            )

        image = self.preprocessor.normalize_image(image)
        x1, x2 = self.preprocessor.image_to_two_feature_views(image)
        h, w = self.preprocessor.image_shape(image)
        c = self.preprocessor.num_channels(image)
        y = self.preprocessor.label_to_tensor(label)

        return {
            "x1": x1,            # (N, F1)
            "x2": x2,            # (N, F2)
            "label": y,          # scalar tensor
            "height": h,
            "width": w,
            "image": image,      # (C, H, W), optional but useful
        }


if __name__ == "__main__":
    from torchvision import datasets

    # Example 1: FashionMNIST (grayscale)
    fm_pre = GenericImagePreprocessor(
        normalize=True,
        mean=(0.5,),
        std=(0.5,),
        add_spatial_coords=True,
        include_intensity=True,
    )

    fm_base = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    fm_ds = GenericGraphReadyImageDataset(
        base_dataset=fm_base,
        preprocessor=fm_pre,
        use_patches=False,
    )

    fm_sample = fm_ds[0]
    print("FashionMNIST")
    print("x1:", fm_sample["x1"].shape)
    print("x2:", fm_sample["x2"].shape)
    print("image:", fm_sample["image"].shape)
    print("label:", fm_sample["label"].item())

    """
    # Example 2: CIFAR-like RGB tensor
    rgb_image = torch.rand(3, 32, 32)
    rgb_label = 2

    class DummyDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return rgb_image, rgb_label

    rgb_pre = GenericImagePreprocessor(
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        add_spatial_coords=True,
        include_intensity=True,
    )

    rgb_ds = GenericGraphReadyImageDataset(
        base_dataset=DummyDataset(),
        preprocessor=rgb_pre,
        use_patches=False,
    )

    rgb_sample = rgb_ds[0]
    print("\nRGB example")
    print("x1:", rgb_sample["x1"].shape)
    print("x2:", rgb_sample["x2"].shape)
    print("image:", rgb_sample["image"].shape)
    print("label:", rgb_sample["label"].item())
    """