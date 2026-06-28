import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.flop_counter import FlopCounterMode
from zeus.monitor import ZeusMonitor


def _resnet_weights(variant: str):
    if variant == "18":
        return models.ResNet18_Weights.DEFAULT
    if variant == "34":
        return models.ResNet34_Weights.DEFAULT
    if variant == "50":
        return models.ResNet50_Weights.DEFAULT
    if variant == "101":
        return models.ResNet101_Weights.DEFAULT
    if variant == "152":
        return models.ResNet152_Weights.DEFAULT
    raise ValueError(f"Unsupported ResNet variant: {variant}")


class _ResNetV3Base(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **_: object,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.use_cuda_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(enabled=self.use_cuda_amp)

        weights = _resnet_weights(variant) if pretrained else None
        if variant == "18":
            self.model = models.resnet18(weights=weights)
        elif variant == "34":
            self.model = models.resnet34(weights=weights)
        elif variant == "50":
            self.model = models.resnet50(weights=weights)
        elif variant == "101":
            self.model = models.resnet101(weights=weights)
        elif variant == "152":
            self.model = models.resnet152(weights=weights)

        self._adapt_input_stem(input_channels)

        classifier_in_features = self.model.fc.in_features
        if in_features is not None and in_features != classifier_in_features:
            print(
                f"Ignoring in_features={in_features}; backbone exposes "
                f"{classifier_in_features} classifier inputs."
            )
        self.model.fc = nn.Linear(classifier_in_features, num_classes)

    def _adapt_input_stem(self, input_channels: int) -> None:
        if input_channels == 3:
            return

        first_conv = self.model.conv1
        replacement = nn.Conv2d(
            input_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=(first_conv.bias is not None),
        )

        with torch.no_grad():
            if input_channels == 1:
                replacement.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
            else:
                repeated = first_conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
                replacement.weight.copy_(repeated / float(input_channels))
            if first_conv.bias is not None:
                replacement.bias.copy_(first_conv.bias)

        self.model.conv1 = replacement

    def forward(self, x):
        return self.model(x)

    def train_model(self, dataloader, loss_fn, optimizer, device, monitor):
        self.train()
        total_loss = 0.0
        monitor.begin_window("epoch")
        mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0

        for X, y in dataloader:
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                pred = self(X)
                loss = loss_fn(pred, y)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()

        measurement = monitor.end_window("epoch")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        return total_loss, round(measurement.total_energy, 2), round(mem_utilized / 1024**2, 2)

    @torch.no_grad()
    def test_model(self, dataloader, loss_fn, device):
        monitor = ZeusMonitor(gpu_indices=[0])
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        total_flops = 0
        test_loss, correct = 0.0, 0.0
        monitor.begin_window("testing")
        mem_before = 0.0
        all_preds, all_labels = [], []

        for X, y in dataloader:
            X = X.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            with FlopCounterMode(display=False) as flop_counter:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    pred = self(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    all_preds.append(pred.cpu())
                    all_labels.append(y.cpu())
            total_flops += flop_counter.get_total_flops()

        measurement = monitor.end_window("testing")
        mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0
        mem_utilized = mem_after - mem_before
        test_loss /= num_batches
        correct /= size
        average_flops = total_flops / size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_parameters}")

        return (
            100 * correct,
            test_loss,
            average_flops,
            round(measurement.total_energy, 2),
            round(mem_utilized, 2),
            num_parameters,
            all_preds,
            all_labels,
        )

    def save_model(self, path):
        torch.save(self.state_dict(), path)


class ResNet18(_ResNetV3Base):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **kwargs: object,
    ):
        super().__init__(
            variant="18",
            num_classes=num_classes,
            input_channels=input_channels,
            in_features=in_features,
            pretrained=pretrained,
            **kwargs,
        )

class ResNet34(_ResNetV3Base):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **kwargs: object,
    ):
        super().__init__(
            variant="34",
            num_classes=num_classes,
            input_channels=input_channels,
            in_features=in_features,
            pretrained=pretrained,
            **kwargs,
        )

class ResNet50(_ResNetV3Base):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **kwargs: object,
    ):
        super().__init__(
            variant="50",
            num_classes=num_classes,
            input_channels=input_channels,
            in_features=in_features,
            pretrained=pretrained,
            **kwargs,
        )

class ResNet101(_ResNetV3Base):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **kwargs: object,
    ):
        super().__init__(
            variant="101",
            num_classes=num_classes,
            input_channels=input_channels,
            in_features=in_features,
            pretrained=pretrained,
            **kwargs,
        )

class ResNet152(_ResNetV3Base):
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        in_features: int | None = None,
        pretrained: bool = True,
        **kwargs: object,
    ):
        super().__init__(
            variant="152",
            num_classes=num_classes,
            input_channels=input_channels,
            in_features=in_features,
            pretrained=pretrained,
            **kwargs,
        )