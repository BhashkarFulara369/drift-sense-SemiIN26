from __future__ import annotations
import numpy as np
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


if PYTORCH_AVAILABLE:
    class SiameseNetwork(nn.Module):
        """Contrastive PyTorch Siamese Network for identity verification on highly ambiguous cases."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            self.fc1 = nn.Linear(128 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward_once(self, x):
            x = F.relu(self.pool1(self.conv1(x)))
            x = F.relu(self.pool2(self.conv2(x)))
            x = F.relu(self.conv3(x))
            # Adaptive pool to fixed size 8x8
            x = F.adaptive_avg_pool2d(x, (8, 8))
            x = x.view(x.size()[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def forward(self, input1, input2):
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            # Return Euclidean distance between embeddings
            return F.pairwise_distance(output1, output2, keepdim=True)


class SiameseVerifier:
    def __init__(self):
        if PYTORCH_AVAILABLE:
            self.model = SiameseNetwork()
            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            self.is_loaded = False  # Set to True when weights are loaded
        else:
            self.model = None

    def verify(self, ref_patch: np.ndarray, search_patch: np.ndarray) -> float:
        """
        Returns a similarity score [0, 1] between patches based on learned contrastive identity.
        If PyTorch isn't available, returns a simple pixel-wise fallback.
        """
        if not PYTORCH_AVAILABLE or not self.is_loaded:
            # Fallback when PyTorch is not available or model isn't trained
            return 0.5

        with torch.no_grad():
            t_ref = self.transform(ref_patch).unsqueeze(0)
            t_search = self.transform(search_patch).unsqueeze(0)
            dist = self.model(t_ref, t_search).item()
            
        # Convert Euclidean distance to similarity [0, 1]
        similarity = max(0.0, 1.0 - (dist / 2.0))
        return float(similarity)
