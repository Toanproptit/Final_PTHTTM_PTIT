"""MLP nhỏ cho vector đã tiền xử lý; sigmoid chỉ áp dụng khi suy luận."""
import torch
from torch import nn

class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int = 15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)
