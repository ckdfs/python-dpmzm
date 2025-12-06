import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


class BiasControlDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

        # 标准化：记录均值和方差，用于训练时归一化和推理时还原
        self.X_mean = X.mean(axis=0, keepdims=True)
        self.X_std = X.std(axis=0, keepdims=True) + 1e-8
        self.Y_mean = Y.mean(axis=0, keepdims=True)
        self.Y_std = Y.std(axis=0, keepdims=True) + 1e-8

        self.X = (X - self.X_mean) / self.X_std
        self.Y = (Y - self.Y_mean) / self.Y_std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 稍微加大网络容量：12 -> 128 -> 128 -> 64 -> 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(
    npz_path: str = "dataset_bias_control.npz",
    batch_size: int = 256,
    num_epochs: int = 500,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = BiasControlDataset(npz_path)
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = full_dataset.X.shape[1]
    output_dim = full_dataset.Y.shape[1]

    model = MLPRegressor(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Using device: {device}")
    print(f"Dataset size: {n_total}, train: {n_train}, val: {n_val}")
    print(f"Input dim: {input_dim}, output dim: {output_dim}")

    # 方便将验证集 MSE 换算为物理电压 RMSE
    # 这里取三路偏压标准差的平均值做一个代表性尺度
    Y_std_np = full_dataset.Y_std.reshape(-1)
    Y_std_mean = float(Y_std_np.mean())

    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)

        val_loss = val_loss_sum / n_val if n_val > 0 else 0.0

        # 将验证集 MSE 换算为电压 RMSE（单位 V，便于理解）
        val_rmse_std = np.sqrt(val_loss)
        val_rmse_V = val_rmse_std * Y_std_mean

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"val_RMSE≈{val_rmse_V:.4f} V"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "X_mean": full_dataset.X_mean,
                "X_std": full_dataset.X_std,
                "Y_mean": full_dataset.Y_mean,
                "Y_std": full_dataset.Y_std,
            }, "mlp_bias_control_best.pth")
            print("  -> Saved new best model to mlp_bias_control_best.pth")


if __name__ == "__main__":
    train()
