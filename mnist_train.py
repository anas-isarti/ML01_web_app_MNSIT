import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (N, 1,  28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # → (N, 32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # → (N, 64, 24, 24)
        self.pool  = nn.MaxPool2d(2)                   # → (N, 64, 12, 12)
        self.drop1 = nn.Dropout(0.25)
        # flatten                                       # → (N, 9216)  [64 * 12 * 12]
        self.fc1   = nn.Linear(9216, 128)              # → (N, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 10)                # → (N, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # (N, 32, 26, 26)
        x = torch.relu(self.conv2(x))   # (N, 64, 24, 24)
        x = self.pool(x)                # (N, 64, 12, 12)
        x = self.drop1(x)
        x = torch.flatten(x, 1)        # (N, 9216)
        x = torch.relu(self.fc1(x))     # (N, 128)
        x = self.drop2(x)
        x = self.fc2(x)                 # (N, 10)
        return x


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
    return correct / len(loader.dataset)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # num_workers=0 et pin_memory=False : obligatoire sous Windows avec peu de RAM
    train_loader = DataLoader(train_dataset, batch_size=64,  shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

    model     = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    report_lines = [
        "Architecture: Conv2d(1→32,k=3) → ReLU → Conv2d(32→64,k=3) → ReLU"
        " → MaxPool2d(2) → Dropout(0.25) → Flatten → Linear(9216→128)"
        " → ReLU → Dropout(0.5) → Linear(128→10)",
        f"Device: {device}",
        "",
        "Training:",
    ]

    print(f"Training on {device}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        accuracy   = evaluate(model, test_loader)
        line = f"  Epoch {epoch}/{epochs}  loss: {train_loss:.4f}  test accuracy: {accuracy:.4f}"
        print(line)
        report_lines.append(line)

    final_acc = evaluate(model, test_loader)
    summary = f"\nFinal test accuracy: {final_acc:.4f}"
    print(summary)
    report_lines.append(summary)

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Saved: mnist_cnn.pt")

    with open("mnist_training_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("Saved: mnist_training_report.txt")


if __name__ == "__main__":
    main()
