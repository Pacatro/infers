import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path

ONNX_PATH = "../onnx_models/mnist_fc_model.onnx"
DATA_PATH = "../data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FCNet(nn.Module):
    """
    The model architecture consist in a FC network (MLP) with 4 types of operations:

    - Linear operations (General Matrix multiplication)
    - Flatten operation (convert a tensor of shape (batch, width, height) into a 1D tensor)
    - ReLU operation (rectified linear unit)
    - Add operation (addition of two tensors)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2_left = nn.Linear(512, 200)
        self.fc2_left2 = nn.Linear(200, 100)
        self.fc2_right = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        left = torch.relu(self.fc2_left(x))
        left = torch.relu(self.fc2_left2(left))
        right = torch.relu(self.fc2_right(x))
        x = torch.add(left, right)
        x = self.fc3(x)
        return x


def main():
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Using {device}")

    print(f"Downloading data to {DATA_PATH}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False, transform=transform),
        batch_size=1000,
        shuffle=False,
    )

    model = FCNet().to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)  # type: ignore
        accuracy = 100.0 * correct / len(test_loader.dataset)  # type: ignore
        print(
            f"Epoch {epoch + 1}/{epochs} - Test loss: {test_loss}, Accuracy: {accuracy:.2f}%"
        )

    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        dynamo=True,
    )
    assert onnx_program is not None
    onnx_program.save(
        ONNX_PATH, include_initializers=True, keep_initializers_as_inputs=False
    )
    print(f"Model saved as {ONNX_PATH}")


if __name__ == "__main__":
    main()
