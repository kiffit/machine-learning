from models.model import Model

import torch
import torchvision


# Build the NN
class NNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.model(x)


class NN(Model):
    # Constructor
    def __init__(self):
        super().__init__("nn")
        self.set_model(NNN())

        # Normalize transform
        normalize_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5), std=(0.5)),
            ]
        )

        # Load training dataset
        train_dataset_raw = torchvision.datasets.MNIST(
            root="./MNIST/train",
            train=True,
            transform=normalize_transform,
            download=True,
        )

        # Load validation dataset
        validation_dataset_raw = torchvision.datasets.MNIST(
            root="./MNIST/test",
            train=False,
            transform=normalize_transform,
            download=True,
        )

        # Save to model
        batch_size = 128
        self.set_training_set(
            torch.utils.data.DataLoader(train_dataset_raw, batch_size)
        )
        self.set_validation_set(
            torch.utils.data.DataLoader(validation_dataset_raw, batch_size)
        )

    # Methods
    def train(self):

        # GPU acceleration
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        model = NNN().to(device)

        # Training parameters
        num_epochs = 10
        learning_rate = 0.001
        weight_decay = 0.0001  # 0.001
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Train
        train_loss_list = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}:", end=" ")
            train_loss = 0
            model.train()
            for images, labels in self.get_training_set():
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss_list.append(train_loss / len(self.get_training_set()))
            print(f"Training loss = {train_loss_list[-1]}")

        # Save model to memory
        self.set_model(model)

    def validate(self):

        # Set device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        # Validating
        test_acc = 0
        self.get_model().eval()
        with torch.no_grad():
            for images, labels in self.get_validation_set():
                images = images.to(device)
                y_true = labels.to(device)
                outputs = self.get_model()(images)
                _, y_pred = torch.max(outputs.data, 1)
                test_acc += (y_pred == y_true).sum().item()

        # Print results
        print(
            f"The model validated with {100 * test_acc / len(self.get_validation_set().dataset):.2f}%"
        )

    def save(self, path):

        # Write it to the file path specified
        torch.save(self.get_model().state_dict(), path)

        # Success
        print(f"\nThe model has been saved to {path}.")

    def load(self, path):

        # GPU acceleration
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

        # Load from file path specified
        self.get_model().load_state_dict(torch.load(path, map_location="cpu"))
        self.get_model().to(device)

        # Print results
        print(f"The model has been loaded from {path}.")
