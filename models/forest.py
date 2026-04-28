from models.model import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torchvision
import joblib
import numpy as np


class Forest(Model):
    # Constructor
    def __init__(self):
        super().__init__("forest")

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
        self.set_training_set(train_dataset_raw)
        self.set_validation_set(validation_dataset_raw)

        # Create random forest model
        self.set_model(
            RandomForestClassifier(
                #Change the size 
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
        )

    def dataset_to_numpy(self, dataset):
        """Converts MNIST dataset into x and y numpy arrays."""

        x = []
        y = []

        for image, label in dataset:
            # image shape is [1, 28, 28]
            # flatten it into 784 features
            x.append(image.numpy().flatten())
            y.append(label)

        return np.array(x), np.array(y)

    # Methods
    def train(self):
        """Trains the machine learning model and stores it in self.__model."""

        print("Converting MNIST training data...")
        x_train, y_train = self.dataset_to_numpy(self.get_training_set())

        print("Training forest...")
        self.get_model().fit(x_train, y_train)

        print("Forest training complete.")

    def validate(self):
        """Evaluates the machine learning model stored in self.__model with self.__validation_set."""

        print("Converting MNIST validation data...")
        x_validation, y_validation = self.dataset_to_numpy(self.get_validation_set())

        print("Running forest predictions...")
        y_prediction = self.get_model().predict(x_validation)

        accuracy = accuracy_score(y_validation, y_prediction)

        print(f"The model validated with {100 * accuracy:.2f}%")

    def save(self, path):
        """Saves the learned model to a specified file path."""

        joblib.dump(self.get_model(), path)

        print(f"\nThe model has been saved to {path}.")

    def load(self, path):
        """Loads the learned model from a specified file path."""

        self.set_model(joblib.load(path))

        print(f"The model has been loaded from {path}.")