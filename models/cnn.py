from models.model import Model


class CNN(Model):
    # Constructor
    def __init__(self):
        super().__init__("cnn")

    # Methods
    def train(self):
        print("The model has finished training.")

    def validate(self):
        print("The model validated with 98.50%.")

    def test(self):
        print("The model tested with 76.60%.")

    def save(self, path):
        # Export model as some payload

        # Write it to the file path specified
        with open(path, "w") as file:
            file.write("data")

        # Success
        print(f"\nThe model has been saved to {path}.")

    def load(self, path):
        print(f"The model has been loaded from {path}.")
