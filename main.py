from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import PathValidator
from models.cnn import CNN
from datetime import datetime
from os import getcwd


def main():
    """The entry point of the program. Loops until a None is returned."""

    current_menu = main_menu

    while current_menu:
        current_menu = current_menu()


def main_menu():
    print("\n|======| Main Menu |======|")

    print()
    menu = inquirer.select(
        message="What would you like to do?",
        choices=[
            Choice(value=train, name="Train"),
            Choice(value=validate, name="Validate"),
            Choice(value=test, name="Test"),
            Choice(value=exit, name="Exit"),
        ],
        default=None,
    ).execute()

    return menu


def train():
    """Handles the logical flow for training, automatic validation, and model exporting.

    Returns:
        main_menu: Menu for the main loop.
    """

    # Get model to train with
    model = choose_model()
    print()

    # Train model
    model.train()

    # Give model validation
    model.validate()

    # Save model submenu
    print()
    if inquirer.confirm(
        message="Would you like to save this model?", default=True
    ).execute():
        save_model(model)

    return main_menu


def validate():
    """Validates a chosen model.

    Returns:
        main_menu: Menu for the main loop.
    """

    # Get model to train with
    model = choose_model()

    # Load model from disk
    load_model_from_disk(model)

    # Validate model
    model.validate()

    return main_menu


def test():
    """Tests a chosen model.

    Returns:
        main_menu: Menu for the main loop.
    """

    # Get model to train with
    model = choose_model()

    # Load model from disk
    load_model_from_disk(model)

    # Test model
    model.test()

    return main_menu


def save_model(model):
    """Saves a trained model to disk.

    Args:
        model: The trained model to export state from.
    """

    # Make file name
    name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"

    # Ask for file path
    print()
    path = inquirer.filepath(
        message="Enter a directory for the new model:",
        default=f"{getcwd()}/saved_models/{model.get_name()}",
        validate=PathValidator(is_dir=True, message="This is not a valid path."),
        only_directories=True,
    ).execute()

    # Full file path
    file_path = f"{path}/{name}"

    # Save model
    model.save(file_path)


def load_model_from_disk(model):

    # Prompt for model
    print()
    path = inquirer.filepath(
        message="Select the model to validate:",
        default=f"{getcwd()}/saved_models/{model.get_name()}",
        validate=PathValidator(is_file=True, message="This is not a valid file."),
        instruction="(Start by typing '/')",
        only_files=True,
    ).execute()
    print()

    # Load model
    model.load(path)


def choose_model():
    """Chooses a machine learning model."""

    print()
    model = inquirer.select(
        message="Choose a machine learning model:",
        choices=[
            Choice(value=CNN, name="Convolutional Neural Network"),
            Choice(value=CNN, name="Tree"),
            Choice(value=CNN, name="Branch"),
        ],
        default=None,
    ).execute()

    return model()


if __name__ == "__main__":
    main()
