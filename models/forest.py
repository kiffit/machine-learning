from models.model import Model


"""
Notes for you, Jung:
- The docstrings explain what each function will need to do.
- Use the getters/setters to save and load stuff. You can find what options there are in model.py
- Saving and loading here aren't the biggest deal, but it probably isn't tough to make work. Just make sure you can train and validate the model for now.
"""

class Forest(Model):
    def __init__(self, name):
        self.set_name(name)

        # These will eventually be done
        # self.set_training_set(training_set)
        # self.set_validation_set(validation_set)

    # Methods
    def train(self):
        """Trains the machine learning model and stores it in self.__model."""
        pass

    def validate(self):
        """Evaluates the machine learning model stored in self.__model. with self.__validation_set."""
        pass

    def save(self, path):
        """Saves the learned model to a specified file path. Should be loadable via load(name).

        Args:
            path (str): The fully qualified file path/name (can be absolute or relative) where the file should be saved.
        """
        pass

    def load(self, path):
        """Loads the learned model from a specified file path.

        Args:
            path (str): The fully qualified file path/name (can be absolute or relative) where the file is saved.
        """
        pass
