from abc import ABC, abstractmethod


class Model(ABC):
    # Attributes
    __name = None
    __training_set = None
    __validation_set = None
    __testing_set = None
    __model = None

    # Constructor
    def __init__(self, name):
        self.set_name(name)

        # These will always be the same
        # self.set_training_set(training_set)
        # self.set_validation_set(validation_set)
        # self.set_testing_set(testing_set)

    # Methods
    @abstractmethod
    def train(self):
        """Trains the machine learning model and stores it in self.__model."""
        pass

    @abstractmethod
    def validate(self):
        """Evaluates the machine learning model stored in self.__model. with self.__validation_set."""
        pass

    @abstractmethod
    def test(self):
        """Evaluates the machine learning model stored in self.__model with self.__testing_set."""
        pass

    @abstractmethod
    def save(self, path):
        """Saves the learned model to a specified file path. Should be loadable via load(name).

        Args:
            path (str): The fully qualified file path/name (can be absolute or relative) where the file should be saved.
        """
        pass

    @abstractmethod
    def load(self, path):
        """Loads the learned model from a specified file path.

        Args:
            path (str): The fully qualified file path/name (can be absolute or relative) where the file is saved.
        """
        pass

    # Getters
    def get_name(self):
        return self.__name

    def get_training_set(self):
        return self.__training_set

    def get_validation_set(self):
        return self.__validation_set

    def get_model(self):
        return self.__model

    # Setters
    def set_name(self, name):
        self.__name = name

    def set_training_set(self, training_set):
        self.__training_set = training_set

    def set_validation_set(self, validation_set):
        self.__validation_set = validation_set

    def set_testing_set(self, testing_set):
        self.__testing_set = testing_set

    def set_model(self, model):
        self.__model = model

    # To string
    def __str__(self):
        return f"Model ({self.get_name()})"
