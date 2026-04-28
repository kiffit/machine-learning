from models.model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


class Forest(Model):
    def __init__(self, name):
        super().__init__(name)

    # Methods
    def train(self):
        """Trains the machine learning model and stores it in self.__model."""

        # Get training set
        training_set = self.get_training_set()

        # Check if training set exists
        if training_set is None:
            print("No training set found.")
            return

        # Split training set into x and y
        x_train = training_set[
            [
                "Age",
                "Gender",
                "Study_Hours_per_Day",
                "Attendance_Rate",
                "Travel_Time_Minutes",
                "Part_Time_Job",
                "Scholarship",
                "Stress_Index",
                "GPA",
                "Semester",
            ]
        ]

        y_train = training_set["Dropout"]

        # Create random forest classifier
        forest = RandomForestClassifier(
            n_estimators=500,
            random_state=42
        )

        # Train model
        print("Training forest...")
        forest.fit(x_train, y_train)

        # Store model
        self.set_model(forest)

        print("Forest training complete.")

    def validate(self):
        """Evaluates the machine learning model stored in self.__model with self.__validation_set."""

        # Get model and validation set
        forest = self.get_model()
        validation_set = self.get_validation_set()

        # Check if model exists
        if forest is None:
            print("No model found. Train or load a model first.")
            return None

        # Check if validation set exists
        if validation_set is None:
            print("No validation set found.")
            return None

        # Split validation set into x and y
        x_validation = validation_set[
            [
                "Age",
                "Gender",
                "Study_Hours_per_Day",
                "Attendance_Rate",
                "Travel_Time_Minutes",
                "Part_Time_Job",
                "Scholarship",
                "Stress_Index",
                "GPA",
                "Semester",
            ]
        ]

        y_validation = validation_set["Dropout"]

        # Predict validation set
        print("Running forest predictions...")
        y_prediction = forest.predict(x_validation)

        # Calculate accuracy
        accuracy = accuracy_score(y_validation, y_prediction)

        # Print and return result
        print(f"The forest achieved an accuracy of {round(accuracy, 5)}")

        return accuracy

    def save(self, path):
        """Saves the learned model to a specified file path.

        Args:
            path (str): The fully qualified file path/name where the file should be saved.
        """

        # Get model
        forest = self.get_model()

        # Check if model exists
        if forest is None:
            print("No model found. Train or load a model first.")
            return

        # Save model
        joblib.dump(forest, path)

        print(f"Forest model saved to {path}")

    def load(self, path):
        """Loads the learned model from a specified file path.

        Args:
            path (str): The fully qualified file path/name where the file is saved.
        """

        # Load model
        forest = joblib.load(path)

        # Store model
        self.set_model(forest)

        print(f"Forest model loaded from {path}")