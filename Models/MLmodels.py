# MLmodels.py
# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from data_loader import DataLoader

class DataProcessor:
    """Class to load and preprocess data."""
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.encoder = OneHotEncoder()

    def load_and_preprocess_data(self):
        """Load and preprocess the RAVDESS dataset."""
        data_loader = DataLoader(self.dataset_path)
        mfccs, file_emotion, ravdess_df = data_loader.load_audio_files_and_extract_mfcc()
        data_loader.replace_emotion_labels()

        # Extract features and target
        X = ravdess_df.drop(columns=['Emotions', 'Path']).values  # Drop non-feature columns
        y = ravdess_df['Emotions'].values  # Target variable (Emotions)

        # One-hot encode the target labels
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray() # Convert to 2D array
        return X, y_encoded # Return features and one-hot encoded labels

    def get_categories(self):
        """Get emotion categories for labels."""
        return self.encoder.categories_[0]


class ClassifierTrainer:
    """Class to train and evaluate classifiers using GridSearchCV."""
    def __init__(self, X, y, param_grids, classifiers, test_size=0.2, random_state=42):
        self.X = X # Input features
        self.y = y # Target labels
        self.param_grids = param_grids # Dictionary of hyperparameter grids
        self.classifiers = classifiers # Dictionary of classifiers
        self.test_size = test_size # Test set size
        self.random_state = random_state # Random seed
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data() # Split data
        self.y_train_flat = np.argmax(self.y_train, axis=1) # Convert one-hot encoded labels to integers
        self.y_test_flat = np.argmax(self.y_test, axis=1) # Convert one-hot encoded labels to integers
        self.best_estimators = {}  # Dictionary to store best estimators

    def split_data(self):
        """Split data into training and testing sets."""
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def tune_hyperparameters(self):
        """Perform GridSearchCV for each classifier."""
        for clf_name, clf in self.classifiers.items():
            print(f"\nTuning hyperparameters for {clf_name} using GridSearchCV")
            grid_search = GridSearchCV(
                clf,
                self.param_grids[clf_name],
                cv=5,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train_flat)
            self.best_estimators[clf_name] = grid_search.best_estimator_

            # Save results for each classifier
            self.save_results(
                clf_name,
                grid_search.best_params_,
                grid_search.best_score_,
                None, None, None, None,  # Placeholder for test metrics
                is_final=False
            )

    def evaluate_classifiers(self):
        """Evaluate each classifier on the test set."""
        for clf_name, clf in self.best_estimators.items():
            print(f"\nFinal evaluation of {clf_name} on the test set")
            y_test_pred = clf.predict(self.X_test)

            # Calculate test metrics
            test_accuracy = accuracy_score(self.y_test_flat, y_test_pred)
            test_report = classification_report(self.y_test_flat, y_test_pred, output_dict=True)

            print(f"{clf_name} Test Accuracy: {test_accuracy:.4f}")
            print(classification_report(self.y_test_flat, y_test_pred))

            # Save the test results to a file
            self.save_results(
                clf_name,
                None,  # No need to repeat hyperparameters
                None,  # No need to repeat CV score
                test_accuracy,
                test_report['weighted avg']['precision'],
                test_report['weighted avg']['recall'],
                test_report['weighted avg']['f1-score'],
                is_final=True
            )

    def save_results(self, clf_name, best_params, cv_score, test_accuracy, test_precision, test_recall, test_f1, is_final):
        """Save results for a given classifier to a file."""
        file_name = f"results_{clf_name}.txt"
        with open(file_name, 'a') as f:
            if not is_final:
                # Save hyperparameter tuning results
                f.write(f"Hyperparameter Tuning Results for {clf_name}\n")
                f.write(f"Best Parameters: {best_params}\n")
                f.write(f"Best Cross-Validated Accuracy: {cv_score:.4f}\n\n")
            else:
                # Save test evaluation results
                f.write(f"Final Test Results for {clf_name}\n")
                f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                f.write(f"Test Precision: {test_precision:.4f}\n")
                f.write(f"Test Recall: {test_recall:.4f}\n")
                f.write(f"Test F1-Score: {test_f1:.4f}\n\n")
        print(f"Results for {clf_name} saved to {file_name}")


if __name__ == "__main__":
    Ravdess = r'./RAVDESS'   # Relative Path to the dataset
    processor = DataProcessor(Ravdess) # Initialize the DataProcessor with the dataset path
    X, y = processor.load_data()  # Load and preprocess the dataset to extract features and labels


    # Define hyperparameter grids
    param_grids = {
        "RandomForest": {
            'n_estimators': [500, 700, 1500],
            'max_depth': [50, 100, None],
            'min_samples_split': [2, 5, 10]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01, 'scale'],
            'kernel': ['rbf', 'linear']
        }
    }

    # Define classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Train and evaluate classifiers
    trainer = ClassifierTrainer(X, y, param_grids, classifiers)
    trainer.tune_hyperparameters() 
    trainer.evaluate_classifiers() 
