#Aug_ERCnn.py
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from data_loader import DataLoader
import seaborn as sns


class DataProcessor:
    """Class for loading and processing data."""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.encoder = OneHotEncoder()

    def load_data(self):
        """Load and preprocess the dataset."""
        data_loader = DataLoader(self.dataset_path)
        mfccs, _, ravdess_df = data_loader.load_audio_files_and_extract_mfcc()
        data_loader.replace_emotion_labels()

        # Reshaping the features for CNN input
        X = mfccs.reshape(-1, 13, 228, 1)
        y = ravdess_df['Emotions'].values

        # Encode the target labels
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        return X, y_encoded

    def get_categories(self):
        """Get the categories for the confusion matrix."""
        return self.encoder.categories_[0]


class DataAugmentation:
    """Class for applying data augmentation techniques."""

    def augment_data(self, X, y):
        """Apply augmentation by adding noise and time-shifting."""
        X_augmented = []
        y_augmented = []

        for i in range(X.shape[0]):
            # Original sample
            X_augmented.append(X[i])
            y_augmented.append(y[i])

            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, X[i].shape)
            X_augmented.append(X[i] + noise)
            y_augmented.append(y[i])

            # Time-shift
            shifted = np.roll(X[i], shift=np.random.randint(1, 5), axis=1)
            X_augmented.append(shifted)
            y_augmented.append(y[i])

        return np.array(X_augmented), np.array(y_augmented)


class CNNModel:
    """Class to define the CNN model."""

    def __init__(self, input_shape, num_classes, learning_rate):
        self.model = self.create_model(input_shape, num_classes, learning_rate)
    #input_shape: shape of the input data
    #num_classes: number of output classes
    def create_model(self, input_shape, num_classes, learning_rate):
        """Create and compile the CNN model."""
        model = Sequential() #Initializing the sequence model
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        # Conv2D: Applies 32 filters of size 3x3 to the input.
        # Activation: 'relu' introduces non-linearity.
        # Padding: 'same' it ensures that the output size remains the same as the input.
        model.add(BatchNormalization())  # Batch normalization
        model.add(MaxPooling2D((2, 2)))  # Down sampling the spatial dimensions by a factor of 2
        model.add(Dropout(0.3))   # Dropout with a rate of 0.3 to reduce overfitting by randomly setting 30% of neurons to zero.

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # Conv2D: Applies 64 filters of size 3x3 to the input from the previous layer
        model.add(BatchNormalization())
        model.add(MaxPooling2D((1, 2))) 
        model.add(Dropout(0.4))  # Dropout with a rate of 0.4 to further reduce overfitting by setting 40% of neurons to zero.

        model.add(Flatten())  # Flatten the 2D feature maps into a 1D feature vector
        model.add(Dense(256, activation='relu')) #Dense layer with 256 neurons
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) 
        return model  # Return the compiled model

#Class for training and evaluating the model
class Trainer: 
    """ 
    Parameters:
    - X: Feature matrix for training and validation(mfcc features).
    - y: Target labels (one-hot encoded) for training and validation.
    - X_test: Feature matrix for testing.
    - y_test: Target labels (one-hot encoded) for testing.
    - batch_size: Size of the mini-batches used during training(16, 32, 64).
    - learning_rate: Learning rate for the optimizer.
    - epochs: Number of training iterations over the entire dataset.
    - k_folds: Number of folds for K-Fold Cross-Validation.
    - results_file: File name to save the results of training and evaluation."""
    def __init__(self, X, y, X_test, y_test, batch_size, learning_rate, epochs, k_folds=5, results_file="results_aug_ERCnn_32.txt"):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.k_folds = k_folds
        self.results_file = results_file
        self.num_classes = y.shape[1] # Number of classes in the target labels (e.g., 8 emotions for RAVDESS dataset).
        self.input_shape = X.shape[1:]  # Shape of the input features for the model (e.g., (13, 228, 1) for MFCC features).

    def train_and_evaluate(self):
        """Train and evaluate the model using K-Fold Cross Validation."""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42) # Initialize K-Fold Cross Validation with 5 folds

        # Lists to store accuracies and per-fold
        fold_accuracies = []
        train_accuracies, val_accuracies = [], []
        
        # Variables to track the best fold based on validation accuracy
        best_fold_index, best_val_accuracy, best_val_conf_matrix = -1, 0, None
        
        # Iterating over each fold in the K-Fold Cross Validation
        for fold, (train_index, val_index) in enumerate(kf.split(self.X)):
            print(f"\nTraining fold {fold + 1}/{self.k_folds}")
            
            # Split the data into training and validation sets for the current fold
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            # Initialize the CNN model
            model = CNNModel(self.input_shape, self.num_classes, self.learning_rate).model

            # Train the model and record the training history
            history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                validation_data=(X_val, y_val), verbose=0)

            # Append training and validation accuracy trends for this fold
            train_accuracies.append(history.history['accuracy'])
            val_accuracies.append(history.history['val_accuracy'])

            # Evaluate the model on the validation set
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_accuracies.append(val_accuracy)

            print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")

            # Updating with the best fold based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_fold_index = fold
                y_val_pred = model.predict(X_val)
                y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                y_val_true_classes = np.argmax(y_val, axis=1)
                best_val_conf_matrix = confusion_matrix(y_val_true_classes, y_val_pred_classes) # Compute the confusion matrix for the best validation fold

        # Evaluate the final model on the test set
        y_test_pred = model.predict(self.X_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(self.y_test, axis=1)

        # Calculating the metrics for the test set
        test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
        test_precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        test_recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        
        # Compute the confusion matrix for the test set
        test_conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)

        # Display the test metrics
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")

        self.save_results(fold_accuracies, best_val_accuracy, best_fold_index, test_accuracy, test_precision,  # Save results (validation and test metrics)
                          test_recall, test_f1)
        self.plot_confusion_matrix(best_val_conf_matrix, best_fold_index, "validation") # Plot and save the confusion matrix for the best validation fold
        self.plot_confusion_matrix(test_conf_matrix, "test", "Test")  # Plot and save the confusion matrix for the test set
        self.plot_training_validation_accuracies(train_accuracies, val_accuracies)  # Plot training vs.validation accuracies across all folds

    def save_results(self, fold_accuracies, best_val_accuracy, best_fold_index, test_accuracy, test_precision,
                     test_recall, test_f1):
         """
    Save metrics and results to a file.

    Parameters:
    - fold_accuracies: List of validation accuracies for each fold in K-Fold Cross-Validation.
    - best_val_accuracy: The highest validation accuracy achieved across all folds.
    - best_fold_index: The index of the fold with the highest validation accuracy.
    - test_accuracy: The accuracy of the final model evaluated on the test set.
    - test_precision: The precision of the final model on the test set.
    - test_recall: The recall of the final model on the test set.
    - test_f1: The F1-Score of the final model on the test set.
    """
         avg_accuracy = np.mean(fold_accuracies)  # Calculate the average validation accuracy across all folds

        # Open the results file in write mode to save the metrics
         with open(self.results_file, 'w') as f:
            f.write(f"K-Fold Cross-Validation Results\n")
            f.write(f"Average Validation Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Best Validation Accuracy: {best_val_accuracy:.4f} (Fold {best_fold_index + 1})\n\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")

            for i, acc in enumerate(fold_accuracies, 1):    # Save the validation accuracy for each fold
                f.write(f"Fold {i}: Validation Accuracy={acc:.4f}\n")

         print(f"\nResults saved to {self.results_file}")

    def plot_confusion_matrix(self, conf_matrix, index, dataset_type):
        """Plot and save the confusion matrix for the best fold or test set."""
        plt.figure(figsize=(10, 8))    # Create a figure to display the confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=processor.get_categories(),   # Use seaborn's heatmap to visualize the confusion matrix with labels
                    yticklabels=processor.get_categories())
        plt.title(f"Confusion Matrix for {dataset_type.capitalize()} Fold {index}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_{dataset_type}_aug_ERCnn_32_{index}.png")  # Save the plot as a PNG file with a descriptive name
        plt.show()

    def plot_training_validation_accuracies(self, train_accuracies, val_accuracies):
        """Plot training vs validation accuracies."""
        plt.figure(figsize=(12, 8))
        # Iterate through each fold and plot training and validation accuracies
        for fold in range(len(train_accuracies)):
            plt.plot(train_accuracies[fold], label=f"Train Accuracy Fold {fold + 1}")  # Plot training accuracy for the current fold
            plt.plot(val_accuracies[fold], label=f"Validation Accuracy Fold {fold + 1}")   # Plot validation accuracy for the current fold

        plt.title("Training vs Validation Accuracies")
        plt.xlabel("Epochs")  # X-axis represents the number of epochs
        plt.ylabel("Accuracy")  # Y-axis represents the accuracy
        
        plt.legend(loc="lower right") # Display the legend in the lower right corner
        plt.grid(True)  # Enable grid lines
        plt.savefig("training_vs_validation_accuracies_aug_ERCnn_32.png")
        plt.show()


# Main execution
if __name__ == "__main__":
    Ravdess = r'./RAVDESS'   # Relative Path to the dataset
    processor = DataProcessor(Ravdess) # Initialize the DataProcessor with the dataset path
    X, y = processor.load_data()  # Load and preprocess the dataset to extract features and labels

    # Apply data augmentation to the features and labels using the DataAugmentation class
    augmenter = DataAugmentation()  # Initialize the DataAugmentation class
    X, y = augmenter.augment_data(X, y)  # Augment the dataset (e.g., noise addition, time-shifting)

    # Split the dataset into training/validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Initialize the Trainer class to train and evaluate the model  
    trainer = Trainer(X_train_val, y_train_val, X_test, y_test, batch_size=32, learning_rate=0.001, epochs=50)
    trainer.train_and_evaluate()  # Train and evaluate the model
