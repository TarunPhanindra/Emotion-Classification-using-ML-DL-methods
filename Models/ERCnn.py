#ERCnn.py
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

        # Reshape features for CNN input
        X = mfccs.reshape(-1, 13, 228, 1)
        y = ravdess_df['Emotions'].values

        # Encode target labels
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        return X, y_encoded

    def get_categories(self):
        """Get the categories for the confusion matrix."""
        return self.encoder.categories_[0]


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
        model.add(BatchNormalization()) # Batch normalization
        model.add(MaxPooling2D((2, 2))) # Down sampling the spatial dimensions by a factor of 2
        model.add(Dropout(0.3))   # Dropout with a rate of 0.3 to reduce overfitting by randomly setting 30% of neurons to zero.

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) # Conv2D: Applies 64 filters of size 3x3 to the input from the previous layer
        model.add(BatchNormalization()) # Batch normalization
        model.add(MaxPooling2D((2, 2))) # Down sampling the spatial dimensions by a factor of 2
        model.add(Dropout(0.4))  # Dropout with a rate of 0.4 to further reduce overfitting by setting 40% of neurons to zero.

        model.add(Flatten()) # Flatten the 2D feature maps into a 1D feature vector
        model.add(Dense(256, activation='relu')) #Dense layer with 256 neurons
        model.add(Dropout(0.3)) # Dropout with a rate of 0.3 to reduce overfitting by randomly setting 30% of neurons to zero.
        model.add(Dense(num_classes, activation='softmax')) # Output layer with softmax activation

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model
        return model # Return the compiled model


class Trainer:
    """Class for training and evaluating the model."""

    def __init__(self, X, y, X_test, y_test, batch_size, learning_rate, epochs, k_folds=5, results_file="results_ERCnn_64.txt"):
        self.X = X # Input features
        self.y = y # Target labels
        self.X_test = X_test # Test input features
        self.y_test = y_test # Test target labels
        self.batch_size = batch_size # Batch size
        self.learning_rate = learning_rate # Learning rate
        self.epochs = epochs # Number of epochs
        self.k_folds = k_folds # Number of folds
        self.results_file = results_file # File to store evaluation results
        self.num_classes = y.shape[1] # Number of classes in the target labels
        self.input_shape = X.shape[1:] # Shape of the input features

    def train_and_evaluate(self):
        """Train and evaluate the model using K-Fold Cross Validation."""
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42) # K-Fold Cross Validation
        
        # Lists to store metrics for each fold
        fold_accuracies = []
        train_accuracies, val_accuracies = [], []
        best_fold_index, best_val_accuracy, best_val_conf_matrix = -1, 0, None
        # Iterate over each fold
        for fold, (train_index, val_index) in enumerate(kf.split(self.X)):
            print(f"\nTraining fold {fold + 1}/{self.k_folds}")
            X_train, X_val = self.X[train_index], self.X[val_index] # Split the data into training and validation sets
            y_train, y_val = self.y[train_index], self.y[val_index] # Split the labels into training and validation sets

            # Initialize the model
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

            # Track the best fold based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_fold_index = fold
                y_val_pred = model.predict(X_val)
                y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                y_val_true_classes = np.argmax(y_val, axis=1)
                best_val_conf_matrix = confusion_matrix(y_val_true_classes, y_val_pred_classes)

        # Evaluate on the test set
        y_test_pred = model.predict(self.X_test)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_test_true_classes = np.argmax(self.y_test, axis=1)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
        test_precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        test_recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
        # Calculate confusion matrix for test set
        test_conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)

        # Print metrics
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")

        self.save_results(fold_accuracies, best_val_accuracy, best_fold_index, test_accuracy, test_precision,
                          test_recall, test_f1)
        self.plot_confusion_matrix(best_val_conf_matrix, best_fold_index, "validation")
        self.plot_confusion_matrix(test_conf_matrix, "test", "Test")
        self.plot_training_validation_accuracies(train_accuracies, val_accuracies)

    def save_results(self, fold_accuracies, best_val_accuracy, best_fold_index, test_accuracy, test_precision,
                     test_recall, test_f1):
        """Save metrics and results to a file."""
        avg_accuracy = np.mean(fold_accuracies) # Calculate average accuracy across all folds
        # Open the results file in write mode to save the metrics
        with open(self.results_file, 'w') as f:
            f.write(f"K-Fold Cross-Validation Results\n")
            f.write(f"Average Validation Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Best Validation Accuracy: {best_val_accuracy:.4f} (Fold {best_fold_index + 1})\n\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")

            for i, acc in enumerate(fold_accuracies, 1):
                f.write(f"Fold {i}: Validation Accuracy={acc:.4f}\n")

        print(f"\nResults saved to {self.results_file}") 

    def plot_confusion_matrix(self, conf_matrix, index, dataset_type):
        """Plot and save the confusion matrix for the best fold or test set."""
        plt.figure(figsize=(10, 8)) # Create a figure to display the confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=processor.get_categories(),
                    yticklabels=processor.get_categories()) # Use seaborn's heatmap to visualize the confusion matrix with labels
        plt.title(f"Confusion Matrix for {dataset_type.capitalize()} Fold {index}")
        plt.xlabel("Predicted") 
        plt.ylabel("True") 
        plt.savefig(f"confusion_matrix_{dataset_type}_ERCnn_64_{index}.png") # Save the plot as a PNG file with a descriptive name
        plt.show() 

    def plot_training_validation_accuracies(self, train_accuracies, val_accuracies):
        """Plot training vs validation accuracies."""
        plt.figure(figsize=(12, 8))
        # Iterate through each fold and plot training and validation accuracies
        for fold in range(len(train_accuracies)):
            plt.plot(train_accuracies[fold], label=f"Train Accuracy Fold {fold + 1}")
            plt.plot(val_accuracies[fold], label=f"Validation Accuracy Fold {fold + 1}")

        plt.title("Training vs Validation Accuracies")
        plt.xlabel("Epochs") # X-axis represents the number of epochs
        plt.ylabel("Accuracy") # Y-axis represents the accuracy
        plt.legend(loc="lower right")  # Display the legend in the lower right corner
        plt.grid(True) # Enable grid lines
        plt.savefig("training_vs_validation_accuracies_ERCnn_64.png") # Save the plot as a PNG file
        plt.show()


# Instantiate and execute
if __name__ == "__main__":
    Ravdess = r'./RAVDESS'   # Relative Path to the dataset
    processor = DataProcessor(Ravdess) # Initialize the DataProcessor with the dataset path
    X, y = processor.load_data()  # Load and preprocess the dataset to extract features and labels

    # Split into training/validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Initialize the Trainer class to train and evaluate the model
    trainer = Trainer(X_train_val, y_train_val, X_test, y_test, batch_size=64, learning_rate=0.001, epochs=50)
    trainer.train_and_evaluate() # Train and evaluate the model
