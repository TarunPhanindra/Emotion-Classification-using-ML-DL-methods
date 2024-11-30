# Emotion-Classification-using-ML-DL-methods

## Project Description
This project focuses on classifying emotions from audio files in the [RAVDESS Dataset](https://zenodo.org/record/1188976) by extracting Mel-Frequency Cepstral Coefficients (MFCC) features. It employs both Machine Learning (ML) methods—Random Forest and Support Vector Machine (SVM)—and Deep Learning (DL) methods, including Convolutional Neural Networks (CNN) and CNN-Bidirectional Long Short-Term Memory (CNN-BiLSTM) models with batch sizes of 16, 32, 64. The goal is to identify the most effective model for emotion classification to facilitate further applications.

#File naming convention of the Dataset

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 

Filename identifiers 

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

## Key Features
- Implementation of Random Forest and SVM classifiers for emotion recognition.
- Development of CNN and CNN-BiLSTM models, with and without data augmentation.
- Evaluation of model performance to determine the impact of augmentation techniques.

## Tools Used
- Python
- TensorFlow
- Scikit-Learn
- Librosa

## Purpose/Audience
This project serves as a resource for students and researchers in the fields of Speech and Emotion Recognition, providing insights into various modeling techniques and their effectiveness.

## Installation and Usage Instructions

### Prerequisites
- Python 3.7 or higher
- Required libraries listed in `requirements.txt`

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/TarunPhanindra/Emotion-Classification-using-ML-DL-methods.git

2. **Navigate to the project directory**
   cd Emotion-Classification-using-ML-DL-methods

3. **Install the required Dependencies**
    pip install -r requirements.txt

4. **Executon**
   python "name of the file in Models folder_batchsize".py (Make changes to the batchsize value in the code as required)

5. **Project Structure**
   Emotion-Classification-using-ML-DL-methods/
├── data_loader.py         # Handles loading of audio data and feature extraction
├── Models                 # Contains all the ML/DL models
├── Results                # Directory containing results and evaluation files
│   ├── Confusion Matrix   # Confusion matrix for each model
│   ├── Training&Vaildation Plot     # Plot for training vs validaton accuracies of all the models 
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies

**Input/Output**
Input
 .Dataset: RAVDESS dataset
 .Audio Preprocessing: Extraction of MFCC features from audio files.
Output
 .Metrics: Accuracy, Precision, Recall, F1-Score.
 .Confusion Matrices: Generated for validation and test results.
**Results and Visuals**
Results
 .The CNN-BiLSTM model with data augmentation achieved the highest accuracy and F1-Score among the tested models.
 .Detailed results, including evaluation metrics and confusion matrices, are available in the RESULTS directory.
**Visuals**
 .Confusion Matrices: Saved as PNG files in the respective model directories within RESULTS folder
 .Training vs. Validation Accuracy Plots: Available in the RESULTS directory, illustrating model performance over epochs.
**Contribution Guidelines**

Contributions are welcome. To contribute:

 1. Fork the repository.
 2. Create a new branch for your feature or bug fix:
    git checkout -b feature-name
 3. Commit your changes with descriptive messages:
    git commit -m "Description of changes"


This README provides a comprehensive overview of your project, including setup instructions, project structure, and contribution guidelines.
::contentReference[oaicite:0]{index=0}
 


  
