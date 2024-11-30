#Impoerting necessary libraries
import os
import numpy as np
import pandas as pd
import librosa

# Define a class to load audio data, extract MFCC features, and preprocess the RAVDESS dataset.
class DataLoader:
    # Initializes the DataLoader class with the dataset path and configuration parameters.
    def __init__(self, dataset_path, n_mfcc=13, max_len=228):
        self.dataset_path = dataset_path
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.file_emotion = []   # To store emotion labels
        self.file_path = []      # To store file paths of audio files
        self.mfccs = []          # To store extracted MFCC features
        self.ravdess_df = None   # DataFrame to store the processed dataset

    def load_audio_files_and_extract_mfcc(self):
        main_folder = os.path.join(self.dataset_path, 'audio_speech_actors_01-24') # Path to the main folder containing audio files
        
        for subfolder in os.listdir(main_folder):     # Iterate through subfolders for each actor
            subfolder_path = os.path.join(main_folder, subfolder)

            if os.path.isdir(subfolder_path):      # Check if the subfolder is a directory
                for file in os.listdir(subfolder_path):   # Iterate through files in the subfolder
                    if file.endswith('.wav'):         # Check if the file is a WAV audio file
                        emotion = int(file.split('-')[2])  # Extract emotion label from the filename
                        self.file_emotion.append(emotion)
                        
                        # Construct the full path to the file
                        file_path = os.path.join(subfolder_path, file)
                        self.file_path.append(file_path)
                        
                        # Load the audio file with librosa
                        audio, sr = librosa.load(file_path, sr=22050)  # Default sampling rate of 22050 Hz
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)  # Extract MFCC features
                        mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant') # Pad MFCC features to a fixed length (max_len) for consistency
                        self.mfccs.append(mfcc)  # Append the processed MFCC features to the list

        # Convert lists to NumPy arrays for efficient processing
        self.mfccs = np.array(self.mfccs)
        self.file_emotion = np.array(self.file_emotion)
        self.file_path = np.array(self.file_path)
        
        # Create a DataFrame to combine emotion labels, file paths, and MFCC features
        emotion_df = pd.DataFrame(self.file_emotion, columns=['Emotions'])
        path_df = pd.DataFrame(self.file_path, columns=['Path'])
        mfccs_df = pd.DataFrame(self.mfccs.reshape(self.mfccs.shape[0], -1))
        self.ravdess_df = pd.concat([emotion_df, path_df, mfccs_df], axis=1)
        
        # Return extracted features, emotion labels, and the combined DataFrame
        return self.mfccs, self.file_emotion, self.ravdess_df

    def replace_emotion_labels(self):
        if self.ravdess_df is not None:
            # Map numerical labels to descriptive emotion labels
            self.ravdess_df.Emotions.replace(
                {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, 
                inplace=True
            )
