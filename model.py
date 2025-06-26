import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class ECGDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self):
        """Load and preprocess all datasets"""
        print("Loading and preprocessing all ECG datasets...")
        
        # Load MIT-BIH datasets
        mitbih_train = self._load_mitbih('mitbih_train.csv')
        mitbih_test = self._load_mitbih('mitbih_test.csv')
        
        # Load PTBDB datasets
        ptbdb_abnormal = self._load_ptbdb('ptbdb_abnormal.csv')
        ptbdb_normal = self._load_ptbdb('ptbdb_normal.csv')
        
        # Combine all datasets
        X = np.vstack([mitbih_train['data'], mitbih_test['data'], 
                      ptbdb_abnormal['data'], ptbdb_normal['data']])
        y = np.concatenate([mitbih_train['labels'], mitbih_test['labels'],
                           ptbdb_abnormal['labels'], ptbdb_normal['labels']])
        
        # Encode labels (MIT-BIH: 0-4, PTBDB: normal=0, abnormal=1)
        y = self.label_encoder.fit_transform(y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Normalize data
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Reshape for CNN (samples, timesteps, channels)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test
    
    def _load_mitbih(self, filename):
        """Load MIT-BIH dataset"""
        filepath = os.path.join(self.data_path, filename)
        df = pd.read_csv(filepath, header=None)
        
        # Last column is the label
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        
        return {'data': data, 'labels': labels}
    
    def _load_ptbdb(self, filename):
        """Load PTBDB dataset"""
        filepath = os.path.join(self.data_path, filename)
        df = pd.read_csv(filepath, header=None)
        
        # Last column is the label (normal/abnormal)
        data = df.iloc[:, :-1].values
        
        # Create labels (filename contains the class)
        if 'abnormal' in filename:
            labels = np.ones(len(data))  # Abnormal = 1
        else:
            labels = np.zeros(len(data))  # Normal = 0
            
        return {'data': data, 'labels': labels}
    
    def get_class_names(self):
        """Get the meaning of encoded class labels"""
        return self.label_encoder.classes_
    
    def save_preprocessor(self, save_path):
        """Save the fitted preprocessor"""
        import joblib
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_path, 'ecg_scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_path, 'label_encoder.pkl'))
    
    @classmethod
    def load_preprocessor(cls, load_path):
        """Load saved preprocessor"""
        import joblib
        preprocessor = cls(None)
        preprocessor.scaler = joblib.load(os.path.join(load_path, 'ecg_scaler.pkl'))
        preprocessor.label_encoder = joblib.load(os.path.join(load_path, 'label_encoder.pkl'))
        return preprocessor


# Example usage
if __name__ == "__main__":
    data_path = r"C:\Users\HP\Desktop\upworkhealt\data"
    preprocessor = ECGDataPreprocessor(data_path)
    
    # Process all datasets
    X_train, X_test, y_train, y_test = preprocessor.load_and_preprocess()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print("Class names:", preprocessor.get_class_names())
    
    # Save the preprocessor for later use
    preprocessor.save_preprocessor('saved_preprocessors')