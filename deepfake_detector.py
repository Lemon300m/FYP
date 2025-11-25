import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
from datetime import datetime

class DeepfakeDetector:
    def __init__(self, real_dataset_path, fake_dataset_path):
        self.real_dataset = real_dataset_path
        self.deepfake_dataset = fake_dataset_path
        self.model = None
        self.model_path = "deepfake_model.pkl"
        self.log_path = "training_log.json"
        
    def __init__(self, real_dataset_path, fake_dataset_path, progress_callback=None):
        self.real_dataset = real_dataset_path
        self.deepfake_dataset = fake_dataset_path
        self.model = None
        self.model_path = "deepfake_model.pkl"
        self.log_path = "training_log.json"
        self.progress_callback = progress_callback
        
    def _report_progress(self, message, percentage=None):
        """Report progress to callback if available"""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        else:
            if percentage is not None:
                print(f"{message} - {percentage:.1f}%")
            else:
                print(message)
    
    def extract_features(self, img_path):
        """Extract features from image to detect artifacts"""
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Resize for consistency
        img = cv2.resize(img, (128, 128))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract various features that might reveal artifacts
        features = []
        
        # Color histogram features
        features.extend(cv2.calcHist([img], [0], None, [8], [0, 256]).flatten())
        features.extend(cv2.calcHist([img], [1], None, [8], [0, 256]).flatten())
        features.extend(cv2.calcHist([img], [2], None, [8], [0, 256]).flatten())
        
        # Texture features using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([laplacian.mean(), laplacian.std(), laplacian.var()])
        
        # Edge detection features
        edges = cv2.Canny(gray, 100, 200)
        features.extend([edges.mean(), edges.std()])
        
        # Frequency domain features (DCT)
        dct = cv2.dct(np.float32(gray))
        features.extend([dct.mean(), dct.std(), dct.var()])
        
        return np.array(features)
    
    def load_dataset(self, path, label):
        """Load images from dataset path"""
        X, y = [], []
        
        if not os.path.exists(path):
            self._report_progress(f"Warning: Path {path} does not exist")
            return X, y
        
        files = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png', '.bmp'))]
        total_files = len(files)
        
        for idx, file in enumerate(files):
            file_path = os.path.join(path, file)
            
            # Report progress
            progress = (idx + 1) / total_files * 100
            self._report_progress(f"Loading {file}", progress)
            
            # Handle video files - pick random frame
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    random_frame = np.random.randint(0, total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                    ret, frame = cap.read()
                    if ret:
                        temp_path = "temp_frame.jpg"
                        cv2.imwrite(temp_path, frame)
                        features = self.extract_features(temp_path)
                        os.remove(temp_path)
                        if features is not None:
                            X.append(features)
                            y.append(label)
                cap.release()
            
            # Handle image files
            elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                features = self.extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
        
        return X, y
    
    def train(self):
        """Train machine learning model on labelled data"""
        self._report_progress("Starting training process...")
        
        # Load real and fake datasets
        self._report_progress("Loading real dataset...", 0)
        X_real, y_real = self.load_dataset(self.real_dataset, 0)  # 0 = real
        
        self._report_progress("Loading fake dataset...", 33)
        X_fake, y_fake = self.load_dataset(self.deepfake_dataset, 1)  # 1 = fake
        
        if len(X_real) == 0 or len(X_fake) == 0:
            self._report_progress("Error: Datasets not properly loaded")
            return None
        
        # Combine datasets
        self._report_progress("Combining datasets...", 66)
        X = np.array(X_real + X_fake)
        y = np.array(y_real + y_fake)
        
        self._report_progress(f"Total samples: {len(X)} (Real: {len(X_real)}, Fake: {len(X_fake)})", 70)
        
        # Split to 80-20
        self._report_progress("Splitting data to 80-20...", 75)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self._report_progress("Training model (this may take a while)...", 80)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
        self.model.fit(X_train, y_train)
        
        # Test on 20% data
        self._report_progress("Testing model on validation set...", 90)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self._report_progress(f"Model accuracy: {accuracy:.4f}", 95)
        
        # Save model
        self._report_progress("Saving model...", 98)
        joblib.dump(self.model, self.model_path)
        self._report_progress(f"Model saved to {self.model_path}", 100)
        
        # Log result
        self._log_training(accuracy, len(X_train), len(X_test))
        
        return accuracy
    
    def maintain(self, new_data_path, label):
        """Maintain model with new data"""
        self._report_progress("Running maintenance...", 0)
        
        # Check if data is properly labelled
        if label not in [0, 1]:
            self._report_progress("Error: Data not properly labelled as real (0) or fake (1)")
            return False
        
        # Load existing model
        self._report_progress("Loading existing model...", 10)
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self._report_progress("Error: No existing model found. Run train() first.")
            return False
        
        # Load new data
        self._report_progress("Loading new data...", 20)
        X_new, y_new = self.load_dataset(new_data_path, label)
        
        if len(X_new) == 0:
            self._report_progress("Error: No valid data found")
            return False
        
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        # Test random set with old model
        self._report_progress("Testing with old model...", 60)
        indices = np.random.choice(len(X_new), min(len(X_new), 50), replace=False)
        X_test = X_new[indices]
        y_test = y_new[indices]
        
        old_pred = self.model.predict(X_test)
        old_accuracy = accuracy_score(y_test, old_pred)
        self._report_progress(f"Old model accuracy on new data: {old_accuracy:.4f}", 70)
        
        # Fine-tune with new data
        self._report_progress("Fine-tuning model with new data...", 75)
        self.model.fit(X_new, y_new)
        
        # Test again with another random set
        self._report_progress("Testing with fine-tuned model...", 85)
        indices2 = np.random.choice(len(X_new), min(len(X_new), 50), replace=False)
        X_test2 = X_new[indices2]
        y_test2 = y_new[indices2]
        
        new_pred = self.model.predict(X_test2)
        new_accuracy = accuracy_score(y_test2, new_pred)
        self._report_progress(f"New model accuracy on new data: {new_accuracy:.4f}", 90)
        
        # If new result better, replace old model
        if new_accuracy > old_accuracy:
            self._report_progress("Saving updated model...", 95)
            joblib.dump(self.model, self.model_path)
            self._report_progress("Model updated!", 100)
            self._log_maintenance(old_accuracy, new_accuracy, True)
            return True
        else:
            # Reload old model
            self._report_progress("Old model retained (better performance)", 95)
            self.model = joblib.load(self.model_path)
            self._log_maintenance(old_accuracy, new_accuracy, False)
            return False
    
    def identify(self, image_path):
        """Check if provided image is real or deepfaked"""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                print("Error: No model found. Train a model first.")
                return None
        
        features = self.extract_features(image_path)
        if features is None:
            print("Error: Could not process image")
            return None
        
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = "DEEPFAKE" if prediction == 1 else "REAL"
        confidence = probability[prediction] * 100
        
        return {"result": result, "confidence": confidence, "prediction": int(prediction)}
    
    def _log_training(self, accuracy, train_size, test_size):
        """Log training results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "training",
            "accuracy": accuracy,
            "train_size": train_size,
            "test_size": test_size
        }
        self._append_log(log_entry)
    
    def _log_maintenance(self, old_acc, new_acc, updated):
        """Log maintenance results"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "maintenance",
            "old_accuracy": old_acc,
            "new_accuracy": new_acc,
            "model_updated": updated
        }
        self._append_log(log_entry)
    
    def _append_log(self, entry):
        """Append entry to log file"""
        logs = []
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                logs = json.load(f)
        
        logs.append(entry)
        
        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Set your dataset paths here
    real_path = "./dataset/real"
    fake_path = "./dataset/fake"
    
    detector = DeepfakeDetector(real_path, fake_path)
    
    # Train initial model
    # detector.train()
    
    # Identify an image
    # result = detector.identify("test_image.jpg")
    # if result:
    #     print(f"Result: {result['result']} (Confidence: {result['confidence']:.2f}%)")
    
    # Maintain with new data
    # detector.maintain("./dataset/new_fake", label=1)