import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
import json

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Deepfake Detection System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Model and detection variables
        self.model = None
        self.model_path = "deepfake_model.pkl"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Camera variables
        self.camera = None
        self.is_scanning = False
        self.current_frame = None
        self.detection_interval = 1.0  # Check every 1 second
        self.last_detection_time = 0
        
        # Training variables
        self.real_dataset_path = ""
        self.fake_dataset_path = ""
        
        self.setup_ui()
        self.load_model_if_exists()
        
    def setup_ui(self):
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-time Deepfake Detection System", 
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # LEFT COLUMN - Camera Feed
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        # Camera controls
        camera_controls = ttk.Frame(left_frame)
        camera_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(camera_controls, text="▶ Start Scanning", 
                                       command=self.start_scanning, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(camera_controls, text="⏹ Stop Scanning", 
                                      command=self.stop_scanning, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(camera_controls, text="Camera:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_controls, textvariable=self.camera_var, 
                                    values=["0", "1", "2"], width=5, state='readonly')
        camera_combo.pack(side=tk.LEFT)
        
        # Video feed display
        video_frame = ttk.LabelFrame(left_frame, text="Live Camera Feed", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here", 
                                     background='black', foreground='white', 
                                     font=('Arial', 14))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Detection result display
        result_frame = ttk.Frame(left_frame)
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.result_label = ttk.Label(result_frame, text="No Detection", 
                                      font=('Arial', 24, 'bold'), 
                                      foreground='gray')
        self.result_label.pack()
        
        self.confidence_label = ttk.Label(result_frame, text="", 
                                         font=('Arial', 14))
        self.confidence_label.pack()
        
        # RIGHT COLUMN - Controls and Logs
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.rowconfigure(3, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        # Model status
        status_frame = ttk.LabelFrame(right_frame, text="Model Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.model_status_var = tk.StringVar(value="No model loaded")
        ttk.Label(status_frame, textvariable=self.model_status_var, 
                 foreground='orange').grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(status_frame, text="Faces Detected:").grid(row=1, column=0, sticky=tk.W)
        self.faces_count_var = tk.StringVar(value="0")
        ttk.Label(status_frame, textvariable=self.faces_count_var).grid(row=1, column=1, 
                                                                        sticky=tk.W, padx=(5, 0))
        
        # Settings
        settings_frame = ttk.LabelFrame(right_frame, text="Detection Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="Detection Interval (seconds):").pack(anchor=tk.W)
        self.interval_var = tk.DoubleVar(value=1.0)
        interval_scale = ttk.Scale(settings_frame, from_=0.5, to=5.0, 
                                  variable=self.interval_var, orient=tk.HORIZONTAL)
        interval_scale.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Confidence Threshold (%):").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=60.0)
        threshold_scale = ttk.Scale(settings_frame, from_=50.0, to=95.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=5)
        
        # Training section
        train_frame = ttk.LabelFrame(right_frame, text="Model Training", padding="10")
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        train_frame.columnconfigure(1, weight=1)
        
        ttk.Label(train_frame, text="Real Dataset:").grid(row=0, column=0, sticky=tk.W)
        self.real_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.real_path_var, width=20).grid(row=0, column=1, 
                                                                                sticky=(tk.W, tk.E), padx=5)
        ttk.Button(train_frame, text="📁", command=lambda: self.browse_dataset("real"), 
                  width=3).grid(row=0, column=2)
        
        ttk.Label(train_frame, text="Fake Dataset:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.fake_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.fake_path_var, width=20).grid(row=1, column=1, 
                                                                                sticky=(tk.W, tk.E), padx=5)
        ttk.Button(train_frame, text="📁", command=lambda: self.browse_dataset("fake"), 
                  width=3).grid(row=1, column=2)
        
        ttk.Button(train_frame, text="🎓 Train New Model", 
                  command=self.train_model).grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        self.train_progress_var = tk.DoubleVar()
        self.train_progress = ttk.Progressbar(train_frame, variable=self.train_progress_var, 
                                             maximum=100, mode='determinate')
        self.train_progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Log output
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding="5")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=15, width=40, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def load_model_if_exists(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.model_status_var.set("Model loaded ✓")
                self.log("Model loaded successfully")
            except Exception as e:
                self.log(f"Error loading model: {str(e)}")
                self.model_status_var.set("Model load failed ✗")
        else:
            self.log("No trained model found. Please train a model first.")
            
    def browse_dataset(self, dataset_type):
        folder = filedialog.askdirectory(title=f"Select {dataset_type.capitalize()} Dataset Folder")
        if folder:
            if dataset_type == "real":
                self.real_path_var.set(folder)
            else:
                self.fake_path_var.set(folder)
                
    def extract_features(self, img):
        """Extract features from image for classification"""
        if img is None or img.size == 0:
            return None
            
        # Resize for consistency
        img = cv2.resize(img, (128, 128))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract features
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
    
    def start_scanning(self):
        if self.model is None:
            messagebox.showerror("Error", "No model loaded. Please train a model first.")
            return
            
        camera_id = int(self.camera_var.get())
        self.camera = cv2.VideoCapture(camera_id)
        
        if not self.camera.isOpened():
            messagebox.showerror("Error", f"Cannot access camera {camera_id}")
            return
            
        self.is_scanning = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("Scanning started")
        self.status_var.set("Scanning active...")
        
        self.update_frame()
        
    def stop_scanning(self):
        self.is_scanning = False
        if self.camera:
            self.camera.release()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.video_label.config(image='', text="Camera feed stopped")
        self.result_label.config(text="Scanning Stopped", foreground='gray')
        self.confidence_label.config(text="")
        self.log("Scanning stopped")
        self.status_var.set("Ready")
        
    def update_frame(self):
        if not self.is_scanning:
            return
            
        ret, frame = self.camera.read()
        if not ret:
            self.log("Failed to read from camera")
            self.stop_scanning()
            return
            
        self.current_frame = frame.copy()
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        self.faces_count_var.set(str(len(faces)))
        
        # Process faces for deepfake detection
        current_time = datetime.now().timestamp()
        if current_time - self.last_detection_time >= self.interval_var.get() and len(faces) > 0:
            self.last_detection_time = current_time
            threading.Thread(target=self.detect_deepfake, args=(faces,), daemon=True).start()
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")
        
        # Schedule next frame
        self.root.after(30, self.update_frame)
        
    def detect_deepfake(self, faces):
        if self.model is None or self.current_frame is None:
            return
            
        results = []
        
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.current_frame.shape[1], x + w + padding)
            y2 = min(self.current_frame.shape[0], y + h + padding)
            
            face_img = self.current_frame[y1:y2, x1:x2]
            
            # Extract features
            features = self.extract_features(face_img)
            if features is None:
                continue
                
            # Predict
            features = features.reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            confidence = probability[prediction] * 100
            
            results.append({
                'prediction': prediction,
                'confidence': confidence
            })
        
        if results:
            # Use the highest confidence detection
            best_result = max(results, key=lambda x: x['confidence'])
            self.update_detection_display(best_result)
            
    def update_detection_display(self, result):
        threshold = self.threshold_var.get()
        
        if result['confidence'] < threshold:
            self.result_label.config(text="Uncertain", foreground='orange')
            self.confidence_label.config(text=f"Confidence too low: {result['confidence']:.1f}%")
        elif result['prediction'] == 1:  # Deepfake
            self.result_label.config(text="⚠ DEEPFAKE", foreground='red')
            self.confidence_label.config(text=f"Confidence: {result['confidence']:.1f}%")
            self.log(f"DEEPFAKE detected! Confidence: {result['confidence']:.1f}%")
        else:  # Real
            self.result_label.config(text="✓ REAL", foreground='green')
            self.confidence_label.config(text=f"Confidence: {result['confidence']:.1f}%")
            
    def train_model(self):
        real_path = self.real_path_var.get()
        fake_path = self.fake_path_var.get()
        
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
            
        if not os.path.exists(real_path) or not os.path.exists(fake_path):
            messagebox.showerror("Error", "Dataset paths do not exist")
            return
            
        self.log("Starting model training...")
        self.status_var.set("Training in progress...")
        
        def train_thread():
            try:
                # Load datasets
                X_real, y_real = self.load_dataset(real_path, 0)
                X_fake, y_fake = self.load_dataset(fake_path, 1)
                
                if len(X_real) == 0 or len(X_fake) == 0:
                    self.log("Error: No data loaded from datasets")
                    return
                    
                # Combine
                X = np.array(X_real + X_fake)
                y = np.array(y_real + y_fake)
                
                self.log(f"Total samples: {len(X)} (Real: {len(X_real)}, Fake: {len(X_fake)})")
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train
                self.log("Training model...")
                self.train_progress_var.set(50)
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Test
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.log(f"Model accuracy: {accuracy:.4f}")
                self.train_progress_var.set(100)
                
                # Save
                joblib.dump(model, self.model_path)
                self.model = model
                self.model_status_var.set("Model loaded ✓")
                
                self.log("Training completed successfully!")
                self.status_var.set("Ready")
                messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
                
            except Exception as e:
                self.log(f"Training error: {str(e)}")
                self.status_var.set("Training failed")
                messagebox.showerror("Error", f"Training failed: {str(e)}")
            finally:
                self.train_progress_var.set(0)
                
        threading.Thread(target=train_thread, daemon=True).start()
        
    def load_dataset(self, path, label):
        """Load images from dataset path"""
        X, y = [], []
        
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total = len(files)
        
        for idx, file in enumerate(files):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path)
            
            if img is not None:
                features = self.extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    
            if idx % 10 == 0:
                progress = (idx + 1) / total * 100
                self.train_progress_var.set(progress * 0.5)  # First 50% for loading
                
        return X, y


def main():
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()