import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
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
import mss


class SettingsWindow:
    """Separate window for settings"""
    def __init__(self, parent, app):
        self.window = Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("600x700")
        self.window.resizable(False, False)
        self.app = app
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container with scrollbar
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        main_frame = ttk.Frame(scrollable_frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="⚙️ Settings & Configuration", 
                  font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # Detection Settings
        self._create_detection_settings(main_frame)
        
        # Model Training
        self._create_training_settings(main_frame)
        
        # General Settings
        self._create_general_settings(main_frame)
        
        # Bottom buttons (non-scrollable)
        self._create_bottom_buttons()
        
    def _create_detection_settings(self, parent):
        section = ttk.LabelFrame(parent, text="Detection Settings", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        # Detection Interval
        self.interval_var = tk.DoubleVar(value=self.app.detection_interval)
        self._create_slider(section, "Detection Interval:", "Time between detections (seconds)",
                           self.interval_var, 0.5, 10.0, 0.5, "s",
                           "⚠ Lower values = more frequent checks, higher CPU usage")
        
        ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Confidence Threshold
        self.threshold_var = tk.DoubleVar(value=self.app.threshold_var.get())
        self._create_slider(section, "Confidence Threshold:", "Minimum confidence for detection (%)",
                           self.threshold_var, 50.0, 95.0, 1, "%",
                           "⚠ Higher values = fewer false positives, may miss some detections")
        
        ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Max No Face Intervals
        self.max_no_face_var = tk.IntVar(value=self.app.max_no_face_intervals)
        self._create_slider(section, "Display Reset:", "Clear detection after N intervals with no face",
                           self.max_no_face_var, 1, 20, 1, "",
                           "ℹ️ How long to keep showing detection result after face disappears")
        
    def _create_training_settings(self, parent):
        section = ttk.LabelFrame(parent, text="Model Training", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(section, text="Train a new deepfake detection model", 
                 font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=(0, 15))
        
        # Real Dataset
        ttk.Label(section, text="Real Images Dataset:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        real_frame = ttk.Frame(section)
        real_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.real_path_var = tk.StringVar(value=self.app.real_path_var.get())
        ttk.Entry(real_frame, textvariable=self.real_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(real_frame, text="📁 Browse", 
                  command=lambda: self._browse_dataset("real")).pack(side=tk.LEFT)
        
        # Fake Dataset
        ttk.Label(section, text="Fake Images Dataset:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        fake_frame = ttk.Frame(section)
        fake_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.fake_path_var = tk.StringVar(value=self.app.fake_path_var.get())
        ttk.Entry(fake_frame, textvariable=self.fake_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(fake_frame, text="📁 Browse", 
                  command=lambda: self._browse_dataset("fake")).pack(side=tk.LEFT)
        
        # Train button and progress
        ttk.Button(section, text="🎯 Train New Model", 
                  command=self._train_model).pack(fill=tk.X, pady=(0, 10))
        
        self.train_progress_var = tk.DoubleVar()
        self.train_progress = ttk.Progressbar(section, variable=self.train_progress_var, 
                                             maximum=100, mode='determinate')
        self.train_progress.pack(fill=tk.X)
        
        ttk.Label(section, text="ℹ️ Training may take several minutes depending on dataset size", 
                 font=('Arial', 8), foreground='blue').pack(anchor=tk.W, pady=(5, 0))
        
    def _create_general_settings(self, parent):
        section = ttk.LabelFrame(parent, text="General Settings", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        self.auto_start_var = tk.BooleanVar(value=self.app.auto_start_var.get())
        ttk.Checkbutton(section, text="Automatically start scanning when program launches",
                       variable=self.auto_start_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Requires a trained model to be available", 
                 font=('Arial', 8), foreground='blue').pack(anchor=tk.W, pady=(0, 5))
        
    def _create_bottom_buttons(self):
        button_container = ttk.Frame(self.window)
        button_container.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)
        
        ttk.Button(button_container, text="Reset to Defaults", 
                  command=self._reset_defaults).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_container, text="Cancel", 
                  command=self.window.destroy).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_container, text="Apply & Close", 
                  command=self._apply_settings, style='Accent.TButton').pack(side=tk.RIGHT)
        
    def _create_slider(self, parent, title, desc, var, from_, to, step, suffix, warning):
        """Helper to create slider control"""
        ttk.Label(parent, text=title, font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(parent, text=desc, font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=(0, 5))
        
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=(0, 15))
        
        scale = ttk.Scale(slider_frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL,
                         command=lambda v: self._snap_slider(var, label, step, suffix))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        label = ttk.Label(slider_frame, text=f"{var.get()}{suffix}", width=6)
        label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Label(parent, text=warning, font=('Arial', 8), foreground='orange').pack(anchor=tk.W)
        return label
    
    def _snap_slider(self, var, label, step, suffix):
        """Snap slider to step intervals"""
        snapped = round(var.get() / step) * step
        var.set(snapped)
        fmt = f"{int(snapped)}{suffix}" if step >= 1 else f"{snapped:.1f}{suffix}"
        label.config(text=fmt)
    
    def _browse_dataset(self, dataset_type):
        folder = filedialog.askdirectory(title=f"Select {dataset_type.capitalize()} Dataset Folder")
        if folder:
            (self.real_path_var if dataset_type == "real" else self.fake_path_var).set(folder)
    
    def _train_model(self):
        if not self.real_path_var.get() or not self.fake_path_var.get():
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
            
        if not os.path.exists(self.real_path_var.get()) or not os.path.exists(self.fake_path_var.get()):
            messagebox.showerror("Error", "Dataset paths do not exist")
            return
        
        self.app.real_path_var.set(self.real_path_var.get())
        self.app.fake_path_var.set(self.fake_path_var.get())
        self.app.train_model(progress_callback=self.train_progress_var)
        
    def _reset_defaults(self):
        defaults = self.app.load_default_config()
        sc = defaults['screen_capture']
        self.interval_var.set(sc['detection_interval'])
        self.threshold_var.set(sc['confidence_threshold'])
        self.max_no_face_var.set(sc['max_no_face_intervals'])
        self.auto_start_var.set(sc['auto_start_scanning'])
        
    def _apply_settings(self):
        self.app.detection_interval = self.interval_var.get()
        self.app.interval_var.set(self.interval_var.get())
        self.app.threshold_var.set(self.threshold_var.get())
        self.app.max_no_face_intervals = self.max_no_face_var.get()
        self.app.auto_start_var.set(self.auto_start_var.get())
        self.app.real_path_var.set(self.real_path_var.get())
        self.app.fake_path_var.set(self.fake_path_var.get())
        
        self.app.log(f"Settings updated: Interval={self.interval_var.get():.1f}s, "
                    f"Threshold={self.threshold_var.get():.0f}%, "
                    f"Max No Face={self.max_no_face_var.get()}, "
                    f"Auto-start={self.auto_start_var.get()}")
        
        self.app.save_config()
        self.window.destroy()


class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_path="config.json", default_path="default.json"):
        self.config_path = config_path
        self.default_path = default_path
        
    @staticmethod
    def get_defaults():
        return {
            "screen_capture": {
                "detection_interval": 3.0,
                "selected_monitor": 0,
                "no_face_count": 0,
                "max_no_face_intervals": 5,
                "confidence_threshold": 60.0,
                "auto_start_scanning": True
            }
        }
    
    def load_defaults(self):
        defaults = self.get_defaults()
        if not os.path.exists(self.default_path):
            try:
                with open(self.default_path, 'w') as f:
                    json.dump(defaults, f, indent=2)
            except Exception as e:
                print(f"Could not create default config: {e}")
        
        try:
            with open(self.default_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading default config: {e}")
            return defaults
    
    def load(self):
        config = self.load_defaults()
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    if 'screen_capture' in loaded:
                        config['screen_capture'].update(loaded['screen_capture'])
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        return config
    
    def save(self, config):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False


class DeepfakeModel:
    """Handles model training and prediction"""
    
    def __init__(self, model_path="deepfake_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def load(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False
    
    def save(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    def extract_features(self, img):
        if img is None or img.size == 0:
            return None
        
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Color histograms
        for i in range(3):
            features.extend(cv2.calcHist([img], [i], None, [8], [0, 256]).flatten())
        
        # Texture and edge features
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = cv2.Canny(gray, 100, 200)
        for arr in [laplacian, edges]:
            features.extend([arr.mean(), arr.std()])
        features.append(laplacian.var())
        
        # DCT features
        dct = cv2.dct(np.float32(gray))
        features.extend([dct.mean(), dct.std(), dct.var()])
        
        return np.array(features)
    
    def predict(self, face_img):
        if not self.model:
            return None
            
        features = self.extract_features(face_img)
        if features is None:
            return None
            
        features = features.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        confidence = probability[prediction] * 100
        
        return {'prediction': prediction, 'confidence': confidence}
    
    def train(self, real_path, fake_path, log_callback, progress_callback=None):
        """Train the model on provided datasets"""
        log_callback("Starting model training...")
        progress_var = progress_callback if progress_callback else tk.DoubleVar()
        
        try:
            # Load datasets
            log_callback("Loading real images...")
            X_real, y_real = self._load_dataset(real_path, 0, progress_var, log_callback)
            log_callback("Loading fake images...")
            X_fake, y_fake = self._load_dataset(fake_path, 1, progress_var, log_callback)
            
            if not X_real or not X_fake:
                log_callback("Error: No data loaded from datasets")
                return False
                
            # Combine and split
            X = np.array(X_real + X_fake)
            y = np.array(y_real + y_fake)
            log_callback(f"Total samples: {len(X)} (Real: {len(X_real)}, Fake: {len(X_fake)})")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train
            log_callback("Training Random Forest model...")
            progress_var.set(60)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            # Test
            progress_var.set(90)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            log_callback(f"Model accuracy: {accuracy:.4f}")
            progress_var.set(100)
            
            # Save
            self.save()
            log_callback("✓ Training completed successfully!")
            
            return accuracy
            
        except Exception as e:
            log_callback(f"✗ Training error: {e}")
            return False
        finally:
            if progress_callback:
                progress_var.set(0)
    
    def _load_dataset(self, path, label, progress_var, log_callback):
        X, y = [], []
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total = len(files)
        
        log_callback(f"Found {total} images in {os.path.basename(path)} dataset")
        
        for idx, file in enumerate(files):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                features = self.extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    
            if idx % 10 == 0:
                progress = (idx + 1) / total * 100
                if progress_var:
                    progress_var.set(progress * 0.5)
                log_callback(f"Processing: {idx + 1}/{total}")
                
        log_callback(f"Loaded {len(X)} valid samples from {os.path.basename(path)}")
        return X, y


class ScreenDeepfakeDetector:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Deepfake Detection System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.model = DeepfakeModel()
        
        # Screen capture
        self.sct = mss.mss()
        self.is_scanning = False
        self.current_frame = None
        self.last_detection_time = 0
        
        # Configuration variables
        sc = self.config['screen_capture']
        self.detection_interval = sc.get('detection_interval', 3.0)
        self.selected_monitor = sc.get('selected_monitor', 0)
        self.no_face_count = sc.get('no_face_count', 0)
        self.max_no_face_intervals = sc.get('max_no_face_intervals', 5)
        
        # UI variables
        self.threshold_var = tk.DoubleVar(value=sc.get('confidence_threshold', 60.0))
        self.interval_var = tk.DoubleVar(value=self.detection_interval)
        self.auto_start_var = tk.BooleanVar(value=sc.get('auto_start_scanning', True))
        self.real_path_var = tk.StringVar()
        self.fake_path_var = tk.StringVar()
        
        # Detection state
        self.last_detection_result = None
        self.last_detected_faces = []
        
        # Statistics
        self.total_scans = 0
        self.deepfakes_detected = 0
        
        # Setup UI and load model
        self.setup_ui()
        self._load_model()
        
        # Auto-start if enabled
        if self.auto_start_var.get() and self.model.model:
            self.root.after(500, self.start_scanning)
            self.log("Auto-start enabled: Scanning will begin automatically")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        ttk.Label(main_frame, text="🖥️ Screen Deepfake Detection System", 
                  font=('Arial', 18, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left and right columns
        self._setup_left_column(main_frame)
        self._setup_right_column(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Train a model to begin")
        ttk.Label(main_frame, textvariable=self.status_var, 
                  relief=tk.SUNKEN, anchor=tk.W).grid(row=2, column=0, columnspan=2, 
                                                      sticky=(tk.W, tk.E), pady=(10, 0))
    
    def _setup_left_column(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        # Controls
        controls = ttk.Frame(left_frame)
        controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(controls, text="▶ Start Scanning", 
                                       command=self.start_scanning, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls, text="⏸ Stop Scanning", 
                                      command=self.stop_scanning, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls, text="Monitor:").pack(side=tk.LEFT, padx=(20, 5))
        self.monitor_var = tk.StringVar(value="All Screens")
        monitor_options = ["All Screens"] + [f"Monitor {i}" for i in range(1, len(self.sct.monitors))]
        monitor_combo = ttk.Combobox(controls, textvariable=self.monitor_var, 
                                    values=monitor_options, width=15, state='readonly')
        monitor_combo.pack(side=tk.LEFT)
        monitor_combo.bind('<<ComboboxSelected>>', self._on_monitor_change)
        
        ttk.Button(controls, text="⚙️ Settings", 
                  command=lambda: SettingsWindow(self.root, self)).pack(side=tk.RIGHT, padx=5)
        
        # Video feed
        video_frame = ttk.LabelFrame(left_frame, text="Screen Capture Feed", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, 
                                     text="Screen capture will appear here\n\nClick 'Start Scanning' to begin", 
                                     background='black', foreground='white', font=('Arial', 14))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results
        result_frame = ttk.Frame(left_frame)
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.result_label = ttk.Label(result_frame, text="No Detection", 
                                      font=('Arial', 24, 'bold'), foreground='gray')
        self.result_label.pack()
        
        self.confidence_label = ttk.Label(result_frame, text="", font=('Arial', 14))
        self.confidence_label.pack()
        
        # Statistics
        self.stats_label = ttk.Label(left_frame, text="Scans: 0 | Deepfakes: 0 | Real: 0", 
                                     font=('Arial', 10), foreground='blue')
        self.stats_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _setup_right_column(self, parent):
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.rowconfigure(2, weight=1)
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
        
        ttk.Label(status_frame, text="Screen Region:").grid(row=2, column=0, sticky=tk.W)
        self.region_var = tk.StringVar(value="Not capturing")
        ttk.Label(status_frame, textvariable=self.region_var, 
                 font=('Arial', 8)).grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Quick actions
        actions_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding="10")
        actions_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(actions_frame, text="🔄 Reset Statistics", 
                  command=self._reset_statistics).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="💾 Save Screenshot", 
                  command=self._save_screenshot).pack(fill=tk.X, pady=2)
        
        # Log
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding="5")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=12, width=40, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_msg + "\n")
            self.log_text.see(tk.END)
    
    def load_default_config(self):
        return self.config_manager.load_defaults()
    
    def save_config(self):
        config = {
            "screen_capture": {
                "detection_interval": self.detection_interval,
                "selected_monitor": self.selected_monitor,
                "no_face_count": self.no_face_count,
                "max_no_face_intervals": self.max_no_face_intervals,
                "confidence_threshold": self.threshold_var.get(),
                "auto_start_scanning": self.auto_start_var.get()
            }
        }
        
        if self.config_manager.save(config):
            self.log(f"Configuration saved to {self.config_manager.config_path}")
            return True
        else:
            self.log("Error saving configuration")
            return False
    
    def _load_model(self):
        if self.model.load():
            self.model_status_var.set("Model loaded ✓")
            self.log("Model loaded successfully")
            self.status_var.set("Ready - Click 'Start Scanning'")
        else:
            self.log("No trained model found. Please train a model first.")
    
    def _on_monitor_change(self, event=None):
        selection = self.monitor_var.get()
        self.selected_monitor = 0 if selection == "All Screens" else int(selection.split()[-1])
        self.log(f"Monitor changed to: {selection}")
    
    def _on_closing(self):
        if self.is_scanning:
            self.stop_scanning()
        self.log("Saving configuration before exit...")
        self.save_config()
        self.root.destroy()
    
    def start_scanning(self):
        if not self.model.model:
            messagebox.showerror("Error", "No model loaded. Please train a model first.")
            return
            
        self.is_scanning = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("Screen scanning started")
        self.status_var.set("Scanning screen...")
        self._update_screen_capture()
    
    def stop_scanning(self):
        self.is_scanning = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.video_label.config(image='', text="Screen scanning stopped")
        self.result_label.config(text="Scanning Stopped", foreground='gray')
        self.confidence_label.config(text="")
        self.region_var.set("Not capturing")
        self.no_face_count = 0
        self.log("Screen scanning stopped")
        self.status_var.set("Ready")
    
    def _capture_screen(self):
        try:
            monitor = self.sct.monitors[self.selected_monitor]
            sct_img = self.sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.region_var.set(f"{monitor['width']}x{monitor['height']}")
            return img
        except Exception as e:
            self.log(f"Screen capture error: {e}")
            return None
    
    def _update_screen_capture(self):
        if not self.is_scanning:
            return
            
        frame = self._capture_screen()
        if frame is None:
            self.log("Failed to capture screen")
            self.stop_scanning()
            return
            
        self.current_frame = frame.copy()
        display_frame = frame.copy()
        
        # Detect faces
        faces = self.model.detect_faces(frame)
        self.faces_count_var.set(str(len(faces)))
        self.last_detected_faces = faces
        
        # Process detection
        current_time = datetime.now().timestamp()
        if current_time - self.last_detection_time >= self.interval_var.get():
            self.last_detection_time = current_time
            
            if len(faces) > 0:
                self.no_face_count = 0
                threading.Thread(target=self._detect_deepfake, args=(faces,), daemon=True).start()
            else:
                self.no_face_count += 1
                if self.no_face_count >= self.max_no_face_intervals:
                    self._reset_detection_display()
        
        # Draw annotations
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
            
            if self.last_detection_result:
                result_text = "DEEPFAKE" if self.last_detection_result['prediction'] == 1 else "REAL"
                conf = self.last_detection_result['confidence']
                color = (0, 0, 255) if self.last_detection_result['prediction'] == 1 else (0, 255, 0)
                cv2.putText(display_frame, f"{result_text} {conf:.1f}%", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display frame
        self._display_frame(display_frame)
        self.root.after(100, self._update_screen_capture)
    
    def _display_frame(self, frame):
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        display_height = 480
        display_width = int(display_height * aspect_ratio)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((min(display_width, 800), display_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")
    
    def _detect_deepfake(self, faces):
        if not self.model.model or self.current_frame is None:
            return
            
        results = []
        for (x, y, w, h) in faces:
            padding = 20
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2 = min(self.current_frame.shape[1], x + w + padding)
            y2 = min(self.current_frame.shape[0], y + h + padding)
            
            face_img = self.current_frame[y1:y2, x1:x2]
            result = self.model.predict(face_img)
            if result:
                results.append(result)
        
        if results:
            best_result = max(results, key=lambda x: x['confidence'])
            self._update_detection_display(best_result)
            self._update_statistics(best_result)
    
    def _update_detection_display(self, result):
        self.last_detection_result = result
        conf = result['confidence']
        threshold = self.threshold_var.get()
        
        if conf < threshold:
            self.result_label.config(text="⚠ Uncertain", foreground='orange')
            self.confidence_label.config(text=f"Confidence too low: {conf:.1f}%")
        elif result['prediction'] == 1:
            self.result_label.config(text="🚨 DEEPFAKE", foreground='red')
            self.confidence_label.config(text=f"Confidence: {conf:.1f}%")
            self.log(f"⚠ DEEPFAKE detected! Confidence: {conf:.1f}%")
        else:
            self.result_label.config(text="✓ REAL", foreground='green')
            self.confidence_label.config(text=f"Confidence: {conf:.1f}%")
    
    def _reset_detection_display(self):
        self.result_label.config(text="No Detection", foreground='gray')
        self.confidence_label.config(text="")
        self.no_face_count = 0
        self.last_detection_result = None
    
    def _update_statistics(self, result):
        if result['confidence'] >= self.threshold_var.get():
            self.total_scans += 1
            if result['prediction'] == 1:
                self.deepfakes_detected += 1
            real_count = self.total_scans - self.deepfakes_detected
            self.stats_label.config(text=f"Scans: {self.total_scans} | "
                                        f"Deepfakes: {self.deepfakes_detected} | Real: {real_count}")
    
    def _reset_statistics(self):
        self.total_scans = 0
        self.deepfakes_detected = 0
        self.stats_label.config(text="Scans: 0 | Deepfakes: 0 | Real: 0")
        self.log("Statistics reset")
    
    def _save_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No screen capture available")
            return
        
        annotated_frame = self.current_frame.copy()
        
        for (x, y, w, h) in self.last_detected_faces:
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(annotated_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
            
            if self.last_detection_result:
                result_text = "DEEPFAKE" if self.last_detection_result['prediction'] == 1 else "REAL"
                conf = self.last_detection_result['confidence']
                color = (0, 0, 255) if self.last_detection_result['prediction'] == 1 else (0, 255, 0)
                cv2.putText(annotated_frame, f"{result_text} {conf:.1f}%", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, annotated_frame)
        
        if self.last_detection_result:
            result_type = "DEEPFAKE" if self.last_detection_result['prediction'] == 1 else "REAL"
            self.log(f"Screenshot saved: {filename} (Detection: {result_type} "
                    f"{self.last_detection_result['confidence']:.1f}%)")
        else:
            self.log(f"Screenshot saved: {filename}")
        
        messagebox.showinfo("Success", f"Screenshot saved as {filename}")
    
    def train_model(self, progress_callback=None):
        real_path = self.real_path_var.get()
        fake_path = self.fake_path_var.get()
        
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
            
        if not os.path.exists(real_path) or not os.path.exists(fake_path):
            messagebox.showerror("Error", "Dataset paths do not exist")
            return
        
        self.status_var.set("Training in progress...")
        
        def train_thread():
            result = self.model.train(real_path, fake_path, self.log, progress_callback)
            
            if result:
                self.model_status_var.set("Model loaded ✓")
                self.status_var.set("Ready - Click 'Start Scanning'")
                messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {result:.4f}")
            else:
                self.status_var.set("Training failed")
                messagebox.showerror("Error", "Training failed. Check log for details.")
        
        threading.Thread(target=train_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = ScreenDeepfakeDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()