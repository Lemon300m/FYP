import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
from tkinter.scrolledtext import ScrolledText
import cv2
from PIL import Image, ImageTk, ImageGrab
import threading
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
import joblib
import os
from datetime import datetime
import json
import shutil
import sys
import winreg
from tray_handler import TrayHandler

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_data_path(relative_path):
    """Get path for data files (always use executable/script directory)"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable - use executable's directory
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script - use script's directory
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

# For multi-monitor support
try:
    from screeninfo import get_monitors
    MULTI_MONITOR_SUPPORT = True
except ImportError:
    MULTI_MONITOR_SUPPORT = False
    print("screeninfo not installed. Multi-monitor support disabled.")

class SettingsWindow:
    """Separate window for settings"""
    
    def __init__(self, parent, app):
        self.window = Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("600x750")
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
        
        # Self-Learning Settings
        self._create_self_learning_settings(main_frame)
        
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
        self.train_progress_var = tk.DoubleVar()
        self.train_progress = ttk.Progressbar(section, variable=self.train_progress_var, 
                                             maximum=100, mode='determinate')
        
        ttk.Button(section, text="🎯 Train New Model", 
                  command=self._train_model).pack(fill=tk.X, pady=(0, 10))
        self.train_progress.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(section, text="🔎 Test Current Model", 
              command=self._test_model).pack(fill=tk.X, pady=(0, 10))
        
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
        
        ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=15)
        
        self.start_minimized_var = tk.BooleanVar(value=self.app.start_minimized_var.get())
        ttk.Checkbutton(section, text="Start minimized to system tray",
                       variable=self.start_minimized_var).pack(anchor=tk.W, pady=5)
        
        ttk.Separator(section, orient='horizontal').pack(fill=tk.X, pady=15)
        
        self.start_with_windows_var = tk.BooleanVar(value=self.app.start_with_windows_var.get())
        ttk.Checkbutton(section, text="Start application with Windows",
                       variable=self.start_with_windows_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Application will auto-launch when you log in", 
                 font=('Arial', 8), foreground='blue').pack(anchor=tk.W, pady=(0, 5))
    
    def _create_self_learning_settings(self, parent):
        section = ttk.LabelFrame(parent, text="Self-Learning Settings", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        self.enable_self_learning_var = tk.BooleanVar(value=self.app.enable_self_learning_var.get())
        ttk.Checkbutton(section, text="Enable automatic model improvement",
                       variable=self.enable_self_learning_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Model will retrain using classified images after each detection session", 
                 font=('Arial', 8), foreground='blue').pack(anchor=tk.W, pady=(0, 10))
        
        # Minimum samples for retraining
        self.min_samples_var = tk.IntVar(value=self.app.min_samples_for_retrain)
        self._create_slider(section, "Minimum Samples:", "Minimum new samples needed to trigger retraining",
                           self.min_samples_var, 5, 50, 5, "",
                           "⚠ Lower values = more frequent retraining, higher processing time")
        
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
    
    def _update_windows_startup(self, enable):
        """Add/remove application from Windows startup registry"""
        try:
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
            
            app_name = "DeepfakeDetector"
            if enable:
                if getattr(sys, 'frozen', False):
                    # Running as compiled executable
                    exe_path = sys.executable
                    exe_dir = os.path.dirname(exe_path)
                    # Use /d to set working directory
                    full_command = f'cmd /c "cd /d "{exe_dir}" && start "" "{exe_path}""'
                else:
                    # Running as Python script
                    exe_path = sys.executable
                    script_path = os.path.abspath(__file__)
                    script_dir = os.path.dirname(script_path)
                    full_command = f'cmd /c "cd /d "{script_dir}" && "{exe_path}" "{script_path}""'
                
                winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, full_command)
                self.app.log("✓ Application added to Windows startup")
            else:
                try:
                    winreg.DeleteValue(key, app_name)
                    self.app.log("✓ Application removed from Windows startup")
                except WindowsError:
                    pass
            
            winreg.CloseKey(key)
        except Exception as e:
            self.app.log(f"⚠ Error updating Windows startup: {e}")
    
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
        self.enable_self_learning_var.set(sc.get('enable_self_learning', True))
        self.min_samples_var.set(sc.get('min_samples_for_retrain', 20))
        
    def _apply_settings(self):
        self.app.detection_interval = self.interval_var.get()
        self.app.interval_var.set(self.interval_var.get())
        self.app.threshold_var.set(self.threshold_var.get())
        self.app.max_no_face_intervals = self.max_no_face_var.get()
        self.app.auto_start_var.set(self.auto_start_var.get())
        self.app.enable_self_learning_var.set(self.enable_self_learning_var.get())
        self.app.min_samples_for_retrain = self.min_samples_var.get()
        self.app.real_path_var.set(self.real_path_var.get())
        self.app.fake_path_var.set(self.fake_path_var.get())
        self.app.start_minimized_var.set(self.start_minimized_var.get())
        
        # Handle Windows startup registry
        if self.start_with_windows_var.get() != self.app.start_with_windows_var.get():
            self.app.start_with_windows_var.set(self.start_with_windows_var.get())
            self._update_windows_startup(self.start_with_windows_var.get())
        
        self.app.log(f"Settings updated: Interval={self.interval_var.get():.1f}s, "
                    f"Threshold={self.threshold_var.get():.0f}%, "
                    f"Max No Face={self.max_no_face_var.get()}, "
                    f"Auto-start={self.auto_start_var.get()}, "
                    f"Start Minimized={self.start_minimized_var.get()}, "
                    f"Start with Windows={self.start_with_windows_var.get()}, "
                    f"Self-learning={self.enable_self_learning_var.get()}")
        
        self.app.save_config()
        self.window.destroy()

    def _test_model(self):
        """Handler to test the currently saved model using the dataset paths in the UI"""
        if not self.real_path_var.get() or not self.fake_path_var.get():
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
        if not os.path.exists(self.real_path_var.get()) or not os.path.exists(self.fake_path_var.get()):
            messagebox.showerror("Error", "Dataset paths do not exist")
            return

        self.app.real_path_var.set(self.real_path_var.get())
        self.app.fake_path_var.set(self.fake_path_var.get())
        threading.Thread(target=self.app.test_model, args=(self.real_path_var.get(), self.fake_path_var.get()), daemon=True).start()

class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self, config_path="config.json", default_path="default.json"):
        self.config_path = get_data_path(config_path)
        self.default_path = get_data_path(default_path)
        
    @staticmethod
    def get_defaults():
        return {
            "screen_capture": {
                "detection_interval": 3.0,
                "selected_monitor": 0,
                "no_face_count": 0,
                "max_no_face_intervals": 5,
                "confidence_threshold": 60.0,
                "auto_start_scanning": False,
                "enable_self_learning": True,
                "min_samples_for_retrain": 20,
                "start_minimized": False,
                "start_with_windows": False,
                "selected_model": None
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
        self.model_archive_dir = get_data_path("model_archive")
        self.model = None
        self.current_model_path = None  # Track which model is loaded
        
        try:
            os.makedirs(self.model_archive_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create archive directory: {e}")
        
        self.face_cascade = None
    
        cascade_paths = [
            get_resource_path('haarcascade_frontalface_default.xml'),  # Bundled with exe
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),  # Same dir as script
            'haarcascade_frontalface_default.xml',  # Current directory
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # OpenCV data
        ]
        
        for cascade_path in cascade_paths:
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.face_cascade.empty():
                    print(f"✓ Loaded Haar Cascade from: {cascade_path}")
                    break
        
        if self.face_cascade is None or self.face_cascade.empty():
            print("⚠ WARNING: Could not load Haar Cascade classifier!")
            print("Face detection will not work properly.")
        
        self.model_archive_dir = get_data_path("model_archive")
        try:
            os.makedirs(self.model_archive_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create archive directory: {e}")
            self.model_archive_dir = "."

    def get_available_models(self):
        """Get list of available model files from model_archive directory"""
        try:
            models = []
            for file in os.listdir(self.model_archive_dir):
                if file.endswith('.pkl'):
                    full_path = os.path.join(self.model_archive_dir, file)
                    models.append({
                        'name': file,
                        'path': full_path,
                        'mtime': os.path.getmtime(full_path),
                        'size': os.path.getsize(full_path)
                    })
            # Sort by modification time (newest first)
            models.sort(key=lambda x: x['mtime'], reverse=True)
            return models
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []
    
    def get_latest_model_path(self):
        """Get path to the latest model in model_archive"""
        models = self.get_available_models()
        if models:
            return models[0]['path']
        return None
            
    def load(self, model_path=None):
        """Load a specific model or the latest one"""
        try:
            if model_path is None:
                model_path = self.get_latest_model_path()
            
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.current_model_path = model_path
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save(self, filename=None):
        """Save model to model_archive directory"""
        if self.model:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_{timestamp}.pkl"
            
            save_path = os.path.join(self.model_archive_dir, filename)
            joblib.dump(self.model, save_path)
            self.current_model_path = save_path
            return save_path
        return None
    
    def archive_current_model(self):
        if os.path.exists(self.model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(self.model_archive_dir, f"model_{timestamp}.pkl")
            try:
                shutil.copy2(self.model_path, archive_path)
                return archive_path
            except Exception as e:
                print(f"Error archiving model: {e}")
                return None
        return None
            
    def detect_faces(self, frame):
        if self.face_cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    def extract_features(self, img):
        if img is None or img.size == 0:
            return None
        
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []
        
        for i in range(3):
            features.extend(cv2.calcHist([img], [i], None, [8], [0, 256]).flatten())
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = cv2.Canny(gray, 100, 200)
        for arr in [laplacian, edges]:
            features.extend([arr.mean(), arr.std()])
        features.append(laplacian.var())
        
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
    
    def train(self, real_path, fake_path, log_callback, progress_callback=None, archive=True):
        log_callback("Starting model training...")
        progress_var = progress_callback if progress_callback else tk.DoubleVar()
        
        try:
            # Models are automatically saved to model_archive with timestamps
            
            log_callback("Loading real images...")
            X_real, y_real = self._load_dataset(real_path, 0, progress_var, log_callback)
            log_callback("Loading fake images...")
            X_fake, y_fake = self._load_dataset(fake_path, 1, progress_var, log_callback)
            
            if not X_real or not X_fake:
                log_callback("Error: No data loaded from datasets")
                return False
                
            X = np.array(X_real + X_fake)
            y = np.array(y_real + y_fake)
            log_callback(f"Total samples: {len(X)} (Real: {len(X_real)}, Fake: {len(X_fake)})")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            log_callback("Training Random Forest model...")
            progress_var.set(60)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            progress_var.set(90)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            log_callback(f"Model accuracy: {accuracy:.4f}")
            progress_var.set(100)
            
            saved_path = self.save()
            if saved_path:
                log_callback(f"Model saved to: {os.path.basename(saved_path)}")
            
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
                
        log_callback(f"Loaded {len(X)} valid samples from {os.path.basename(path)}")
        return X, y

class SelfLearningManager:
    """Manages self-learning data collection and retraining"""
    
    def __init__(self, base_dir="self_learning_data"):
        self.base_dir = get_data_path(base_dir)
        self.real_dir = os.path.join(self.base_dir, "real")
        self.fake_dir = os.path.join(self.base_dir, "fake")
        try:
            os.makedirs(self.real_dir, exist_ok=True)
            os.makedirs(self.fake_dir, exist_ok=True)
            print(f"✓ Self-learning directories ready: {self.base_dir}")
        except Exception as e:
            print(f"Warning: Failed to create self-learning directories: {e}")
        self.session_active = False
        self.session_samples = {'real': 0, 'fake': 0}
        
    def start_session(self):
        self.session_active = True
        self.session_samples = {'real': 0, 'fake': 0}
        
    def end_session(self):
        self.session_active = False
        total = self.session_samples['real'] + self.session_samples['fake']
        return total
        
    def save_classified_image(self, face_img, prediction, confidence):
        if not self.session_active:
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            label = "real" if prediction == 0 else "fake"
            target_dir = self.real_dir if prediction == 0 else self.fake_dir
            filename = f"{label}_{confidence:.1f}_{timestamp}.png"
            filepath = os.path.join(target_dir, filename)
            cv2.imwrite(filepath, face_img)
            self.session_samples[label] += 1
            return True
        except Exception as e:
            print(f"Error saving classified image: {e}")
            return False
    
    def get_sample_counts(self):
        real_count = len([f for f in os.listdir(self.real_dir) if f.endswith('.png')])
        fake_count = len([f for f in os.listdir(self.fake_dir) if f.endswith('.png')])
        return {'real': real_count, 'fake': fake_count}
    
    def clear_data(self):
        try:
            for file in os.listdir(self.real_dir):
                os.remove(os.path.join(self.real_dir, file))
            for file in os.listdir(self.fake_dir):
                os.remove(os.path.join(self.fake_dir, file))
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False

class ScreenCaptureManager:
    """Manages screen capture using PIL ImageGrab - PyInstaller friendly"""
    
    def __init__(self):
        self.monitors = self._detect_monitors()
        self.selected_monitor = 0
        
    def _detect_monitors(self):
        """Detect available monitors"""
        monitors = [{"name": "All Screens", "bbox": None}]
        
        if MULTI_MONITOR_SUPPORT:
            try:
                screen_monitors = get_monitors()
                for idx, monitor in enumerate(screen_monitors, 1):
                    monitors.append({
                        "name": f"Monitor {idx}",
                        "bbox": (monitor.x, monitor.y, 
                                monitor.x + monitor.width, 
                                monitor.y + monitor.height),
                        "width": monitor.width,
                        "height": monitor.height
                    })
            except Exception as e:
                print(f"Error detecting monitors: {e}")
        
        return monitors
    
    def get_monitor_names(self):
        """Get list of monitor names"""
        return [m["name"] for m in self.monitors]
    
    def set_monitor(self, index):
        """Set the active monitor"""
        if 0 <= index < len(self.monitors):
            self.selected_monitor = index
            return True
        return False
    
    def capture(self):
        """Capture screen using mss (PyInstaller friendly) with PIL fallback"""
        try:
            # Try mss first (works better with PyInstaller)
            if MSS_AVAILABLE:
                try:
                    with mss.mss() as sct:
                        monitor = self.monitors[self.selected_monitor]
                        
                        if monitor["bbox"] is None:
                            # Capture primary monitor
                            bbox = sct.monitors[1]
                        else:
                            bbox = monitor["bbox"]
                        
                        # Capture with mss
                        screenshot = sct.grab(bbox)
                        
                        # Convert to numpy array
                        img = np.array(screenshot)
                        
                        # mss returns BGRA, we need BGR
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
                        height, width = img.shape[:2]
                        region_info = f"{width}x{height}"
                        
                        return img, region_info
                except Exception as e:
                    raise
            else:
                raise ImportError("mss not available, falling back to PIL")
                
        except Exception as e:
            # Fallback to PIL ImageGrab
            try:
                monitor = self.monitors[self.selected_monitor]
                
                if monitor["bbox"] is None:
                    screenshot = ImageGrab.grab()
                else:
                    screenshot = ImageGrab.grab(bbox=monitor["bbox"])
                
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                height, width = img.shape[:2]
                region_info = f"{width}x{height}"
                
                return img, region_info
                
            except Exception as e2:
                return None, None

class ScreenDeepfakeDetector:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Deepfake Detection System - Self-Learning Edition")
        self.root.geometry("1200x850")
        self.root.resizable(True, True)
        
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.model = DeepfakeModel()
        self.self_learning = SelfLearningManager()
        self.screen_capture = ScreenCaptureManager()
        sc = self.config['screen_capture']
        self.selected_model_path = sc.get('selected_model', None)
        # Tray handler (created but not started until needed)
        try:
            self.tray_handler = TrayHandler(app_name="Deepfake Detector",
                                            show_callback=self._show_from_tray,
                                            exit_callback=self._exit_from_tray)
        except Exception:
            self.tray_handler = None
        
        self.is_scanning = False
        self.current_frame = None
        self.last_detection_time = 0
        
        self.detection_interval = sc.get('detection_interval', 1.0)
        self.selected_monitor = sc.get('selected_monitor', 0)
        self.no_face_count = sc.get('no_face_count', 0)
        self.max_no_face_intervals = sc.get('max_no_face_intervals', 5)
        self.min_samples_for_retrain = sc.get('min_samples_for_retrain', 20)
        
        self.threshold_var = tk.DoubleVar(value=sc.get('confidence_threshold', 60.0))
        self.interval_var = tk.DoubleVar(value=self.detection_interval)
        self.auto_start_var = tk.BooleanVar(value=sc.get('auto_start_scanning', False))
        self.enable_self_learning_var = tk.BooleanVar(value=sc.get('enable_self_learning', True))
        self.start_minimized_var = tk.BooleanVar(value=sc.get('start_minimized', False))
        self.start_with_windows_var = tk.BooleanVar(value=sc.get('start_with_windows', False))
        self.real_path_var = tk.StringVar()
        self.fake_path_var = tk.StringVar()
        
        self.last_detection_result = None
        self.last_detected_faces = []
        self.had_detection_in_session = False
        self.is_retraining = False
        
        self.total_scans = 0
        self.deepfakes_detected = 0
        
        self.setup_ui()
        self._load_model()
        
        if self.auto_start_var.get() and self.model.model:
            self.root.after(500, self.start_scanning)
            self.log("Auto-start enabled: Scanning will begin automatically")
        
        # Start minimized if configured
        if self.start_minimized_var.get():
            self.root.after(100, self._hide_to_tray)
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        # When minimized to taskbar, hide to system tray
        try:
            self.root.bind('<Unmap>', self._on_minimize)
        except Exception:
            pass
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        
        ttk.Label(main_frame, text="🖥️ Screen Deepfake Detection System", 
                  font=('Arial', 18, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        self._setup_left_column(main_frame)
        self._setup_right_column(main_frame)
        
        self.status_var = tk.StringVar(value="Ready - Train a model to begin")
        ttk.Label(main_frame, textvariable=self.status_var, 
                  relief=tk.SUNKEN, anchor=tk.W).grid(row=2, column=0, columnspan=2, 
                                                      sticky=(tk.W, tk.E), pady=(10, 0))
    
    def _setup_left_column(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        controls = ttk.Frame(left_frame)
        controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(controls, text="▶ Start Scanning", 
                                       command=self.start_scanning, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls, text="⏹ Stop Scanning", 
                                      command=self.stop_scanning, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls, text="Monitor:").pack(side=tk.LEFT, padx=(20, 5))
        self.monitor_var = tk.StringVar(value="All Screens")
        monitor_options = self.screen_capture.get_monitor_names()
        monitor_combo = ttk.Combobox(controls, textvariable=self.monitor_var, 
                                    values=monitor_options, width=15, state='readonly')
        monitor_combo.pack(side=tk.LEFT)
        monitor_combo.bind('<<ComboboxSelected>>', self._on_monitor_change)
        
        ttk.Button(controls, text="⚙️ Settings", 
                  command=lambda: SettingsWindow(self.root, self)).pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(controls, text="Model:").pack(side=tk.LEFT, padx=(20, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, 
                                        width=25, state='readonly')
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        ttk.Button(controls, text="🔄", command=self._refresh_model_list, 
                width=3).pack(side=tk.LEFT)

        video_frame = ttk.LabelFrame(left_frame, text="Screen Capture Feed", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, 
                                     text="Screen capture will appear here\n\nClick 'Start Scanning' to begin", 
                                     background='black', foreground='white', font=('Arial', 14))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        result_frame = ttk.Frame(left_frame)
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.result_label = ttk.Label(result_frame, text="No Detection", 
                                      font=('Arial', 24, 'bold'), foreground='gray')
        self.result_label.pack()
        self.confidence_label = ttk.Label(result_frame, text="", font=('Arial', 14))
        self.confidence_label.pack()
        
        self.stats_label = ttk.Label(left_frame, text="Scans: 0 | Deepfakes: 0 | Real: 0", 
                                     font=('Arial', 10), foreground='blue')
        self.stats_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _setup_right_column(self, parent):
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.rowconfigure(3, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
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
        
        learning_frame = ttk.LabelFrame(right_frame, text="Self-Learning Status", padding="10")
        learning_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        learning_frame.columnconfigure(1, weight=1)
        
        ttk.Label(learning_frame, text="Enabled:").grid(row=0, column=0, sticky=tk.W)
        self.learning_status_var = tk.StringVar(value="Yes" if self.enable_self_learning_var.get() else "No")
        ttk.Label(learning_frame, textvariable=self.learning_status_var).grid(row=0, column=1, 
                                                                              sticky=tk.W, padx=(5, 0))
        
        ttk.Label(learning_frame, text="Collected Samples:").grid(row=1, column=0, sticky=tk.W)
        self.samples_count_var = tk.StringVar(value="Real: 0 | Fake: 0")
        ttk.Label(learning_frame, textvariable=self.samples_count_var, 
                 font=('Arial', 8)).grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Button(learning_frame, text="🗑️ Clear Training Data", 
                  command=self._clear_training_data).grid(row=2, column=0, columnspan=2, 
                                                          sticky=(tk.W, tk.E), pady=(5, 0))
        
        actions_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding="10")
        actions_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Button(actions_frame, text="🔄 Reset Statistics", 
                  command=self._reset_statistics).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="💾 Save Screenshot", 
                  command=self._save_screenshot).pack(fill=tk.X, pady=2)
        
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding="5")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
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
                "auto_start_scanning": self.auto_start_var.get(),
                "enable_self_learning": self.enable_self_learning_var.get(),
                "min_samples_for_retrain": self.min_samples_for_retrain,
                "start_minimized": self.start_minimized_var.get(),
                "start_with_windows": self.start_with_windows_var.get(),
                "selected_model": self.selected_model_path  # ADD THIS LINE
            }
        }
        if self.config_manager.save(config):
            self.log(f"Configuration saved to {self.config_manager.config_path}")
            return True
        else:
            self.log("Error saving configuration")
            return False
    
    def _load_model(self):
        """Load model on startup"""
        # Try to load selected model from config, or latest
        model_path = self.selected_model_path
        
        if self.model.load(model_path):
            model_name = os.path.basename(self.model.current_model_path)
            self.model_status_var.set(f"Model: {model_name[:30]}...")
            self.log(f"Model loaded: {model_name}")
            self.status_var.set("Ready - Click 'Start Scanning'")
        else:
            self.log("No trained model found. Please train a model first.")
        
        # Populate model dropdown
        self._refresh_model_list()
        self._update_learning_status()
    
    def _update_learning_status(self):
        self.learning_status_var.set("Yes" if self.enable_self_learning_var.get() else "No")
        counts = self.self_learning.get_sample_counts()
        self.samples_count_var.set(f"Real: {counts['real']} | Fake: {counts['fake']}")
    
    def _clear_training_data(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all collected training data?"):
            if self.self_learning.clear_data():
                self.log("Training data cleared successfully")
                self._update_learning_status()
            else:
                self.log("Error clearing training data")
                messagebox.showerror("Error", "Failed to clear training data")
    
    def _on_monitor_change(self, event=None):
        selection = self.monitor_var.get()
        monitor_names = self.screen_capture.get_monitor_names()
        try:
            self.selected_monitor = monitor_names.index(selection)
            self.screen_capture.set_monitor(self.selected_monitor)
            self.log(f"Monitor changed to: {selection}")
        except ValueError:
            self.log(f"Invalid monitor selection: {selection}")
    
    def _on_closing(self):
        # Exit the program completely
        if self.is_scanning:
            self.stop_scanning()
        self.log("Closing application")
        self.save_config()
        if self.tray_handler:
            try:
                self.tray_handler.stop()
            except Exception:
                pass
        self.root.destroy()

    def _on_minimize(self, event=None):
        try:
            if str(self.root.state()) == 'iconic':
                self.log("Minimized to taskbar — moving to system tray")
                self._hide_to_tray()
        except Exception:
            pass

    def _hide_to_tray(self):
        try:
            # Hide the window
            self.root.withdraw()
            # Start tray icon if available
            if self.tray_handler:
                self.tray_handler.start()
            self.log("Application is running in background (system tray)")
        except Exception as e:
            self.log(f"Error hiding to tray: {e}")

    def _show_from_tray(self):
        try:
            # Stop tray icon and show window
            if self.tray_handler:
                try:
                    self.tray_handler.stop()
                except Exception:
                    pass
            self.root.deiconify()
            self.root.after(0, lambda: self.root.state('normal'))
            self.log("Restored application from system tray")
        except Exception as e:
            self.log(f"Error showing from tray: {e}")

    def _exit_from_tray(self):
        try:
            if self.tray_handler:
                try:
                    self.tray_handler.stop()
                except Exception:
                    pass
            self.log("Exiting application from system tray")
            self.save_config()
            self.root.destroy()
        except Exception as e:
            self.log(f"Error exiting from tray: {e}")
    
    def start_scanning(self):
        if not self.model.model:
            messagebox.showerror("Error", "No model loaded. Please train a model first.")
            return
        self.is_scanning = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("Screen scanning started")
        self.status_var.set("Scanning screen...")
        if self.enable_self_learning_var.get():
            self.self_learning.start_session()
            self.log("Self-learning session started")
        self.had_detection_in_session = False
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
        if self.enable_self_learning_var.get() and self.had_detection_in_session:
            samples_collected = self.self_learning.end_session()
            self.log(f"Session ended: {samples_collected} samples collected")
            self._update_learning_status()
            counts = self.self_learning.get_sample_counts()
            total_samples = counts['real'] + counts['fake']
            if total_samples >= self.min_samples_for_retrain and not self.is_retraining:
                self.log(f"Sufficient samples collected ({total_samples}). Initiating retraining...")
                threading.Thread(target=self._retrain_model, daemon=True).start()
    
    def _update_screen_capture(self):
        if not self.is_scanning:
            return
            
        frame, region_info = self.screen_capture.capture()
        
        if frame is None:
            self.log("Failed to capture screen")
            self.stop_scanning()
            return
            
        self.current_frame = frame.copy()
        self.region_var.set(region_info)
        
        display_frame = frame.copy()
        faces = self.model.detect_faces(frame)
        self.faces_count_var.set(str(len(faces)))
        self.last_detected_faces = faces
        
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
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if self.last_detection_result:
                result_text = "DEEPFAKE" if self.last_detection_result['prediction'] == 1 else "REAL"
                conf = self.last_detection_result['confidence']
                color = (0, 0, 255) if self.last_detection_result['prediction'] == 1 else (0, 255, 0)
                cv2.putText(display_frame, f"{result_text} {conf:.1f}%", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
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
            if result and result['confidence'] >= self.threshold_var.get():
                results.append((result, face_img))
        if results:
            best_result, best_face_img = max(results, key=lambda x: x[0]['confidence'])
            self._update_detection_display(best_result)
            self._update_statistics(best_result)
            self.had_detection_in_session = True
            if self.enable_self_learning_var.get() and self.self_learning.session_active:
                self.self_learning.save_classified_image(best_face_img, best_result['prediction'], best_result['confidence'])
    
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
            # Send a Windows notification if tray handler available
            try:
                if self.tray_handler:
                    self.tray_handler.notify(title="Deepfake detected",
                                              message=f"Confidence: {conf:.1f}%")
            except Exception:
                pass
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
            self.stats_label.config(text=f"Scans: {self.total_scans} | Deepfakes: {self.deepfakes_detected} | Real: {real_count}")
    
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
            cv2.putText(annotated_frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if self.last_detection_result:
                result_text = "DEEPFAKE" if self.last_detection_result['prediction'] == 1 else "REAL"
                conf = self.last_detection_result['confidence']
                color = (0, 0, 255) if self.last_detection_result['prediction'] == 1 else (0, 255, 0)
                cv2.putText(annotated_frame, f"{result_text} {conf:.1f}%", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, annotated_frame)
        messagebox.showinfo("Success", f"Screenshot saved as {filename}")
        self.log(f"Screenshot saved: {filename}")
    
    def _retrain_model(self):
        if self.is_retraining:
            return
        self.is_retraining = True
        self.log("=" * 50)
        self.log("RETRAINING INITIATED")
        self.log("=" * 50)
        self.status_var.set("Retraining model with new data...")
        try:
            real_path = self.self_learning.real_dir
            fake_path = self.self_learning.fake_dir
            result = self.model.train(real_path, fake_path, self.log, archive=True)
            if result:
                self.log(f"✓ Retraining completed! New accuracy: {result:.4f}")
                self.model_status_var.set("Model loaded ✓ (Retrained)")
                self._refresh_model_list()  
                self.self_learning.clear_data()
                self._update_learning_status()
                messagebox.showinfo("Retraining Complete", f"Model retrained successfully!\nNew accuracy: {result:.4f}")
            else:
                self.log("✗ Retraining failed")
        except Exception as e:
            self.log(f"✗ Retraining error: {e}")
        finally:
            self.is_retraining = False
            self.status_var.set("Ready")
            self.log("=" * 50)
    
    def train_model(self, progress_callback=None):
        real_path = self.real_path_var.get()
        fake_path = self.fake_path_var.get()
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
        self.status_var.set("Training in progress...")
        def train_thread():
            result = self.model.train(real_path, fake_path, self.log, progress_callback)
            if result:
                self.model_status_var.set("Model loaded ✓")
                self.status_var.set("Ready - Click 'Start Scanning'")
                self._refresh_model_list()  # ADD THIS LINE
                messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {result:.4f}")
            else:
                self.status_var.set("Training failed")
        threading.Thread(target=train_thread, daemon=True).start()

    def test_model(self, real_path, fake_path):
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
        self.status_var.set("Testing model...")
        try:
            if not self.model.load():
                messagebox.showerror("Error", "No trained model found.")
                return
            Xr, yr = self.model._load_dataset(real_path, 0, None, self.log)
            Xf, yf = self.model._load_dataset(fake_path, 1, None, self.log)
            X = np.array(Xr + Xf)
            y = np.array(yr + yf)
            if X.size == 0:
                messagebox.showerror("Error", "No valid samples loaded.")
                return
            y_pred = self.model.model.predict(X)
            acc = accuracy_score(y, y_pred)
            self.log(f"Test completed — Accuracy: {acc:.4f}")
            messagebox.showinfo("Model Test Results", f"Accuracy: {acc:.4f}")
        except Exception as e:
            self.log(f"✗ Testing error: {e}")
        finally:
            self.status_var.set("Ready")
    def _refresh_model_list(self):
        """Refresh the list of available models"""
        models = self.model.get_available_models()
        if models:
            model_names = [m['name'] for m in models]
            self.model_combo['values'] = model_names
            
            # Select current model or latest
            if self.selected_model_path:
                current_name = os.path.basename(self.selected_model_path)
                if current_name in model_names:
                    self.model_var.set(current_name)
                else:
                    self.model_var.set(model_names[0])
            else:
                self.model_var.set(model_names[0])
            
            self.log(f"Found {len(models)} model(s)")
        else:
            self.model_combo['values'] = ["No models available"]
            self.model_var.set("No models available")
            self.log("No models found in model_archive")

    def _on_model_change(self, event=None):
        """Handle model selection change"""
        selected_name = self.model_var.get()
        if selected_name and selected_name != "No models available":
            model_path = os.path.join(self.model.model_archive_dir, selected_name)
            if os.path.exists(model_path):
                # Stop scanning if active
                was_scanning = self.is_scanning
                if was_scanning:
                    self.stop_scanning()
                
                # Load new model
                if self.model.load(model_path):
                    self.selected_model_path = model_path
                    self.model_status_var.set(f"Model: {selected_name[:30]}...")
                    self.log(f"Switched to model: {selected_name}")
                    self.save_config()
                    
                    # Resume scanning if it was active
                    if was_scanning:
                        self.root.after(500, self.start_scanning)
                else:
                    self.log(f"Failed to load model: {selected_name}")
                    messagebox.showerror("Error", f"Failed to load model: {selected_name}")

def main():
    root = tk.Tk()
    app = ScreenDeepfakeDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()