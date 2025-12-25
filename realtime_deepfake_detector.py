import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
from tkinter.scrolledtext import ScrolledText
import cv2
from PIL import Image, ImageTk, ImageGrab
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow. keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
import os
from datetime import datetime
import json
import shutil
import sys
import winreg
from tray_handler import TrayHandler

# Color scheme - Warm & Welcoming theme
THEME_COLORS = {
    # Backgrounds (warm & inviting)
    'bg_dark': '#faf6f1',          # App background (warm cream)
    'bg_light': '#ffffff',         # Panels / cards (pure white)

    # Accents (warm & natural)
    'accent_light': '#f5e6d3',     # Soft peach
    'accent_medium': '#d4845c',    # Warm terracotta
    'accent_dark': '#b85d3b',      # Deep warm brown

    # Text (warm & readable)
    'text_primary': '#3d2817',     # Warm dark brown
    'text_secondary': '#8b7355',   # Warm medium brown

    # Highlights & states
    'highlight': '#d4845c',        # Warm terracotta highlight
    'success': '#5a9d6e',          # Natural green
    'warning': '#d4845c',          # Warm terracotta
    'error': '#a85a47',            # Warm rust red
}


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
        self.window.configure(bg=THEME_COLORS['bg_dark'])
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container with scrollbar
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_container, highlightthickness=0, 
                          background=THEME_COLORS['bg_light'],
                          highlightbackground=THEME_COLORS['accent_dark'])
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
                  font=('Consolas', 16, 'bold')).pack(pady=(0, 20))
        
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
                 font=('Consolas', 9), foreground=THEME_COLORS['text_secondary']).pack(anchor=tk.W, pady=(0, 15))
        
        # Real Dataset
        ttk.Label(section, text="Real Images Dataset:", 
                 font=('Consolas', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        real_frame = ttk.Frame(section)
        real_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.real_path_var = tk.StringVar(value=self.app.real_path_var.get())
        ttk.Entry(real_frame, textvariable=self.real_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(real_frame, text="📁 Browse", 
                  command=lambda: self._browse_dataset("real")).pack(side=tk.LEFT)
        
        # Fake Dataset
        ttk.Label(section, text="Fake Images Dataset:", 
                 font=('Consolas', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        fake_frame = ttk.Frame(section)
        fake_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.fake_path_var = tk.StringVar(value=self.app.fake_path_var.get())
        ttk.Entry(fake_frame, textvariable=self.fake_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(fake_frame, text="📁 Browse", 
                  command=lambda: self._browse_dataset("fake")).pack(side=tk.LEFT)
        
        # Balance Dataset Checkbox
        self.balance_dataset_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="Balance dataset (match sizes of real and fake folders)",
                       variable=self.balance_dataset_var).pack(anchor=tk.W, pady=(0, 15))
        
        ttk.Label(section, text="ℹ️ Equalize number of samples", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(0, 15))
        
        # Train button and progress
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_label_var = tk.StringVar(value="0%")
        
        ttk.Button(section, text="🎯 Train New Model", 
                  command=self._train_model).pack(fill=tk.X, pady=(0, 10))
        
        # Progress text (instead of bar)
        progress_frame = ttk.Frame(section)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.train_progress_label = ttk.Label(progress_frame, textvariable=self. train_progress_label_var, 
                                            font=('Consolas', 11, 'bold'), foreground=THEME_COLORS['success'])
        self.train_progress_label.pack(side=tk. LEFT)
        
        ttk.Button(section, text="🔎 Test Current Model", 
              command=self._test_model).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(section, text="ℹ️ Training may take several minutes depending on dataset size", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(5, 0))

    def _create_general_settings(self, parent):
        section = ttk.LabelFrame(parent, text="General Settings", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        self.auto_start_var = tk.BooleanVar(value=self.app.auto_start_var.get())
        ttk.Checkbutton(section, text="Automatically start scanning when program launches",
                       variable=self.auto_start_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Requires a trained model to be available", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(0, 5))
        
        self.start_minimized_var = tk.BooleanVar(value=self.app.start_minimized_var.get())
        ttk.Checkbutton(section, text="Start minimized to system tray",
                       variable=self.start_minimized_var).pack(anchor=tk.W, pady=5)
        
        self.start_with_windows_var = tk.BooleanVar(value=self.app.start_with_windows_var.get())
        ttk.Checkbutton(section, text="Start application with Windows",
                       variable=self.start_with_windows_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Application will auto-launch when you log in", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(0, 5))
    
    def _create_self_learning_settings(self, parent):
        section = ttk.LabelFrame(parent, text="Self-Learning Settings", padding="15")
        section.pack(fill=tk.X, pady=(0, 15))
        
        self.enable_self_learning_var = tk.BooleanVar(value=self.app.enable_self_learning_var.get())
        ttk.Checkbutton(section, text="Enable automatic model improvement",
                       variable=self.enable_self_learning_var).pack(anchor=tk.W, pady=5)
        
        ttk.Label(section, text="ℹ️ Model will retrain using classified images after each detection session", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(0, 10))
        
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
        ttk.Label(parent, text=title, font=('Consolas', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(parent, text=desc, font=('Consolas', 9), foreground=THEME_COLORS['text_secondary']).pack(anchor=tk.W, pady=(0, 5))
        
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=(0, 15))
        
        scale = ttk.Scale(slider_frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL,
                         command=lambda v: self._snap_slider(var, label, step, suffix))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        label = ttk.Label(slider_frame, text=f"{var.get()}{suffix}", width=6)
        label.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Label(parent, text=warning, font=('Consolas', 8), foreground=THEME_COLORS['warning']).pack(anchor=tk.W)
        return label
    
    def _snap_slider(self, var, label, step, suffix):
        """Snap slider to step intervals"""
        snapped = round(var.get() / step) * step
        var.set(snapped)
        fmt = f"{int(snapped)}{suffix}" if step >= 1 else f"{snapped:.1f}{suffix}"
        label.config(text=fmt)
    
    def _update_progress_label(self, *args):
        """Update the progress percentage label in real-time"""
        progress = self.train_progress_var.get()
        self.train_progress_label_var.set(f"{int(progress)}%")
    
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
        self.app.train_model(progress_callback=self.train_progress_var, balance_dataset=self.balance_dataset_var.get())
        
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
    """Handles CNN model training and prediction using MobileNetV2 transfer learning"""
    
    def __init__(self, model_path="deepfake_model.h5"):
        self.model_archive_dir = get_data_path("model_archive")
        self.model = None
        self.current_model_path = None
        
        try:
            os.makedirs(self.model_archive_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create archive directory: {e}")
        
        self.face_cascade = None
    
        cascade_paths = [
            get_resource_path('haarcascade_frontalface_default.xml'),
            os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
            'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
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

    def _build_cnn_basic(self, input_shape):
        """Build a basic 3-layer CNN"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _build_cnn_mobilenet(self, input_shape):
        """Build MobileNetV2 transfer learning model (RECOMMENDED - Fast & Accurate)"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False  # Freeze base weights for faster training
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),  # Reduced from 256
            Dropout(0.4),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _build_cnn(self, input_shape, use_mobilenet=True):
        """Build CNN model - chooses between basic and MobileNetV2"""
        if use_mobilenet:
            print("Building MobileNetV2 transfer learning model...")
            return self._build_cnn_mobilenet(input_shape)
        else:
            print("Building basic CNN model...")
            return self._build_cnn_basic(input_shape)

    def get_available_models(self):
        """Get list of available model files from model_archive directory"""
        try:
            models = []
            for file in os.listdir(self.model_archive_dir):
                if file.endswith('.h5'):
                    full_path = os.path.join(self.model_archive_dir, file)
                    models.append({
                        'name': file,
                        'path': full_path,
                        'mtime': os.path.getmtime(full_path),
                        'size': os.path.getsize(full_path)
                    })
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
                model_path = self. get_latest_model_path()
            
            if model_path and os.path.exists(model_path):
                self.model = load_model(model_path)
                self.current_model_path = model_path
                print(f"✓ Model loaded successfully:  {os.path.basename(model_path)}")
                return True
            return False
        except Exception as e:
            print(f"Error loading Keras model: {e}")
            return False
    
    def save(self, filename=None):
        """Save model to model_archive directory"""
        if self.model:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cnn_model_{timestamp}.h5"
            
            save_path = os.path.join(self.model_archive_dir, filename)
            self.model.save(save_path)
            self.current_model_path = save_path
            print(f"✓ Model saved:  {save_path}")
            return save_path
        return None

    def detect_faces(self, frame):
        """Detect faces using Haar Cascade"""
        if self.face_cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    
    def preprocess_face(self, img):
        """Preprocess face image for CNN input"""
        try:
            if img is None or img.size == 0:
                return None
            
            # Resize to 96x96 (input size for CNN)
            img = cv2.resize(img, (96, 96))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            return img
        except Exception as e: 
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, face_img):
        if not self.model:
            return None
        
        img = self.preprocess_face(face_img)
        if img is None:
            return None
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Get prediction probability
        prob = float(self.model.predict(img, verbose=0)[0][0])
        
        # Convert to binary prediction (0=Real, 1=Fake)
        prediction = 1 if prob >= 0.5 else 0
        
        # Calculate confidence
        confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
        
        return {'prediction': prediction, 'confidence':  confidence}
    
    def train(self, real_path, fake_path, log_callback, progress_callback=None, balance_dataset=False, archive=True, use_mobilenet=True, augment=True):
        """Train CNN model with data augmentation and optional transfer learning"""
        log_callback("=" * 60)
        log_callback("Starting CNN Training with MobileNetV2 Transfer Learning")
        log_callback("=" * 60)
        
        try:
            # Determine target sample count for balancing BEFORE loading
            if balance_dataset:
                real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                fake_files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '. png', '.bmp'))]
                target_count = min(len(real_files), len(fake_files))
                log_callback(f"Balancing dataset: Using {target_count} samples from each class")
            else:
                target_count = None
            
            # Load datasets with augmentation
            X_real, y_real = self._load_images(real_path, 0, log_callback, balance_dataset, target_count, augment=augment)
            X_fake, y_fake = self._load_images(fake_path, 1, log_callback, balance_dataset, target_count, augment=augment)
            if not X_real or not X_fake:
                log_callback("✗ Error:  No data loaded from datasets")
                return False
            
            # Combine datasets
            X = np.array(X_real + X_fake)
            y = np.array(y_real + y_fake)
            log_callback(f"Total samples: {len(X)} (Real: {len(X_real)}, Fake: {len(X_fake)})")
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            log_callback(f"Train samples:  {len(X_train)}, Test samples: {len(X_test)}")
            
            # Build model
            input_shape = (96, 96, 3)
            self.model = self._build_cnn(input_shape, use_mobilenet=use_mobilenet)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ModelCheckpoint(
                    os.path.join(self.model_archive_dir, "tmp_best_cnn.keras"),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
            ]
            
            # Train model
            log_callback("Training CNN...   (this may take several minutes)")
            log_callback("Starting training loop...")
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, log_callback, total_epochs):
                    self.log_callback = log_callback
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    percent = int((epoch + 1) / self.total_epochs * 100)
                    loss = logs['loss']
                    val_loss = logs['val_loss']
                    acc = logs['accuracy']
                    val_acc = logs['val_accuracy']
                    print(f"\r  Epoch {epoch + 1}/{self.total_epochs} ({percent}%) - Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | Val Acc: {val_acc:.4f}", end='', flush=True)

            callbacks.append(ProgressCallback(log_callback, 15))

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=15,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            print()  # New line after training completes
            log_callback("Training completed!")

            # Evaluate on test set
            test_loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            log_callback(f"✓ Model accuracy on test set: {accuracy:.4f} ({accuracy*100:.2f}%)")
            log_callback(f"  Test loss: {test_loss:.4f}")
            
            # Save model
            saved_path = self.save()
            if saved_path:  
                log_callback(f"✓ Model saved:   {os.path.basename(saved_path)}")
            
            log_callback("=" * 60)
            log_callback("✓ CNN Training completed successfully!")
            log_callback("=" * 60)
            return accuracy
            
        except Exception as e:
            log_callback(f"✗ Training error:  {e}")
            import traceback
            log_callback(traceback.format_exc())
            return False
            
    
    def _load_images(self, path, label, log_callback=None, balance_dataset=False, target_count=None, augment=True):
        """Load and preprocess images with optional data augmentation and balancing"""
        import random
        images, labels = [], []
        
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '. png', '.bmp'))]
        total = len(files)
        
        if log_callback: 
            log_callback(f"Found {total} images in {os.path.basename(path)} dataset")
        
        # Balance dataset by randomly sampling if target_count is specified
        if balance_dataset and target_count is not None and total > target_count:
            if log_callback:
                log_callback(f"  Balancing:  Sampling {target_count} files from {total}")
            files = random.sample(files, target_count)
        
        # Load images
        for idx, file in enumerate(files):
            try:
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                img = self. preprocess_face(img)
                
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    
                    # Data augmentation:  flip horizontally
                    if augment:  
                        img_flipped = np.fliplr(img)
                        images.append(img_flipped)
                        labels. append(label)
                        
                        # Optional: slight brightness variation
                        img_bright = np.clip(img * 1.1, 0, 1)
                        images.append(img_bright)
                        labels.append(label)
                
                # Single line progress update (overwrites previous line)
                percent = int((idx + 1) / len(files) * 100)
                print(f"\r  Loading {os.path.basename(path)}: {idx + 1}/{len(files)} ({percent}%)", end='', flush=True)
                            
            except Exception as e: 
                if log_callback:
                    log_callback(f"  Warning: Could not load {file}: {e}")
                continue
        
        print()  # New line after loading completes
        
        if log_callback:
            aug_text = " (with augmentation x3)" if augment else ""
            log_callback(f"✓ Loaded {len(images)} samples from {os.path.basename(path)}{aug_text}")
        
        return images, labels
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
        self.root.geometry("675x700")
        self.root.resizable(True, True)
        self._setup_theme()
        
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
        
        self.log_expanded = False
        self.log_window = None
        self.log_buffer = []  # Buffer to store log messages when window is closed
        
        # Initialize learning status variables (used by separate window)
        self.learning_status_var = tk.StringVar(value="Yes" if self.enable_self_learning_var.get() else "No")
        self.samples_count_var = tk.StringVar(value="Real: 0 | Fake: 0")
        
        # Model status variable (tracked internally, not displayed in main window)
        self.model_status_var = tk.StringVar(value="No model loaded")
        self.faces_count_var = tk.StringVar(value="0")
        self.region_var = tk.StringVar(value="Not capturing")
        
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
    
    def _setup_theme(self):
        """Configure the theme with bluish-purple colors"""
        style = ttk.Style()
        
        # Configure colors
        bg = THEME_COLORS['bg_dark']
        fg = THEME_COLORS['text_primary']
        accent = THEME_COLORS['accent_medium']
        light_accent = THEME_COLORS['highlight']
        
        # Root window background
        self.root.configure(bg=bg)
        
        # TkDefaultFont colors
        style.theme_use('clam')
        
        # Configure main frame
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('TLabelframe', background=bg, foreground=fg, borderwidth=1)
        style.configure('TLabelframe.Label', background=bg, foreground=light_accent)
        
        # Configure buttons
        style.configure('TButton', background=THEME_COLORS['accent_light'], 
                       foreground=fg, padding=8, borderwidth=1)
        style.map('TButton',
                 foreground=[('active', THEME_COLORS['text_primary']),
                           ('pressed', THEME_COLORS['text_secondary']),
                           ('disabled', THEME_COLORS['text_secondary'])],
                 background=[('active', THEME_COLORS['highlight']),
                           ('pressed', THEME_COLORS['accent_dark']),
                           ('disabled', THEME_COLORS['accent_dark'])])
        
        # Accent button style for primary actions
        style.configure('Accent.TButton', background=THEME_COLORS['highlight'],
                       foreground=fg, padding=8, borderwidth=1)
        style.map('Accent.TButton',
                 foreground=[('active', THEME_COLORS['text_primary']),
                           ('pressed', THEME_COLORS['text_secondary']),
                           ('disabled', THEME_COLORS['text_secondary'])],
                 background=[('active', THEME_COLORS['success']),
                           ('pressed', THEME_COLORS['accent_dark']),
                           ('disabled', THEME_COLORS['accent_dark'])])
        
        # Configure comboboxes
        style.configure('TCombobox', fieldbackground=THEME_COLORS['accent_light'],
                       background=THEME_COLORS['accent_light'], foreground=fg)
        style.map('TCombobox',
                 fieldbackground=[('readonly', THEME_COLORS['accent_light']),
                                ('active', THEME_COLORS['highlight']),
                                ('focus', THEME_COLORS['highlight'])],
                 background=[('readonly', THEME_COLORS['accent_light']),
                           ('active', THEME_COLORS['highlight']),
                           ('focus', THEME_COLORS['highlight'])])
        
        # Configure scrollbars
        style.configure('Vertical.TScrollbar', background=THEME_COLORS['accent_light'],
                       troughcolor=THEME_COLORS['bg_light'], bordercolor=THEME_COLORS['accent_dark'],
                       arrowcolor=fg)
        
        # Custom style for status label
        style.configure('Status.TLabel', background=bg, foreground=THEME_COLORS['text_secondary'],
                       font=('Consolas', 9))

        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        ttk.Label(main_frame, text="🖥️ Screen Deepfake Detection System", 
                  font=('Consolas', 18, 'bold')).grid(row=0, column=0, columnspan=1, pady=10)
        
        self._setup_left_column(main_frame)
        self._setup_bottom_panel(main_frame)
        
        self.status_var = tk.StringVar(value="Ready - Train a model to begin")
        ttk.Label(main_frame, textvariable=self.status_var, 
                  relief=tk.SUNKEN, anchor=tk.W, style='Status.TLabel').grid(row=3, column=0, 
                                                      sticky=(tk.W, tk.E), pady=(10, 0))
    
    def _setup_left_column(self, parent):
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        
        controls = ttk.Frame(left_frame)
        controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls.columnconfigure(1, weight=1)  # Make model selector expandable
        
        self.start_button = ttk.Button(controls, text="▶", 
                                       command=self.start_scanning, style='Accent.TButton', width=3)
        self.start_button.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.start_button, "Start Scanning")
        
        self.stop_button = ttk.Button(controls, text="⏹", 
                                      command=self.stop_scanning, state='disabled', width=3)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.stop_button, "Stop Scanning")
        
        ttk.Label(controls, text="Model:").pack(side=tk.LEFT, padx=(15, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, 
                                        width=20, state='readonly')
        self.model_combo.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        ttk.Button(controls, text="🔄", command=self._refresh_model_list, 
                width=3).pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.model_combo, "Select a trained model")
        
        settings_btn = ttk.Button(controls, text="⚙️", width=3,
                                 command=lambda: SettingsWindow(self.root, self))
        settings_btn.pack(side=tk.RIGHT, padx=2)
        self._create_tooltip(settings_btn, "Settings")

        video_frame = ttk.LabelFrame(left_frame, text="Screen Capture Feed", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, 
                                     text="Screen capture will appear here\n\nClick 'Start Scanning' to begin", 
                                     background=THEME_COLORS['bg_light'], 
                                     foreground=THEME_COLORS['text_secondary'], 
                                     font=('Consolas', 14))
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        result_frame = ttk.Frame(left_frame)
        result_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.result_label = ttk.Label(result_frame, text="No Detection", 
                                      font=('Consolas', 24, 'bold'), foreground='gray')
        self.result_label.pack()
        self.confidence_label = ttk.Label(result_frame, text="", font=('Consolas', 14))
        self.confidence_label.pack()
        
        self.stats_label = ttk.Label(left_frame, text="Scans: 0 | Deepfakes: 0 | Real: 0", 
                                     font=('Consolas', 10), foreground=THEME_COLORS['success'])
        self.stats_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _create_tooltip(self, widget, text):
        """Create a tooltip that appears on hover"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background=THEME_COLORS['accent_medium'],
                           foreground=THEME_COLORS['text_primary'], 
                           relief=tk.SOLID, borderwidth=1, font=('Consolas', 9))
            label.pack(padx=4, pady=2)
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
    def _setup_bottom_panel(self, parent):
        """Setup bottom quick action buttons"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(bottom_frame, text="Quick Actions:", font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        # Save screenshot button
        screenshot_btn = ttk.Button(bottom_frame, text="📸", width=3,
                                   command=self._save_screenshot)
        screenshot_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(screenshot_btn, "Save Screenshot")
        
        # Reset statistics button
        reset_btn = ttk.Button(bottom_frame, text="↻", width=3,
                              command=self._reset_statistics)
        reset_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(reset_btn, "Reset Statistics")
        
        # Activity log button
        log_btn = ttk.Button(bottom_frame, text="📋", width=3,
                            command=self._toggle_log_window)
        log_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(log_btn, "Activity Log")
                
        # Monitor selection
        ttk.Label(bottom_frame, text="Monitor:").pack(side=tk.LEFT, padx=(10, 5))
        self.monitor_var = tk.StringVar(value="All Screens")
        monitor_options = self.screen_capture.get_monitor_names()
        monitor_combo = ttk.Combobox(bottom_frame, textvariable=self.monitor_var, 
                                    values=monitor_options, width=15, state='readonly')
        monitor_combo.pack(side=tk.LEFT, padx=(0, 10))
        monitor_combo.bind('<<ComboboxSelected>>', self._on_monitor_change)
            
    def _show_learning_window(self):
        """Show self-learning status in a separate window"""
        window = Toplevel(self.root)
        window.title("Self-Learning Status")
        window.geometry("400x300")
        window.resizable(False, False)
        window.transient(self.root)
        
        # Apply theme to window
        window.configure(bg=THEME_COLORS['bg_dark'])
        
        frame = ttk.Frame(window, padding="15")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Self-Learning Status", 
                 font=('Consolas', 14, 'bold')).pack(pady=(0, 10))
        
        ttk.Label(frame, text="Enabled:").pack(anchor=tk.W)
        self.learning_status_var = tk.StringVar(value="Yes" if self.enable_self_learning_var.get() else "No")
        ttk.Label(frame, textvariable=self.learning_status_var,
                 font=('Consolas', 12)).pack(anchor=tk.W, padx=20)
        
        ttk.Label(frame, text="Collected Samples:", font=('Consolas', 11)).pack(anchor=tk.W, pady=(15, 0))
        self.samples_count_var = tk.StringVar(value="Real: 0 | Fake: 0")
        ttk.Label(frame, textvariable=self.samples_count_var,
                 font=('Consolas', 11)).pack(anchor=tk.W, padx=20)
        
        ttk.Button(frame, text="🗑️ Clear Training Data", 
                  command=self._clear_training_data).pack(fill=tk.X, pady=(20, 5))
        
        ttk.Button(frame, text="↻ Retrain with Collected Data",
                  command=self._retrain_model).pack(fill=tk.X, pady=5)
        
        self._update_learning_status()
    
    def _toggle_log_window(self):
        """Toggle activity log window"""
        if self.log_window is None or not self.log_window.winfo_exists():
            self.log_window = Toplevel(self.root)
            self.log_window.title("Activity Log")
            self.log_window.geometry("500x400")
            self.log_window.resizable(True, True)
            self.log_window.transient(self.root)
            
            # Apply theme
            self.log_window.configure(bg=THEME_COLORS['bg_dark'])
            
            frame = ttk.Frame(self.log_window, padding="5")
            frame.pack(fill=tk.BOTH, expand=True)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            
            self.log_text = ScrolledText(frame, wrap=tk.WORD, 
                                        background=THEME_COLORS['bg_light'],
                                        foreground=THEME_COLORS['text_primary'],
                                        insertbackground=THEME_COLORS['text_primary'])
            self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            for msg in self.log_buffer:
                self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            
            self.log_expanded = True
        else:
            self.log_window.destroy()
            self.log_expanded = False
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.log_buffer.append(log_msg)
        
        if self.log_window is not None and self.log_window.winfo_exists():
            if hasattr(self, 'log_text') and self.log_text.winfo_exists():
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
        self.result_label.config(text="Scanning Stopped", foreground=THEME_COLORS['text_secondary'])
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
        
        # Get available space from the label widget
        self.video_label.update_idletasks()
        available_width = self.video_label.winfo_width()
        available_height = self.video_label.winfo_height()
        
        # Use default if not yet rendered
        if available_width <= 1:
            available_width = 750
        if available_height <= 1:
            available_height = 420
        
        # Calculate dimensions that maintain aspect ratio and fit in available space
        if available_width / available_height > aspect_ratio:
            # Height is the limiting factor
            display_height = available_height
            display_width = int(display_height * aspect_ratio)
        else:
            # Width is the limiting factor
            display_width = available_width
            display_height = int(display_width / aspect_ratio)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
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
            self.result_label.config(text="⚠ Uncertain", foreground=THEME_COLORS['warning'])
            self.confidence_label.config(text=f"Confidence too low: {conf:.1f}%")
        elif result['prediction'] == 1:
            self.result_label.config(text="🚨 DEEPFAKE", foreground=THEME_COLORS['error'])
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
            self.result_label.config(text="✓ REAL", foreground=THEME_COLORS['success'])
            self.confidence_label.config(text=f"Confidence: {conf:.1f}%")
    
    def _reset_detection_display(self):
        self.result_label.config(text="No Detection", foreground=THEME_COLORS['text_secondary'])
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
        
        screenshot_dir = get_data_path("screenshot")
        try:
            os.makedirs(screenshot_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create screenshot folder: {e}")
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
        filepath = os.path.join(screenshot_dir, filename)
        cv2.imwrite(filepath, annotated_frame)
        messagebox.showinfo("Success", f"Screenshot saved to screenshot/{filename}")
        self.log(f"Screenshot saved: screenshot/{filename}")
    
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
            result = self.model.train(real_path, fake_path, self. log, balance_dataset=True, augment=True)
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
    
    def train_model(self, progress_callback=None, balance_dataset=False):
        real_path = self.real_path_var.get()
        fake_path = self.fake_path_var.get()
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
        self.status_var.set("Training in progress...")
        def train_thread():
            result = self.model.train(real_path, fake_path, self.log, progress_callback, balance_dataset=balance_dataset)
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
            
            # Load test images
            X_real, y_real = self.model._load_images(real_path, 0, self.log, augment=False)
            X_fake, y_fake = self.model._load_images(fake_path, 1, self.log, augment=False)
            
            X = np.array(X_real + X_fake)
            y = np.array(y_real + y_fake)
            
            if X.size == 0:
                messagebox.showerror("Error", "No valid samples loaded.")
                return
            
            # Evaluate
            _, accuracy = self.model.model.evaluate(X, y, verbose=0)
            self.log(f"✓ Test completed — Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            messagebox.showinfo("Model Test Results", f"Accuracy: {accuracy:.4f}\n({accuracy*100:.2f}%)")
        except Exception as e:
            self.log(f"✗ Testing error: {e}")
            messagebox.showerror("Error", f"Testing failed: {e}")
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