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
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_data_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Running as compiled executable - use executable's directory
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script - use script's directory
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

def create_icon_button(parent, text, command, width=None, is_accent=False, tooltip=None):
    # Split emoji and text
    parts = text.split(' ', 1)
    emoji = parts[0] if parts else ''
    label = parts[1] if len(parts) > 1 else ''
    
    # Create button with larger emoji but normal text size
    if label:
        # For buttons with emoji + text, create on separate lines with emoji larger
        display_text = emoji + '\n' + label
        btn = tk.Button(parent, text=display_text, command=command, 
                       font=('Segoe UI Emoji', 12),  # Larger for emoji, but not too big
                       bg=THEME_COLORS['highlight'] if is_accent else THEME_COLORS['accent_light'],
                       fg=THEME_COLORS['text_primary'],
                       relief=tk.RAISED, borderwidth=1,
                       activebackground=THEME_COLORS['success'] if is_accent else THEME_COLORS['highlight'],
                       activeforeground=THEME_COLORS['text_primary'],
                       padx=5, pady=3)
    else:
        # For icon-only buttons, use larger font
        btn = tk.Button(parent, text=emoji, command=command, 
                       font=('Segoe UI Emoji', 14, 'bold'),
                       bg=THEME_COLORS['highlight'] if is_accent else THEME_COLORS['accent_light'],
                       fg=THEME_COLORS['text_primary'],
                       relief=tk.RAISED, borderwidth=1,
                       activebackground=THEME_COLORS['success'] if is_accent else THEME_COLORS['highlight'],
                       activeforeground=THEME_COLORS['text_primary'])
    
    if width:
        btn.config(width=width)
    
    return btn

def create_tooltip(widget, text):
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

# For multi-monitor support
try:
    from screeninfo import get_monitors
    MULTI_MONITOR_SUPPORT = True
except ImportError:
    MULTI_MONITOR_SUPPORT = False
    print("screeninfo not installed. Multi-monitor support disabled.")

class SettingsWindow:
    
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
        create_icon_button(real_frame, "📁 Browse", 
                  lambda: self._browse_dataset("real")).pack(side=tk.LEFT)
        
        # Fake Dataset
        ttk.Label(section, text="Fake Images Dataset:", 
                 font=('Consolas', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        fake_frame = ttk.Frame(section)
        fake_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.fake_path_var = tk.StringVar(value=self.app.fake_path_var.get())
        ttk.Entry(fake_frame, textvariable=self.fake_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        create_icon_button(fake_frame, "📁 Browse", 
                  lambda: self._browse_dataset("fake")).pack(side=tk.LEFT)
        
        # Balance Dataset Checkbox
        self.balance_dataset_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="Balance dataset (match sizes of real and fake folders)",
                       variable=self.balance_dataset_var).pack(anchor=tk.W, pady=(0, 15))
        
        ttk.Label(section, text="ℹ️ Equalize number of samples", 
                 font=('Consolas', 8), foreground=THEME_COLORS['accent_medium']).pack(anchor=tk.W, pady=(0, 15))
        
        # Train button and progress
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_label_var = tk.StringVar(value="0%")
        
        create_icon_button(section, "🎯 Train New Model", 
                  self._train_model).pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar with percentage label
        progress_frame = ttk.Frame(section)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.train_progress = ttk.Progressbar(progress_frame, variable=self.train_progress_var, 
                                             maximum=100, mode='determinate')
        self.train_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        progress_label = ttk.Label(progress_frame, textvariable=self.train_progress_label_var, 
                                  font=('Consolas', 10, 'bold'), width=5)
        progress_label.pack(side=tk.RIGHT)
        
        # Bind progress var to update label
        self.train_progress_var.trace('w', self._update_progress_label)
        
        create_icon_button(section, "🔎 Test Current Model", 
              self._test_model).pack(fill=tk.X, pady=(0, 10))
        
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
        snapped = round(var.get() / step) * step
        var.set(snapped)
        if suffix == "%":
            fmt = f"{int(snapped)}{suffix}"
        elif step >= 1:
            fmt = f"{int(snapped)}{suffix}"
        else:
            fmt = f"{snapped:.1f}{suffix}"
        label.config(text=fmt)
    
    def _update_progress_label(self, *args):
        progress = self.train_progress_var.get()
        self.train_progress_label_var.set(f"{int(progress)}%")
    
    def _update_windows_startup(self, enable):
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
        models = self.get_available_models()
        if models:
            return models[0]['path']
        return None
            
    def load(self, model_path=None):
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
        
        img = cv2.resize(img, (96, 96))  # Reduced from 128x128 for faster processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Faster histogram with 4 bins instead of 8
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [4], [0, 256])
            features.extend(hist.flatten())
        
        # Laplacian variance for texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([laplacian.mean(), laplacian.std(), laplacian.var()])
        
        # Edge detection stats
        edges = cv2.Canny(gray, 100, 200)
        features.extend([edges.mean(), edges.std()])
        
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
    
    def train(self, real_path, fake_path, log_callback, progress_callback=None, balance_dataset=False, archive=True):
        log_callback("Starting model training...")
        progress_var = progress_callback if progress_callback else tk.DoubleVar()
        
        try:
            # Models are automatically saved to model_archive with timestamps
            
            # If balancing is enabled, pre-calculate target sample count
            target_samples = None
            if balance_dataset:
                real_files = [f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                fake_files = [f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                target_samples = min(len(real_files), len(fake_files))
                log_callback(f"Balance enabled: Will load {target_samples} samples from each class")
            
            log_callback("Loading real images...")
            X_real, y_real = self._load_dataset(real_path, 0, progress_var, log_callback, balance_dataset, target_samples)
            log_callback("Loading fake images...")
            X_fake, y_fake = self._load_dataset(fake_path, 1, progress_var, log_callback, balance_dataset, target_samples)
            
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
    
    def _load_dataset(self, path, label, progress_var, log_callback, balance_dataset=False, target_samples=None, progress_callback=None):
        import random
        X, y = [], []
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total = len(files)
        
        # If balancing is enabled and target_samples is set, randomly select files to load
        if balance_dataset and target_samples is not None and total > target_samples:
            files = random.sample(files, target_samples)
            total = len(files)
        
        dataset_name = os.path.basename(path)
        
        for idx, file in enumerate(files):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                features = self.extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
            
            # Update progress every file
            progress_pct = int((idx + 1) / total * 100)
            if progress_var:
                progress_var.set(progress_pct * 0.4)
            
            # Log progress frequently with carriage return to update same line
            if (idx + 1) % max(1, total // 200) == 0:  # ~200 updates per dataset
                print(f"\rLoading {dataset_name}: {idx + 1}/{total} files ({progress_pct}%)", end='', flush=True)
        
        print()  # New line after progress completes
        log_callback(f"Loaded {len(X)} valid samples from {dataset_name}")
        return X, y

class SelfLearningManager:    
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
    def __init__(self):
        self.monitors = self._detect_monitors()
        self.selected_monitor = 0
        
    def _detect_monitors(self):
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
        return [m["name"] for m in self.monitors]
    
    def set_monitor(self, index):
        if 0 <= index < len(self.monitors):
            self.selected_monitor = index
            return True
        return False
    
    def capture(self):
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
        self.is_in_tray = False
        
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
            self.root.bind('<Unmap>', self._on_unmap)
        except Exception:
            pass
    
    def _setup_theme(self):
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
        
        self.toggle_button = create_icon_button(controls, "▶", self.toggle_scanning, width=3, is_accent=True)
        self.toggle_button.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.toggle_button, "Start Scanning")
        
        ttk.Label(controls, text="Model:").pack(side=tk.LEFT, padx=(15, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, 
                                        width=20, state='readonly')
        self.model_combo.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        refresh_btn = create_icon_button(controls, "🔄", self._refresh_model_list, width=3)
        refresh_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(self.model_combo, "Select a trained model")
        
        settings_btn = create_icon_button(controls, "⚙️", 
                                         lambda: SettingsWindow(self.root, self), width=3)
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
        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(bottom_frame, text="Quick Actions:", font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        # Save screenshot button
        screenshot_btn = create_icon_button(bottom_frame, "📸", self._save_screenshot, width=3)
        screenshot_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(screenshot_btn, "Save Screenshot")
        
        # Reset statistics button
        reset_btn = create_icon_button(bottom_frame, "↻", self._reset_statistics, width=3)
        reset_btn.pack(side=tk.LEFT, padx=2)
        self._create_tooltip(reset_btn, "Reset Statistics")
        
        # Activity log button
        log_btn = create_icon_button(bottom_frame, "📋", self._toggle_log_window, width=3)
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
        
        create_icon_button(frame, "🗑️ Clear Training Data", 
                  self._clear_training_data).pack(fill=tk.X, pady=(20, 5))
        
        ttk.Button(frame, text="↻ Retrain with Collected Data",
                  command=self._retrain_model).pack(fill=tk.X, pady=5)
        
        self._update_learning_status()
    
    def _toggle_log_window(self):
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
        try:
            if self.is_scanning:
                self.stop_scanning()
            
            self.log("Closing application")
            self.save_config()
            
            # Stop tray if running
            if self.tray_handler:
                try:
                    self.tray_handler.stop()
                except Exception:
                    pass
            
            self.is_in_tray = False
            self.root.quit()
        except Exception as e:
            self.log(f"Error during close: {e}")
            self.root.quit()

    def _on_unmap(self, event):
        # Only hide to tray if the event is from the root window being minimized
        if event.widget == self.root:
            # Small delay to check actual state
            self.root.after(100, self._check_minimize_state)

    def _check_minimize_state(self):
        try:
            if self.root.state() == 'iconic' and not self.is_in_tray:
                self.log("Minimized to taskbar — moving to system tray")
                self._hide_to_tray()
        except Exception as e:
            self.log(f"Error checking minimize state: {e}")

    def _hide_to_tray(self):
        try:
            if self.is_in_tray:
                return  # Already in tray
            
            self.root.withdraw()  # Hide the window
            self.is_in_tray = True
            
            # Start tray icon if available
            if self.tray_handler:
                self.tray_handler.start()
                self.log("Application is running in background (system tray)")
            else:
                self.log("Application hidden (tray handler not available)")
        except Exception as e:
            self.log(f"Error hiding to tray: {e}")
            self.is_in_tray = False

    def _show_from_tray(self):
        try:
            if not self.is_in_tray:
                return  # Not in tray
            
            # Stop tray icon first
            if self.tray_handler:
                try:
                    self.tray_handler.stop()
                except Exception as e:
                    self.log(f"Warning stopping tray: {e}")
            
            # Show and restore window
            self.root.deiconify()
            self.root.state('normal')
            self.root.lift()
            self.root.focus_force()
            
            self.is_in_tray = False
            self.log("Restored application from system tray")
        except Exception as e:
            self.log(f"Error showing from tray: {e}")
            self.is_in_tray = False

    def _exit_from_tray(self):
        try:
            self.is_in_tray = False
            if self.tray_handler:
                try:
                    self.tray_handler.stop()
                except Exception:
                    pass
            self.log("Exiting application from system tray")
            self.save_config()
            self.root.quit()  # Use quit() instead of destroy()
        except Exception as e:
            self.log(f"Error exiting from tray: {e}")
    
    def toggle_scanning(self):
        if self.is_scanning:
            self.stop_scanning()
        else:
            self.start_scanning()
    
    def start_scanning(self):
        if not self.model.model:
            messagebox.showerror("Error", "No model loaded. Please train a model first.")
            return
        self.is_scanning = True
        self.toggle_button.config(text="⏹")
        self._create_tooltip(self.toggle_button, "Stop Scanning")
        self.log("Screen scanning started")
        self.status_var.set("Scanning screen...")
        if self.enable_self_learning_var.get():
            self.self_learning.start_session()
            self.log("Self-learning session started")
        self.had_detection_in_session = False
        self._update_screen_capture()
    
    def stop_scanning(self):
        self.is_scanning = False
        self.toggle_button.config(text="▶")
        self._create_tooltip(self.toggle_button, "Start Scanning")
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
        self.log("Starting model test...")
        try:
            if not self.model.load():
                messagebox.showerror("Error", "No trained model found.")
                return
            self.log("Loading real test images...")
            Xr, yr = self.model._load_dataset(real_path, 0, None, self.log)
            self.log("Loading fake test images...")
            Xf, yf = self.model._load_dataset(fake_path, 1, None, self.log)
            X = np.array(Xr + Xf)
            y = np.array(yr + yf)
            if X.size == 0:
                messagebox.showerror("Error", "No valid samples loaded.")
                return
            self.log(f"Testing on {len(X)} samples (Real: {len(Xr)}, Fake: {len(Xf)})...")
            y_pred = self.model.model.predict(X)
            acc = accuracy_score(y, y_pred)
            self.log(f"✓ Test completed — Accuracy: {acc:.4f}")
            messagebox.showinfo("Model Test Results", f"Accuracy: {acc:.4f}")
        except Exception as e:
            self.log(f"✗ Testing error: {e}")
        finally:
            self.status_var.set("Ready")

    def _refresh_model_list(self):
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