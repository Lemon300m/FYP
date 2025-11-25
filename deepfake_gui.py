import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import sys
import os

# Import the model (assumes it's in the same directory or installed)
# You may need to adjust the import based on your file structure
try:
    from deepfake_detector import DeepfakeDetector
except ImportError:
    # If running as standalone, you can include the class here
    print("Note: Import deepfake_detector module or include it in the same directory")


class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detector")
        self.root.geometry("700x650")
        self.root.resizable(True, True)
        
        self.detector = None
        self.real_path = ""
        self.fake_path = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Deepfake Detection System", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Dataset Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="Dataset Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        config_frame.columnconfigure(1, weight=1)
        
        # Real dataset path
        ttk.Label(config_frame, text="Real Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.real_path_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.real_path_var, width=40).grid(row=0, column=1, 
                                                                                  sticky=(tk.W, tk.E), padx=5)
        ttk.Button(config_frame, text="Browse", command=lambda: self.browse_folder("real")).grid(row=0, column=2)
        
        # Fake dataset path
        ttk.Label(config_frame, text="Fake Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.fake_path_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.fake_path_var, width=40).grid(row=1, column=1, 
                                                                                  sticky=(tk.W, tk.E), padx=5)
        ttk.Button(config_frame, text="Browse", command=lambda: self.browse_folder("fake")).grid(row=1, column=2)
        
        # Initialize button
        ttk.Button(config_frame, text="Initialize Detector", 
                  command=self.initialize_detector).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Progress Bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                            maximum=100, mode='determinate', length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready", foreground='blue')
        self.progress_label.grid(row=1, column=0, sticky=tk.W, padx=5)
        
        # Training Section
        train_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        train_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(train_frame, text="Train New Model", 
                  command=self.train_model, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="Maintain Model", 
                  command=self.maintain_model, width=20).pack(side=tk.LEFT, padx=5)
        
        # Identification Section
        identify_frame = ttk.LabelFrame(main_frame, text="Image Identification", padding="10")
        identify_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        identify_frame.columnconfigure(1, weight=1)
        
        ttk.Label(identify_frame, text="Image Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.image_path_var = tk.StringVar()
        ttk.Entry(identify_frame, textvariable=self.image_path_var, width=40).grid(row=0, column=1, 
                                                                                     sticky=(tk.W, tk.E), padx=5)
        ttk.Button(identify_frame, text="Browse", command=self.browse_image).grid(row=0, column=2)
        
        ttk.Button(identify_frame, text="Identify Image", 
                  command=self.identify_image).grid(row=1, column=0, columnspan=3, pady=10)
        
        # Result Display
        self.result_var = tk.StringVar(value="No results yet")
        result_label = ttk.Label(identify_frame, textvariable=self.result_var, 
                                font=('Arial', 12, 'bold'), foreground='blue')
        result_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Log Output Section
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(5, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=12, width=70, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def progress_callback(self, message, percentage):
        """Callback for progress updates from detector"""
        self.log(message)
        if percentage is not None:
            self.progress_var.set(percentage)
            self.progress_label.config(text=f"{message} - {percentage:.1f}%")
        else:
            self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def browse_folder(self, dataset_type):
        folder = filedialog.askdirectory(title=f"Select {dataset_type.capitalize()} Dataset Folder")
        if folder:
            if dataset_type == "real":
                self.real_path_var.set(folder)
            else:
                self.fake_path_var.set(folder)
    
    def browse_image(self):
        file = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file:
            self.image_path_var.set(file)
    
    def initialize_detector(self):
        real_path = self.real_path_var.get()
        fake_path = self.fake_path_var.get()
        
        if not real_path or not fake_path:
            messagebox.showerror("Error", "Please specify both dataset paths")
            return
        
        try:
            self.detector = DeepfakeDetector(real_path, fake_path, self.progress_callback)
            self.log("Detector initialized successfully")
            self.status_var.set("Detector initialized")
            self.progress_label.config(text="Detector initialized")
            messagebox.showinfo("Success", "Detector initialized successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector: {str(e)}")
            self.log(f"Error: {str(e)}")
    
    def train_model(self):
        if self.detector is None:
            messagebox.showerror("Error", "Please initialize detector first")
            return
        
        self.log("Starting model training...")
        self.status_var.set("Training in progress...")
        self.progress_var.set(0)
        self.progress_label.config(text="Starting training...")
        
        def train_thread():
            try:
                accuracy = self.detector.train()
                if accuracy is not None:
                    self.log(f"Training completed! Accuracy: {accuracy:.4f}")
                    self.status_var.set(f"Training completed - Accuracy: {accuracy:.4f}")
                    self.progress_label.config(text=f"Training complete - Accuracy: {accuracy:.4f}")
                    messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.4f}")
            except Exception as e:
                self.log(f"Training error: {str(e)}")
                self.status_var.set("Training failed")
                self.progress_label.config(text="Training failed")
                self.progress_var.set(0)
                messagebox.showerror("Error", f"Training failed: {str(e)}")
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def maintain_model(self):
        if self.detector is None:
            messagebox.showerror("Error", "Please initialize detector first")
            return
        
        # Ask for new data path
        new_path = filedialog.askdirectory(title="Select New Training Data Folder")
        if not new_path:
            return
        
        # Ask for label
        label_window = tk.Toplevel(self.root)
        label_window.title("Data Label")
        label_window.geometry("300x150")
        
        ttk.Label(label_window, text="Is this data Real or Fake?", 
                 font=('Arial', 10)).pack(pady=20)
        
        label_var = tk.IntVar(value=0)
        ttk.Radiobutton(label_window, text="Real (0)", variable=label_var, value=0).pack()
        ttk.Radiobutton(label_window, text="Fake (1)", variable=label_var, value=1).pack()
        
        def confirm_maintain():
            label = label_var.get()
            label_window.destroy()
            
            self.log(f"Starting model maintenance with label={label}...")
            self.status_var.set("Maintenance in progress...")
            self.progress_var.set(0)
            self.progress_label.config(text="Starting maintenance...")
            
            def maintain_thread():
                try:
                    success = self.detector.maintain(new_path, label)
                    if success:
                        self.log("Model updated successfully!")
                        self.status_var.set("Model updated")
                        self.progress_label.config(text="Model updated successfully")
                        messagebox.showinfo("Success", "Model updated successfully!")
                    else:
                        self.log("Old model retained (better performance)")
                        self.status_var.set("Old model retained")
                        self.progress_label.config(text="Old model retained")
                        messagebox.showinfo("Info", "Old model retained (better performance)")
                except Exception as e:
                    self.log(f"Maintenance error: {str(e)}")
                    self.status_var.set("Maintenance failed")
                    self.progress_label.config(text="Maintenance failed")
                    self.progress_var.set(0)
                    messagebox.showerror("Error", f"Maintenance failed: {str(e)}")
            
            threading.Thread(target=maintain_thread, daemon=True).start()
        
        ttk.Button(label_window, text="Confirm", command=confirm_maintain).pack(pady=10)
    
    def identify_image(self):
        if self.detector is None:
            messagebox.showerror("Error", "Please initialize detector first")
            return
        
        image_path = self.image_path_var.get()
        if not image_path:
            messagebox.showerror("Error", "Please select an image")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Image file not found")
            return
        
        self.log(f"Identifying image: {image_path}")
        self.status_var.set("Identifying...")
        
        def identify_thread():
            try:
                result = self.detector.identify(image_path)
                if result:
                    result_text = f"{result['result']} (Confidence: {result['confidence']:.2f}%)"
                    self.result_var.set(result_text)
                    self.log(f"Result: {result_text}")
                    self.status_var.set("Identification complete")
                    
                    # Change color based on result
                    color = 'red' if result['result'] == 'DEEPFAKE' else 'green'
                    for widget in self.root.winfo_children():
                        if isinstance(widget, ttk.Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, ttk.LabelFrame) and child.cget('text') == 'Image Identification':
                                    for label in child.winfo_children():
                                        if isinstance(label, ttk.Label) and 'result' in str(label.cget('textvariable')):
                                            label.configure(foreground=color)
                else:
                    self.log("Failed to identify image")
                    self.status_var.set("Identification failed")
            except Exception as e:
                self.log(f"Identification error: {str(e)}")
                self.status_var.set("Identification failed")
                messagebox.showerror("Error", f"Identification failed: {str(e)}")
        
        threading.Thread(target=identify_thread, daemon=True).start()
    
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()


def main():
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()