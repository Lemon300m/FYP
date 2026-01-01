import threading
import sys
import os
from datetime import datetime

# Log errors to file for debugging
LOG_FILE = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__), 'notification_debug.log')

def debug_log(msg):
    try:  
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except:  
        pass

try:
    from pystray import Icon, MenuItem, Menu
    from PIL import Image, ImageDraw
    PYSTRAY_AVAILABLE = True
except Exception as e:
    PYSTRAY_AVAILABLE = False
    debug_log(f"pystray import failed: {e}")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except Exception as e:
    WINSOUND_AVAILABLE = False
    debug_log(f"winsound import failed: {e}")


class TrayHandler:
    def __init__(self, app_name, show_callback=None, exit_callback=None):
        self.app_name = app_name
        self.show_callback = show_callback
        self.exit_callback = exit_callback
        self.icon = None
        self._icon_thread = None
        
        debug_log(f"TrayHandler initialized.  pystray={PYSTRAY_AVAILABLE}, winsound={WINSOUND_AVAILABLE}")

        if PYSTRAY_AVAILABLE:
            try:
                self._create_icon_image()
                menu = Menu(
                    MenuItem('Show', lambda icon, item: self._on_show()),
                    MenuItem('Exit', lambda icon, item: self._on_exit())
                )
                self.icon = Icon('deepfake_detector', self.icon_image, app_name, menu)
                debug_log("pystray icon created successfully")
            except Exception as e:   
                self.icon = None
                debug_log(f"pystray icon creation failed: {e}")

    def _create_icon_image(self):
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((8, 8, 56, 56), fill=(200, 30, 30, 255))
        d.ellipse((20, 20, 44, 44), fill=(255, 255, 255, 255))
        self.icon_image = img

    def start(self):
        if self.icon:
            try:
                self._icon_thread = threading.Thread(target=self.icon.run, daemon=True)
                self._icon_thread.start()
                debug_log("pystray icon started")
            except Exception as e:     
                debug_log(f"pystray start failed: {e}")

    def stop(self):
        if self.icon:
            try:
                self.icon.stop()
                debug_log("pystray icon stopped")
            except Exception as e:      
                debug_log(f"pystray stop failed: {e}")

    def _on_show(self):
        if callable(self.show_callback):
            try:
                self.show_callback()
            except Exception as e:
                debug_log(f"Show callback failed: {e}")

    def _on_exit(self):
        if callable(self.exit_callback):
            try:
                self.exit_callback()
            except Exception as e:  
                debug_log(f"Exit callback failed: {e}")

    def notify(self, title, message, duration=5):
        debug_log(f"ALERT: {title} - {message}")
        
        # Run beep in separate thread so it doesn't block
        if WINSOUND_AVAILABLE: 
            threading.Thread(
                target=self._play_alert, 
                daemon=True
            ).start()
    
    def _play_alert(self):
        try:
            # Play alert beep:  1000Hz for 500ms
            winsound. Beep(1000, 500)
            debug_log("Alert beep played")
        except Exception as e:
            debug_log(f"Alert beep failed:  {e}")