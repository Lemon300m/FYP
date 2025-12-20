import threading
try:
    from pystray import Icon, MenuItem, Menu
    from PIL import Image, ImageDraw
    PYSTRAY_AVAILABLE = True
except Exception:
    PYSTRAY_AVAILABLE = False

try:
    from win10toast import ToastNotifier
    WIN10TOAST_AVAILABLE = True
except Exception:
    WIN10TOAST_AVAILABLE = False


class TrayHandler:
    """Simple system tray handler using pystray and win10toast (Windows).
    If libraries are missing, methods gracefully degrade to prints.
    """

    def __init__(self, app_name, show_callback=None, exit_callback=None):
        self.app_name = app_name
        self.show_callback = show_callback
        self.exit_callback = exit_callback
        self.icon = None
        self._icon_thread = None
        self.notifier = None

        if PYSTRAY_AVAILABLE:
            try:
                self._create_icon_image()
                menu = Menu(
                    MenuItem('Show', lambda icon, item: self._on_show()),
                    MenuItem('Exit', lambda icon, item: self._on_exit())
                )
                self.icon = Icon('deepfake_detector', self.icon_image, app_name, menu)
            except Exception:
                self.icon = None

        if WIN10TOAST_AVAILABLE:
            try:
                self.notifier = ToastNotifier()
            except Exception:
                self.notifier = None

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
            except Exception:
                pass

    def stop(self):
        if self.icon:
            try:
                self.icon.stop()
            except Exception:
                pass

    def _on_show(self):
        if callable(self.show_callback):
            try:
                self.show_callback()
            except Exception:
                pass

    def _on_exit(self):
        if callable(self.exit_callback):
            try:
                self.exit_callback()
            except Exception:
                pass

    def notify(self, title, message, duration=5):
        if self.notifier:
            try:
                self.notifier.show_toast(title, message, duration=duration, threaded=True)
            except Exception:
                pass
        else:
            print(f"NOTIFY: {title} - {message}")
