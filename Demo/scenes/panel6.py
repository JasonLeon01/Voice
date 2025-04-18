import tkinter as tk
from scenes.panel import Panel

class Panel6(Panel):
    def __init__(self, parent, size):
        self.count = 0
        super().__init__(parent, size, image_path="bg/5.png")
        self.shown = False

    def update(self):
        if not self.shown:
            self.shown = True
            print(self.user_info)
        super().update()