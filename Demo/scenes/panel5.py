import tkinter as tk
from scenes.panel import Panel

class Panel5(Panel):
    def __init__(self, parent, size):
        self.count = 0
        super().__init__(parent, size, image_path="bg/4.png")

    def update(self):
        if self.count >= 100:
            self.count = 0
            if self.next_panel:
                self.next_node()
        self.count += 1
        super().update()