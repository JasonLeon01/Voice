import tkinter as tk
from scenes.panel import Panel

class Panel4(Panel):
    def __init__(self, parent, size):
        self.count = 0
        super().__init__(parent, size, image_path="bg/3.png")
        self.shown = False

    def update(self):
        if not self.shown:
            self.shown = True
            print(self.belonging_flight)
        if self.parent.esc_pressed:
            self.parent.esc_pressed = False
            self.shown = False
            self.change_to_node(1)
            self.parent.start_listening()
            return
        super().update()
