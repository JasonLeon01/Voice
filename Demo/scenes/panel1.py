import re
import tkinter as tk
from scenes.panel import Panel

class Panel1(Panel):
    def __init__(self, parent, size):
        self.count = 0
        super().__init__(parent, size, image_path="bg/1.png")

    def update(self):
        self.parent.esc_pressed = False
        if self.voie_text is not None and self.voie_text != "":
            if ("你好" in self.voie_text and "小助手" in self.voie_text) or ("Hi" in self.voie_text and "Assistant" in self.voie_text):
                self.parent.listener.record_embedding = True
                self.voie_text = None
                self.clear_text()
                self.next_node()
                self.parent.start_listening()
                return
            self.voie_text = None
            self.parent.start_listening()

        super().update()
