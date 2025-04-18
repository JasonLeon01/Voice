import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Resemblyzer'))
import tkinter as tk
from typing import List
import VoiceWatch
from scenes.panel import Panel
from scenes.panel1 import Panel1
from scenes.panel2 import Panel2
from scenes.panel3 import Panel3
from scenes.panel4 import Panel4
from scenes.panel5 import Panel5
from scenes.panel6 import Panel6

class App:
    def __init__(self, root, size):
        self.running = True
        self.root = root
        self.root.title('Voice Recognition System')
        self.esc_pressed = False
        self.root.bind("<Escape>", self.on_escape_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.listener = VoiceWatch.Listener(lambda text: root.after(0, self.response, text))

        Panel.init_data()
        self.panels: List[Panel] = [Panel1(self, size), Panel2(self, size), Panel5(self, size), Panel6(self, size)]
        self.airflight = Panel3(self, size)
        self.seat = Panel4(self, size)
        for i in range(len(self.panels) - 1):
            self.panels[i].next_panel = self.panels[(i + 1) % len(self.panels)]

        self.current_panel: Panel = self.panels[0]

        self.start_listening()
        self.current_panel.show()

    def on_escape_press(self, event):
        self.esc_pressed = True

    def start_listening(self):
        if not self.running:
            return
        self.listener.start_listening()

    def stop_listening(self):
        self.listener.stop_listening()

    def response(self, text=None):
        if text is not None and text != "":
            self.current_panel.get_text(text, False)
        else:
            self.current_panel.get_text("", True)
            self.start_listening()

    def on_closing(self):
        self.running = False
        if hasattr(self, 'listener'):
            self.listener.stop_listening()  # 停止监听
        self.root.destroy()  # 销毁窗口

def main():
    root = tk.Tk()
    size = (800, 600)
    root.geometry(f"{size[0]}x{size[1]}")
    app = App(root, size)
    root.mainloop()

if __name__ == '__main__':
    main()