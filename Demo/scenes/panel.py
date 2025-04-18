import os
import tkinter as tk
from tkinter import PhotoImage
import LLM

class Panel(tk.Frame):
    def __init__(self, parent, size, image_path=None):
        self.parent = parent
        width, height = size
        super().__init__(parent.root, width=width, height=height)
        self.voie_text = None
        self.next_panel = None

        self.bg_image = PhotoImage(file=image_path)
        self.bg_label = tk.Label(master=self, image=self.bg_image)
        self.bg_label.place(x=0, y=0)

        self.fg_label = tk.Label(
            master=self,
            text="",
            font=("Arial", 24),
            wraplength=width - 100,  # 设置换行宽度，留出左右边距
            justify="left"           # 换行后文字左对齐
        )
        self.fg_label.place(x=50, y=50)
        self._update_id = None

    @classmethod
    def init_data(cls):
        cls.date = "2025-02-04"
        cls.belonging_flight = {
            "flight": [
                {
                    "id": "Air-Beijing-Shanghai-2025-03-05-10-25-00",
                    "class": "economy"
                },
                {
                    "id": "Air-Chongqing-Guangzhou-2025-02-05-19-45-00",
                    "class": "first"
                }
            ]
        }

        cls.flight_seat = {
            "Air-Beijing-Shanghai-2025-03-05-10-25-00": {
                "first": [
                    [False, True],
                    [False, True],
                    [False, True],
                    [False, False]
                ],
                "business": [
                    [True, True, False, False],
                    [False, False, True, True],
                    [True, True, True, False],
                    [False, True, True, True],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, False, False],
                    [True, False, False, True]
                ],
                "economy": [
                    [True, False, True, True, False, True],
                    [True, False, True, False, True, False],
                    [True, False, True, False, True, False],
                    [True, True, False, False, True, False],
                    [True, True, True, False, False, True],
                    [False, True, True, False, False, True],
                    [True, False, True, False, True, True],
                    [False, False, True, False, True, True],
                    [False, True, False, False, False, True],
                    [True, True, True, False, True, False],
                    [True, True, False, False, False, True],
                    [True, True, False, True, False, True],
                    [True, False, False, True, False, True],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, True],
                    [False, False, True, False, True, True],
                    [False, True, False, False, True, False],
                    [False, False, True, True, True, False],
                    [True, True, False, False, False, True],
                    [False, False, False, True, False, True],
                    [True, True, True, True, False, False],
                    [False, True, True, True, True, True],
                    [True, True, True, True, False, False],
                    [True, False, True, False, True, False],
                    [False, True, True, True, True, True],
                    [True, False, True, True, False, True],
                    [False, True, True, False, False, False],
                    [True, True, True, False, True, True],
                    [True, False, False, True, False, False],
                    [False, True, True, True, False, True]
                ]
            },
            "Air-Chongqing-Guangzhou-2025-02-05-19-45-00": {
                "first": [
                    [False, False],
                    [True, False],
                    [True, False],
                    [True, False]
                ],
                "business": [
                    [False, False, False, False],
                    [True, False, False, False],
                    [False, True, True, False],
                    [True, False, False, True],
                    [True, True, True, True],
                    [True, False, True, True],
                    [False, False, False, False],
                    [False, True, True, True]
                ],
                "economy": [
                    [True, True, True, False, False, False],
                    [False, True, True, True, False, True],
                    [False, True, True, False, False, True],
                    [True, False, False, False, False, False],
                    [True, False, True, False, False, False],
                    [False, True, True, True, False, True],
                    [True, False, True, True, True, True],
                    [True, False, True, False, True, False],
                    [True, False, False, False, True, True],
                    [False, False, True, True, True, True],
                    [False, True, False, True, False, True],
                    [False, True, False, True, True, True],
                    [False, True, False, False, True, False],
                    [True, True, False, False, False, False],
                    [True, False, False, True, False, True],
                    [True, True, True, False, True, True],
                    [True, True, False, False, True, True],
                    [True, False, True, False, False, True],
                    [True, True, True, False, False, False],
                    [False, False, False, True, False, False],
                    [True, False, False, True, True, False],
                    [True, False, True, False, False, False],
                    [True, False, False, True, False, True],
                    [True, False, False, True, True, True],
                    [True, False, True, False, True, True],
                    [True, False, False, True, True, True],
                    [True, False, True, False, False, True],
                    [True, False, False, False, False, False],
                    [False, False, True, False, True, False],
                    [False, True, False, False, True, False],
                ]
            }
        }

        cls.user_info = {
            "flight": "None",
            "seat": "None"
        }

    def get_text(self, text, NPS):
        if NPS == False:
            self.fg_label.config(text=text)
        self.voie_text = text

    def clear_text(self):
        self.voie_text = None
        self.fg_label.config(text="")

    def update(self):
        # 基类内容，后面通过super调用
        if self.parent.current_panel != self:
            self._update_id = -1
            return
        if self._update_id == -1:
            return
        else:
            self._update_id = self.after(16, self.update)

    def show(self):
        self.place(x=0, y=0)
        if self._update_id is None:
            self.update()

    def hide(self):
        self.place_forget()
        if self._update_id:
            self.after_cancel(self._update_id)
            self._update_id = -1

    def next_node(self):
        self.hide()
        self.parent.current_panel = self.next_panel
        self.parent.current_panel.show()

    def change_to_node(self, index):
        self.hide()
        self.parent.stop_listening()
        self.clear_text()
        self.parent.current_panel = self.parent.panels[index]
        self.parent.current_panel.show()

    def change_to_node_by_name(self, name):
        self.hide()
        self.parent.stop_listening()
        self.clear_text()
        self.parent.current_panel = eval(f"self.parent.{name}")
        self.parent.current_panel.show()

    def get_LLM_answer(self, date, belonging_flight, user_info, dialogue):
        return LLM.LLM.get_flight_info(date, belonging_flight, user_info, dialogue)