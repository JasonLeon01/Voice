import tkinter as tk
from scenes.panel import Panel

class Panel2(Panel):
    def __init__(self, parent, size):
        self.count = 0
        super().__init__(parent, size, image_path="")

        # 航班信息显示
        self.flight_label = tk.Label(self, text="当前选择航班：None", font=('Arial', 12))
        self.flight_label.place(relx=0.5, rely=0.3, anchor='center')

        # 座位信息显示
        self.seat_label = tk.Label(self, text="当前选择座位：None", font=('Arial', 12))
        self.seat_label.place(relx=0.5, rely=0.4, anchor='center')

        # 查看航班按钮
        self.view_flights_btn = tk.Button(self, text="查看名下航班", font=('Arial', 10),
                                        command=lambda: self.change_to_node_by_name("airflight"))
        self.view_flights_btn.place(relx=0.5, rely=0.5, anchor='center')

        # 查看座位按钮
        self.view_seats_btn = tk.Button(self, text="查看当前航班座位", font=('Arial', 10),
                                        command=lambda: self.change_to_node_by_name("seat"))
        self.view_seats_btn.place(relx=0.5, rely=0.6, anchor='center')

    def update(self):
        self.parent.esc_pressed = False
        if self.user_info["flight"] != "None" and self.user_info["seat"] != "None":
            self.parent.stop_listening()
            self.next_node()
            return

        if self.voie_text is not None and self.voie_text != "":
            try:
                result = self.get_LLM_answer(Panel.date, Panel.belonging_flight, Panel.user_info, self.voie_text)
                result = result.replace(" ", "").replace("[", "").replace("]", "").replace("'", "").replace("\"", "")
                flight_id, seat_type = result.split(",")
                print(flight_id, seat_type)

                if flight_id != "None":
                    has_it = False
                    for flight in Panel.belonging_flight["flight"]:
                        if flight["id"] == flight_id:
                            has_it = True
                            break
                    if has_it:
                        Panel.user_info["flight"] = flight_id
                        Panel.user_info["seat"] = "None"
                        self.seat_label.config(text=f"当前选择座位：None")
                    else:
                        print("没有此航班")
                    self.flight_label.config(text=f"当前选择航班：{Panel.user_info['flight']}")

                if seat_type != "None":
                    if Panel.user_info["flight"] == "None":
                        print("请先选择航班")
                    else:
                        if seat_type == "window-random":
                            print("靠窗")
                            seat = self.window_random()
                            if seat is None:
                                print("座位不可用")
                            else:
                                self.user_info["seat"] = seat
                            self.seat_label.config(text=f"当前选择座位：{self.user_info['seat']}")
                        elif seat_type == "aisle-random":
                            print("靠走廊")
                            seat = self.aisle_random()
                            if seat is None:
                                print("座位不可用")
                            else:
                                self.user_info["seat"] = seat
                            self.seat_label.config(text=f"当前选择座位：{self.user_info['seat']}")
                        else:
                            X, Y = seat_type.split("-")
                            print(X, Y)
                            if Panel.flight_seat[flight_id][seat_class][int(X)-1][ord(Y)-65]:
                                self.user_info["seat"] = f"{X}-{Y}"
                            else:
                                print("座位不可用")
                            self.seat_label.config(text=f"当前选择座位：{X}-{Y}")
            except Exception as e:
                print(e)

            self.voie_text = None
            self.parent.start_listening()
        super().update()

    def aisle_random(self):
        # 获取当前航班的座位信息
        flight_id = Panel.user_info["flight"]
        if flight_id == "None":
            return None

        # 获取当前航班的座位布局
        for flight in Panel.belonging_flight["flight"]:
            if flight["id"] == flight_id:
                seat_class = flight["class"]
                break
        else:
            return None

        seats = Panel.flight_seat[flight_id][seat_class]
        available_seats = []

        # 根据不同舱位类型确定走廊位置
        if seat_class == "first":  # 2列
            aisle_cols = [0, 1]  # 第2列是走廊
        elif seat_class == "business":  # 4列
            aisle_cols = [1, 2]  # 第2、3列是走廊
        else:  # economy 6列
            aisle_cols = [2, 3]  # 第3、4列是走廊

        # 收集所有可用的走廊座位
        for row in range(len(seats)):
            for col in aisle_cols:
                if seats[row][col]:  # 如果座位可用
                    seat_num = f"{row+1}-{chr(65+col)}"  # 转换为座位号格式 (如 1-A)
                    available_seats.append(seat_num)

        # 随机选择一个可用座位
        if available_seats:
            import random
            Panel.user_info["seat"] = random.choice(available_seats)
            return Panel.user_info["seat"]
        return None

    def window_random(self):
        # 获取当前航班的座位信息
        flight_id = Panel.user_info["flight"]
        if flight_id == "None":
            return None

        # 获取当前航班的座位布局
        for flight in Panel.belonging_flight["flight"]:
            if flight["id"] == flight_id:
                seat_class = flight["class"]
                break
        else:
            return None

        seats = Panel.flight_seat[flight_id][seat_class]
        available_seats = []

        # 根据不同舱位类型确定窗户位置
        if seat_class == "first":  # 2列
            window_cols = [0, 1]  # 第1列是窗户
        elif seat_class == "business":  # 4列
            window_cols = [0, 3]  # 第1、4列是窗户
        else:  # economy 6列
            window_cols = [0, 5]  # 第1、6列是窗户

        # 收集所有可用的窗户座位
        for row in range(len(seats)):
            for col in window_cols:
                if seats[row][col]:  # 如果座位可用
                    seat_num = f"{row+1}-{chr(65+col)}"  # 转换为座位号格式 (如 1-A)
                    available_seats.append(seat_num)

        # 随机选择一个可用座位
        if available_seats:
            import random
            Panel.user_info["seat"] = random.choice(available_seats)
            return Panel.user_info["seat"]
        return None
