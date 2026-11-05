"""Experimental camera, sharpness plot and keyboard jog workspace."""
from __future__ import annotations

import argparse
from collections import deque
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

from autofocus_v2.capture import FrameStream
from autofocus_v2.models import ROI
from camera_opencv import BACKENDS, cv2, startup_camera_settings
from camera_preview import _score_frame, _tune_exposure
from stage_controller import StageController, StageControllerError
from z_jog import JogWindow


class EmbeddedJog(JogWindow):
    def __init__(self, root, status, controller, target):
        self.target = target
        super().__init__(root, status, controller, 1, 10)
        root.unbind('<Button-1>')
        root.bind('<KeyPress>', self._press_in_panel)
        root.bind('<KeyRelease>', self._release_in_panel)
        root.bind('<FocusOut>', self._focus_out)

    def _focus_out(self, event):
        if event.widget is self.target:
            self._set_armed(False)
            for timer in self._releases.values():
                self.root.after_cancel(timer)
            self._releases.clear()
            self._held.clear()

    def _set_armed(self, armed):
        self._armed = armed
        if armed:
            self.target.focus_set()
        if self._move is None and not self._cancel.is_set() and not self._closing:
            self.status.set('已启用点动，每次按键移动一步' if armed else '松开移动键，再点击启用点动')

    def _press_in_panel(self, event):
        if event.keysym.lower() in ('space', 'escape'):
            self._request_stop()
            return 'break'
        if self.root.focus_get() is self.target:
            return self._key_press(event)
        return None

    def _release_in_panel(self, event):
        if self.root.focus_get() is self.target:
            return self._key_release(event)
        return None


class TrialApp:
    def __init__(self, root):
        self.root = root
        self.stream = FrameStream()
        self.roi = None
        self.shape = None
        self.display_size = None
        self.drag_start = None
        self.last_sequence = 0
        self.exposure_status = ''
        self.history = deque(maxlen=180)
        self.tile_mask = None
        self.jog = None
        self.controller = None
        self.connection_queue = queue.Queue()
        self.connection_lock = threading.Lock()
        self.closing = threading.Event()
        self.connecting = False
        self.camera_error = None
        self.photo = None
        self.root.title('探针台试验 · 相机与点动')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._build()
        self.root.after(50, self._tick)

    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        self.camera_index = tk.StringVar(value='0')
        self.backend = tk.StringVar(value='DSHOW' if sys.platform == 'win32' else 'ANY')
        ttk.Label(top, text='相机编号').pack(side='left')
        ttk.Entry(top, textvariable=self.camera_index, width=4).pack(side='left', padx=4)
        ttk.Combobox(top, textvariable=self.backend, values=tuple(BACKENDS), state='readonly', width=7).pack(side='left')
        self.camera_button = ttk.Button(top, text='打开相机', command=self.open_camera)
        self.camera_button.pack(side='left', padx=5)
        ttk.Button(top, text='关闭相机', command=self.close_camera).pack(side='left')
        ttk.Label(top, text='控制器串口').pack(side='left', padx=(18, 3))
        self.port = tk.StringVar()
        ttk.Entry(top, textvariable=self.port, width=14).pack(side='left')
        self.connect_button = ttk.Button(top, text='连接控制器', command=self.connect)
        self.connect_button.pack(side='left', padx=5)

        content = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        content.pack(fill='both', expand=True)
        left = ttk.Frame(content)
        left.pack(side='left', fill='both', expand=True)
        ttk.Label(left, text='相机预览 · 在画面上拖动选择样品区域').pack(anchor='w')
        self.preview = tk.Canvas(left, width=720, height=480, bg='#151a21', highlightthickness=0)
        self.preview.pack(fill='both', expand=True)
        self.preview.bind('<ButtonPress-1>', self._drag_start)
        self.preview.bind('<B1-Motion>', self._drag_move)
        self.preview.bind('<ButtonRelease-1>', self._drag_end)
        self.camera_status = tk.StringVar(value='相机未连接')
        ttk.Label(left, textvariable=self.camera_status, wraplength=710).pack(anchor='w')
        right = ttk.Frame(content, width=360)
        right.pack(side='left', fill='y', padx=(12, 0))
        ttk.Label(right, text='样品区域 V2 清晰度', font=('', 13, 'bold')).pack(anchor='w')
        self.score_status = tk.StringVar(value='等待画面')
        ttk.Label(right, textvariable=self.score_status).pack(anchor='w', pady=(4, 0))
        self.plot = tk.Canvas(right, width=350, height=170, bg='#f5f7f9', highlightthickness=1,
                              highlightbackground='#d2d7dd')
        self.plot.pack(pady=(4, 4))
        ttk.Label(right, text='横轴：最近采样时间 · 纵轴：V2 分数', wraplength=350).pack(anchor='w')
        ttk.Separator(right).pack(fill='x', pady=12)
        ttk.Label(right, text='键盘点动', font=('', 13, 'bold')).pack(anchor='w')
        ttk.Label(right, text='W / S   Y− / Y+\nA / D   X+ / X−\nQ / E   Z− / Z+\n↑ / ↓   Z+ / Z−\n\nSpace / Esc   软件急停\nX   退出', justify='left').pack(anchor='w', pady=6)
        ttk.Label(right, text='默认 1 脉冲 / 10% 速度。先松开移动键，再点击启用点动。', wraplength=350).pack(anchor='w')
        self.jog_target = tk.Label(right, text='启用点动 / 键盘焦点', bg='#dbe8f3', padx=15, pady=10,
                                   takefocus=True, relief='groove')
        self.jog_target.pack(fill='x', pady=(8, 4))
        self.jog_target.bind('<Button-1>', self._arm)
        self.motor_status = tk.StringVar(value='控制器未连接')
        ttk.Label(right, textvariable=self.motor_status, wraplength=350).pack(anchor='w')
        ttk.Button(right, text='软件急停', command=self.stop).pack(fill='x', pady=(8, 0))
        ttk.Label(right, text='软件急停作用于所有轴，请确保物理急停可用。', wraplength=350).pack(anchor='w')

    def _arm(self, _event=None):
        if self.jog is not None and not self.jog._cancel.is_set():
            self.jog._set_armed(True)

    def open_camera(self):
        if self.stream.running:
            self.camera_status.set('相机正在运行或关闭，请稍候')
            return
        try:
            index = int(self.camera_index.get())
            backend = self.backend.get().strip().upper()
            if index < 0 or backend not in BACKENDS:
                raise ValueError('请输入非负相机编号和可用后端')
        except ValueError as exc:
            self.camera_status.set(str(exc))
            return
        self.roi = None
        self.shape = None
        self.display_size = None
        self.last_sequence = 0
        self.history.clear()
        self.tile_mask = None
        self._draw_plot()
        self.camera_error = None
        self.camera_status.set('正在打开相机并调整曝光，请保持样品静止')
        self.camera_button.configure(state='disabled')

        def prepare(camera, stop):
            self.exposure_status = _tune_exposure(
                camera, stop, None, float(startup_camera_settings(backend)['exposure']),
            )

        self.stream.start(index, backend, prepare=prepare)

    def close_camera(self):
        if self.stream.running:
            self.stream.stop(timeout=0)
            self.camera_status.set('正在关闭相机…')
        self.history.clear()
        self.tile_mask = None
        self.last_sequence = 0
        self.preview.delete('all')
        self.display_size = None
        self.score_status.set('相机已关闭')
        self._draw_plot()

    def connect(self):
        if self.connecting or self.controller is not None:
            return
        port = self.port.get().strip()
        if not port:
            self.motor_status.set('请输入已确认的本机控制器串口')
            return
        self.connecting = True
        self.connect_button.configure(state='disabled')
        self.motor_status.set('正在连接控制器…')

        def worker():
            controller = StageController(port=port, motion_timeout=5.0,
                                         invert_x=False, invert_y=True, invert_z=False)
            try:
                controller.open()
                if not controller.test_connection():
                    raise StageControllerError('控制器握手响应不匹配')
            except Exception as exc:
                controller.close()
                with self.connection_lock:
                    if not self.closing.is_set():
                        self.connection_queue.put((None, str(exc)))
                return
            with self.connection_lock:
                if self.closing.is_set():
                    controller.close()
                else:
                    self.connection_queue.put((controller, None))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.jog is not None:
            self.jog._request_stop()

    def _drag_start(self, event):
        if self.display_size is not None:
            self.drag_start = (event.x, event.y)

    def _drag_move(self, event):
        if self.drag_start is None:
            return
        self.preview.delete('drag')
        self.preview.create_rectangle(*self.drag_start, event.x, event.y, outline='#ffcf4a', tags='drag')

    def _drag_end(self, event):
        if self.drag_start is None or self.display_size is None or self.shape is None:
            return
        dw, dh = self.display_size
        height, width = self.shape
        x1, x2 = sorted((max(0, min(dw, self.drag_start[0])), max(0, min(dw, event.x))))
        y1, y2 = sorted((max(0, min(dh, self.drag_start[1])), max(0, min(dh, event.y))))
        self.drag_start = None
        self.preview.delete('drag')
        x0, x3 = int(x1 * width / dw), int(x2 * width / dw)
        y0, y3 = int(y1 * height / dh), int(y2 * height / dh)
        try:
            self.roi = ROI(x0, y0, x3 - x0, y3 - y0)
        except ValueError:
            self.camera_status.set('选择区域至少需要 16 × 16 个原图像素')
            return
        self.history.clear()
        self.tile_mask = None
        self._draw_plot()
        self._draw_roi()

    def _draw_roi(self):
        self.preview.delete('selection')
        if self.roi is None or self.display_size is None or self.shape is None:
            return
        dw, dh = self.display_size
        height, width = self.shape
        r = self.roi
        self.preview.create_rectangle(r.x * dw / width, r.y * dh / height,
                                      (r.x + r.width) * dw / width, (r.y + r.height) * dh / height,
                                      outline='#6be191', width=2, tags='selection')

    def _draw_plot(self):
        self.plot.delete('all')
        self.plot.create_line(40, 12, 40, 145, 340, 145, fill='#8c98a4')
        valid = [value for _, value in self.history if value is not None]
        if not valid:
            return
        low, high = min(valid), max(valid)
        span = max(high - low, abs(high) * 0.02, 1e-12)
        upper, lower = high + span * 0.05, max(0.0, low - span * 0.05)
        duration = max(self.history[-1][0] - self.history[0][0], 0.001)
        self.plot.create_text(39, 5, text=f'{upper:.3g}', anchor='ne', fill='#53616e')
        self.plot.create_text(39, 145, text=f'{lower:.3g}', anchor='se', fill='#53616e')
        points = []
        for when, value in self.history:
            if value is None:
                if len(points) >= 4:
                    self.plot.create_line(*points, fill='#168c6b', width=2)
                points = []
                continue
            points.extend((40 + 300 * (when - self.history[0][0]) / duration,
                           145 - 133 * (value - lower) / (upper - lower)))
        if len(points) >= 4:
            self.plot.create_line(*points, fill='#168c6b', width=2)

    def _show_frame(self, frame):
        height, width = frame.image.shape[:2]
        if self.shape != (height, width):
            self.shape = (height, width)
            self.roi = ROI(width // 3, height // 3, width - 2 * (width // 3), height - 2 * (height // 3))
            self.history.clear()
            self.tile_mask = None
        result = _score_frame(frame.image, self.roi)
        mask = tuple(tile.valid for tile in result.tiles)
        if self.tile_mask is not None and mask != self.tile_mask:
            self.history.clear()
        self.tile_mask = mask
        score = result.score if result.valid and math.isfinite(result.score) else None
        self.history.append((frame.received_at, score))
        self.score_status.set(f'V2 {score:.4g} · {sum(mask)}/4 区块有效' if score is not None else '曝光无效，分数不可用')
        self._draw_plot()
        scale = min(max(1, self.preview.winfo_width()) / width, max(1, self.preview.winfo_height()) / height)
        dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
        image = cv2.resize(frame.image, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB)
        ppm = f'P6\n{dw} {dh}\n255\n'.encode() + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.root, data=ppm, format='PPM')
        self.preview.delete('frame')
        self.preview.create_image(0, 0, anchor='nw', image=self.photo, tags='frame')
        self.display_size = (dw, dh)
        self._draw_roi()

    def _tick(self):
        if self.closing.is_set():
            return
        try:
            controller, error = self.connection_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.connecting = False
            if error:
                self.motor_status.set(f'连接失败：{error}')
                self.connect_button.configure(state='normal')
            else:
                self.controller = controller
                self.jog = EmbeddedJog(self.root, self.motor_status, controller, self.jog_target)
                self.root.protocol('WM_DELETE_WINDOW', self.close)
                self.motor_status.set('已连接。松开移动键，再点击启用点动')
                self.jog._poll()
        if self.stream.error and self.camera_error != self.stream.error:
            self.camera_error = self.stream.error
            self.camera_status.set(f'相机错误：{self.camera_error}')
        if not self.stream.running:
            self.camera_button.configure(state='normal')
        frame = self.stream.latest()
        if self.stream.active and frame is not None and frame.sequence != self.last_sequence:
            try:
                self._show_frame(frame)
                self.last_sequence = frame.sequence
                self.camera_status.set(self.exposure_status or '相机已打开')
            except (ValueError, cv2.error, tk.TclError) as exc:
                self.camera_status.set(f'画面错误：{exc}')
                self.stream.stop(timeout=0)
        self.root.after(50, self._tick)

    def close(self):
        with self.connection_lock:
            self.closing.set()
            while True:
                try:
                    controller, _error = self.connection_queue.get_nowait()
                except queue.Empty:
                    break
                if controller is not None:
                    controller.close()
        self.stream.stop(timeout=0)
        if self.jog is not None:
            self.jog._request_exit()
        else:
            self.root.quit()

    def finish(self):
        self.stream.stop(timeout=0.1)
        if self.jog is not None:
            self.jog._workers.shutdown(wait=True)
        if self.controller is not None:
            self.controller.close()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if cv2 is None:
        print('请安装 opencv-python', file=sys.stderr)
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f'需要 Tk 图形桌面：{exc}', file=sys.stderr)
        return 1
    app = TrialApp(root)
    try:
        root.mainloop()
    finally:
        app.close()
        app.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
