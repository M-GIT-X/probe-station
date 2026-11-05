"""Experimental camera, sharpness plot and keyboard jog workspace."""
from __future__ import annotations

import argparse
from collections import deque
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from autofocus_v2.autofocus import AutofocusConfig, run_autofocus
from autofocus_v2.capture import FrameStream
from autofocus_v2.models import ROI
from camera_opencv import BACKENDS, cv2, startup_camera_settings
from camera_preview import _score_frame, _tune_exposure
from stage_controller import StageController, StageControllerError
from z_jog import JogWindow


class EmbeddedJog(JogWindow):
    def __init__(self, root, status, controller, target, *, blocked=None, on_stop=None, on_exit=None):
        self.target = target
        self.blocked = blocked or (lambda: False)
        self.on_stop = on_stop
        self.on_exit = on_exit
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
        self._armed = armed and not self.blocked()
        if self._armed:
            self.target.focus_set()
        if self._move is None and not self._cancel.is_set() and not self._closing:
            self.status.set('已启用点动，每次按键移动一步' if self._armed else '松开移动键，再点击启用点动')

    def _request_stop(self):
        if self.on_stop is not None:
            self.on_stop()
        super()._request_stop()

    def _request_exit(self):
        if self.on_exit is not None:
            self.on_exit()
        else:
            super()._request_exit()

    def _press_in_panel(self, event):
        if event.keysym.lower() in ('space', 'escape'):
            self._request_stop()
            return 'break'
        if self.root.focus_get() is self.target:
            if self.blocked() and event.keysym.lower() != 'x':
                self._set_armed(False)
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
        self.connection_thread = None
        self.af_running = False
        self.af_cancel = threading.Event()
        self.af_queue = queue.Queue()
        self.af_thread = None
        self.camera_error = None
        self.photo = None
        self.root.title('探针台 V2 试验 · 相机、点动与自动对焦')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._build()
        self.root.after(50, self._tick)

    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        self.camera_index = tk.StringVar(value='0')
        self.backend = tk.StringVar(value='DSHOW' if sys.platform == 'win32' else 'ANY')
        ttk.Label(top, text='相机编号').pack(side='left')
        self.camera_entry = ttk.Entry(top, textvariable=self.camera_index, width=4)
        self.camera_entry.pack(side='left', padx=4)
        self.backend_combo = ttk.Combobox(top, textvariable=self.backend, values=tuple(BACKENDS), state='readonly', width=7)
        self.backend_combo.pack(side='left')
        self.camera_button = ttk.Button(top, text='打开相机', command=self.open_camera)
        self.camera_button.pack(side='left', padx=5)
        self.close_camera_button = ttk.Button(top, text='关闭相机', command=self.close_camera)
        self.close_camera_button.pack(side='left')
        ttk.Label(top, text='控制器串口').pack(side='left', padx=(18, 3))
        self.port = tk.StringVar()
        self.port_entry = ttk.Entry(top, textvariable=self.port, width=14)
        self.port_entry.pack(side='left')
        self.connect_button = ttk.Button(top, text='连接控制器', command=self.connect)
        self.connect_button.pack(side='left', padx=5)

        autofocus = ttk.LabelFrame(self.root, text='Z 轴自动对焦 · 单位为脉冲', padding=6)
        autofocus.pack(fill='x', padx=8, pady=(0, 8))
        self.af_values = {}
        self.af_entries = []
        for column, (name, label, default) in enumerate((
            ('half_range', '半范围 ±', '20'), ('coarse_step', '粗扫步长', '5'),
            ('fine_step', '细扫步长', '1'), ('speed', '速度 %', '10'),
            ('settle_seconds', '停稳秒数', '0.5'), ('frames', '每点帧数', '5'),
        )):
            ttk.Label(autofocus, text=label).grid(row=0, column=column, sticky='w', padx=4)
            variable = tk.StringVar(value=default)
            entry = ttk.Entry(autofocus, textvariable=variable, width=9)
            entry.grid(row=1, column=column, padx=4, sticky='w')
            self.af_values[name] = variable
            self.af_entries.append(entry)
        self.af_button = ttk.Button(autofocus, text='开始自动对焦', command=self.start_autofocus)
        self.af_button.grid(row=0, column=6, rowspan=2, padx=12)
        self.af_status = tk.StringVar(value='先确认手动方向、选择样品区域，再启动有界 Z 扫描。')
        ttk.Label(autofocus, textvariable=self.af_status, wraplength=1050).grid(
            row=2, column=0, columnspan=7, sticky='w', pady=(5, 0),
        )

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
        if self.jog is not None and not self.jog._cancel.is_set() and not self.af_running:
            self.jog._set_armed(True)

    def open_camera(self):
        if self.af_running:
            return
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
        if self.af_running:
            return
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
        if self.af_running or self.connecting or self.controller is not None:
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

        self.connection_thread = threading.Thread(target=worker, daemon=True)
        self.connection_thread.start()

    def stop(self):
        self.af_cancel.set()
        if self.jog is not None:
            self.jog._request_stop()

    def _refresh_controls(self):
        locked = self.af_running or self.closing.is_set()
        for widget in (*self.af_entries, self.camera_entry):
            widget.configure(state='disabled' if locked else 'normal')
        self.backend_combo.configure(state='disabled' if locked else 'readonly')
        self.port_entry.configure(state='disabled' if locked or self.controller is not None else 'normal')
        self.camera_button.configure(state='disabled' if locked or self.stream.running else 'normal')
        self.close_camera_button.configure(state='disabled' if locked else 'normal')
        self.connect_button.configure(
            state='disabled' if locked or self.connecting or self.controller is not None else 'normal',
        )
        ready = (not locked and self.controller is not None and self.jog is not None
                 and not self.jog._cancel.is_set() and not self.af_cancel.is_set()
                 and self.stream.active and self.stream.latest() is not None and self.roi is not None)
        self.af_button.configure(state='normal' if ready else 'disabled')

    def start_autofocus(self):
        if self.af_running or self.closing.is_set():
            return
        if (self.controller is None or self.jog is None or self.jog._cancel.is_set()
                or self.af_cancel.is_set()):
            self.af_status.set('请先连接控制器。停止锁定后需重启程序。')
            return
        if self.jog._move is not None or self.jog._stop is not None:
            self.af_status.set('请等待当前手动点动完成。')
            return
        frame = self.stream.latest()
        if not self.stream.active or frame is None or self.roi is None:
            self.af_status.set('请先打开相机并选择样品区域。')
            return
        try:
            self.roi.validate_shape(frame.image.shape)
            values = {name: (float(variable.get()) if name == 'settle_seconds' else int(variable.get()))
                      for name, variable in self.af_values.items()}
            config = AutofocusConfig(**values)
        except (ValueError, TypeError) as exc:
            self.af_status.set(f'参数无效：{exc}')
            return

        self.af_running = True
        self.jog._set_armed(False)
        self.drag_start = None
        self.preview.delete('drag')
        self.af_status.set('正在读取当前 Z，准备确认扫描范围…')
        self._refresh_controls()
        roi = self.roi
        expected_shape = frame.image.shape

        def prepare():
            try:
                state = self.controller.read_axis_state('Z')
                if state['running']:
                    raise StageControllerError('Z 轴仍在运动，请停稳后再启动')
                if type(state['position']) is not int:
                    raise StageControllerError('控制器返回了无效 Z 位置')
                self.af_queue.put(('prepared', (state['position'], roi, config, expected_shape)))
            except Exception as exc:
                self.af_queue.put(('error', str(exc)))

        self.af_thread = threading.Thread(target=prepare, daemon=True)
        self.af_thread.start()

    def _finish_autofocus(self, message):
        self.af_running = False
        self.af_status.set(message)
        self._refresh_controls()

    def _poll_autofocus(self):
        while True:
            try:
                kind, payload = self.af_queue.get_nowait()
            except queue.Empty:
                return
            if kind == 'prepared':
                if self.closing.is_set() or self.af_cancel.is_set():
                    self._finish_autofocus('自动对焦已取消。重启程序后可继续运动。')
                    continue
                position, roi, config, expected_shape = payload
                approved = messagebox.askyesno(
                    '确认 Z 轴自动对焦范围',
                    f'当前控制器 Z 计数为 {position}。\n'
                    f'扫描范围为 [{position - config.half_range}, {position + config.half_range}] 脉冲。\n'
                    '请确认整个范围内探针与样品有安全间隙，物理急停可用。\n是否开始扫描？',
                    parent=self.root,
                )
                if not approved or self.closing.is_set() or self.af_cancel.is_set():
                    self._finish_autofocus('本次扫描已取消。')
                    continue
                latest = self.stream.latest()
                if not self.stream.active or latest is None or latest.image.shape != expected_shape:
                    self.stop()
                    self._finish_autofocus('相机状态或画面尺寸改变，本次扫描已取消。')
                    continue
                self.af_status.set('自动对焦正在启动…')

                def worker():
                    try:
                        result = run_autofocus(
                            self.controller, self.stream, roi, config, self.af_cancel,
                            progress=lambda event, data: self.af_queue.put((event, data)),
                            expected_start=position, expected_shape=expected_shape,
                        )
                        self.af_queue.put(('result', result))
                    except Exception as exc:
                        self.af_cancel.set()
                        self.af_queue.put(('error', str(exc)))

                self.af_thread = threading.Thread(target=worker, daemon=True)
                self.af_thread.start()
            elif kind == 'result':
                message = f'{payload.status} · {payload.message}'
                if payload.status != 'verified':
                    self.af_cancel.set()
                    if self.jog is not None:
                        self.jog._cancel.set()
                    message += ' 运动已锁定，重启程序后继续。'
                self._finish_autofocus(message)
            elif kind == 'error':
                self.stop()
                self._finish_autofocus(f'自动对焦失败：{payload}。重启程序后可继续运动。')
            elif kind == 'status':
                self.af_status.set(str(payload))
            elif kind == 'measurement':
                self.af_status.set(f'已采样 Z={payload.z:g}，共 {len(payload.frames)} 帧；自动对焦继续中…')

    def _drag_start(self, event):
        if not self.af_running and self.display_size is not None:
            self.drag_start = (event.x, event.y)

    def _drag_move(self, event):
        if self.drag_start is None:
            return
        self.preview.delete('drag')
        self.preview.create_rectangle(*self.drag_start, event.x, event.y, outline='#ffcf4a', tags='drag')

    def _drag_end(self, event):
        if self.af_running:
            return
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
            if self.af_running:
                self.stop()
                self.af_status.set('相机画面尺寸改变，已停止自动对焦。')
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
        self._poll_autofocus()
        if self.closing.is_set():
            connecting = self.connection_thread is not None and self.connection_thread.is_alive()
            focusing = self.af_thread is not None and self.af_thread.is_alive()
            jog_busy = self.jog is not None and (self.jog._move is not None or self.jog._stop is not None)
            if not connecting and not focusing and not jog_busy:
                self.root.quit()
            else:
                self.root.after(50, self._tick)
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
                self.jog = EmbeddedJog(
                    self.root, self.motor_status, controller, self.jog_target,
                    blocked=lambda: self.af_running or self.closing.is_set(),
                    on_stop=self.af_cancel.set, on_exit=self.close,
                )
                self.root.protocol('WM_DELETE_WINDOW', self.close)
                self.motor_status.set('已连接。松开移动键，再点击启用点动')
                self.jog._poll()
        if self.stream.error and self.camera_error != self.stream.error:
            self.camera_error = self.stream.error
            self.camera_status.set(f'相机错误：{self.camera_error}')
            if self.af_running:
                self.stop()
        frame = self.stream.latest()
        if self.stream.active and frame is not None and frame.sequence != self.last_sequence:
            try:
                self._show_frame(frame)
                self.last_sequence = frame.sequence
                self.camera_status.set(self.exposure_status or '相机已打开')
            except (ValueError, cv2.error, tk.TclError) as exc:
                self.camera_status.set(f'画面错误：{exc}')
                if self.af_running:
                    self.stop()
                self.stream.stop(timeout=0)
        self._refresh_controls()
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
        self.stop()

    def finish(self):
        if self.af_thread is not None:
            self.af_thread.join()
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
