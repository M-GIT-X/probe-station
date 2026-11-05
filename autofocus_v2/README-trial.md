# V2 临时试验程序

## 启动

在 Windows 的 VS Code 中打开仓库根目录，在 PowerShell 终端运行：

```powershell
git switch codex/v2-lab-tools
git pull --ff-only
& .\.venv\Scripts\python.exe .\trial_gui.py
```

首次使用、尚未创建 Python 环境时，先运行：

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 程序简介

程序启动后显示 GUI。先打开探针台照明，填写相机编号并点击“打开相机”，等待启动曝光调节结束。绿色框为评分区域，可以在画面中拖动选择样品纹理。右侧显示该区域的 V2 清晰度和随时间变化的曲线。

填写已经确认的控制器本机串口，点击“连接控制器”。松开移动键后点击“启用点动”，使用 W/S、A/D、Q/E 分别点动 Y、X、Z。默认每次 1 脉冲、速度 10%。运动键在点动区获得焦点时生效。Space、Esc 和软件急停按钮停止全部轴，停止后重新启动程序才能继续运动。启动和连接过程均无回零动作。

先确认手动运动方向及安全间隙，再选择固定样品区域并点击“开始自动对焦”。程序会请求确认本次 Z 扫描范围。默认以起始位置为中心扫描 ±20 脉冲，粗步长 5、细步长 1，每点等待 0.5 秒并采集 5 帧。程序逐点确认控制器到位和脉冲计数，在粗扫峰区细扫，最后以相同方向接近候选位置并重新采样验证。

自动对焦期间保持样品、照明及曝光不变。平坦曲线、边界峰、多个焦峰、分块冲突及异常运动会显示对应结果；停止或失败后程序停留在当前点。控制器反馈来自脉冲计数，实际机械位置、相机缓存及最终对焦效果需要现场确认。

主要文件按功能分工：

| 文件 | 用途 |
| --- | --- |
| `trial_gui.py` | GUI、相机画面、控制提示、实时曲线与操作协调 |
| `autofocus_v2/metrics.py` | 固定样品区域的 V2 清晰度与曝光质量评分 |
| `autofocus_v2/capture.py` | 单线程相机采集、最新帧和图像组采样 |
| `autofocus_v2/analysis.py` | 图像组稳定性和焦峰分析 |
| `autofocus_v2/motion.py` | 自动对焦的 Z 范围、到位及计数检查 |
| `autofocus_v2/autofocus.py` | 粗扫、细扫、回位和独立验证 |
| `autofocus_v2/models.py` | 区域、帧、分数和测量结果类型 |
| `stage_controller.py`、`stage_protocol.py` | 串口控制器和运动协议 |
| `camera_opencv.py`、`camera_preview.py` | 相机后端、启动曝光调节与独立预览 |
| `z_jog.py` | 独立三轴键盘点动 |

两个独立试验入口也可以单独启动：

```powershell
& .\.venv\Scripts\python.exe .\camera_preview.py --camera-index 1 --backend DSHOW
$port = Read-Host '输入已确认的控制器 COM 口'
& .\.venv\Scripts\python.exe .\z_jog.py --port $port
```

相机编号按现场设备选择。控制器使用本机串口；接在树莓派上的兼容串口需要在可访问该串口、具有 Tk 图形桌面的主机上运行程序。
