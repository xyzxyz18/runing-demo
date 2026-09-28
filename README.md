# 跑步机跑姿分析 Web Demo

本项目按 `方案.md` 实现本地的 Video → MediaPipe Pose → 步态事件 → 指标 → 反馈流程，适合固定机位拍摄的单人侧面跑步视频。

## 安装

推荐 Python 3.10 或 3.11，并使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行（浏览器界面）

启动本地 Web 服务：

```bash
python app.py
```

程序会自动打开 `http://127.0.0.1:8000`。界面提供三种模式：

- **视频分析**：上传 MP4、MOV 或 AVI 后，在原视频上逐帧叠加 MediaPipe 骨架。播放或拖动进度条时，膝/髋关节曲线、游标和当前数值同步更新。
- **实时摄像头**：经浏览器授权后，在摄像头画面上实时绘制骨架，并连续绘制左右膝关节角度曲线。
- **分析历史**：查看已完成的指标和报告；原视频仍存在时，可以创建一次新的重新分析，不覆盖旧结果。

视频结果页会将同侧足尖相对髋部的平面轨迹按跨步周期叠加，显示平均轨迹和周期波动。点击“查看完整分析简报”可以展开结论；也可下载独立的 `report.html`。周期不足时明确显示“数据不足”。稳定性仅表示这段侧面视频中轨迹的重复程度，不等于跑姿优劣或伤病风险。

新分析会保存每帧的实际时间戳，用于可变帧率视频的骨架、曲线与播放进度同步。

所有视频和摄像头画面都由当前服务处理。部署到远程服务器时，浏览器会把画面发送到该服务器。

如不希望自动打开浏览器：

```bash
python app.py --no-browser
```

## 命令行用法

```bash
python main.py input/test.mp4 --output output
```

输出包括：

- `annotated.mp4`：骨架、膝角、步频和步态事件叠加视频
- `landmarks.csv`：33 个关键点的原始值与平滑值
- `metrics.json`：步频、步时、步幅周期、关节 ROM、左右差异与规则反馈
- `report.png`：指标 Dashboard、膝角曲线和脚部轨迹
- `report.html`：可展开的简要文字报告
- `timeline.json`：逐帧骨架、角度、事件与时间戳

## 部署到服务器

服务器安装 Docker 和 Compose 后，在项目目录运行：

```bash
docker compose up -d --build
```

默认只监听服务器的 `127.0.0.1:8000`，分析历史和上传视频保存在 `pace_data` Docker 卷中，容器重建后仍可读取。查看运行状态：

```bash
docker compose ps
docker compose logs -f pace-lab
```

远程访问时，建议由现有反向代理将 HTTPS 域名转发到 `127.0.0.1:8000`，并在代理层设置登录保护和至少 2 GB 的上传限制。浏览器摄像头在远程访问时需要 HTTPS；直接打开服务器的 HTTP 地址通常无法授权摄像头。此 Demo 的任务状态保存在单个 Web 进程内，因此部署配置固定为 1 个 Gunicorn worker；不要横向启动多个副本共用同一份历史目录。

不用 Docker 时也可以设置持久化目录并以单进程启动：

```bash
export PACE_DATA_DIR=/srv/pace-data
gunicorn --bind 127.0.0.1:8000 --workers 1 --threads 4 --timeout 300 app:app
```

## 拍摄建议

- 固定机位、完整侧面、全身入镜；尽量避免遮挡。
- 建议 720p 以上、约 30 FPS，至少包含多个完整步态周期。
- 二维单目视频无法提供实验室级三维运动学结果；输出仅供运动观察。
