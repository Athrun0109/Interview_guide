# AI 面试分析助手

一款帮助求职者复盘在线面试表现、提供改进建议和后续面试策略的 AI 工具。

面试者上传面试录音/录像后，工具会自动完成语音转文字、说话人分离，并调用大语言模型对每一对问答进行评价，最终给出面试总结、优缺点分析和改进建议。

---

## 核心功能

- **语音转文字**：基于 faster-whisper / WhisperX 的 Large V3 模型，对日语、英语面试录音进行高质量转写
- **说话人分离**：使用 pyannote.audio 3.1 进行说话人聚类，支持用户手动指定说话人数量以提升准确率
- **重叠语音检测**：自动识别"多人同时说话"的片段并单独标记，避免将别人的声音混入某人的发言中
- **说话人身份选择**：录音中所有说话人会被提取片段样本，用户试听后一键点选"这是我"，其余默认标记为面试官
- **公司背景检索**（可选）：通过 Serper API 搜索目标公司信息，辅助分析上下文
- **LLM 面试分析**：调用 Gemini 对每个问答环节进行定性评价（好/一般/不好），并给出具体改进建议
- **优雅降级**：未提供公司信息时退化为纯面试内容分析
- **面试失败模式**：用户标记"被拒"后，分析报告会转为支持性语气，重点提供改进方向和心理鼓励
- **可视化进度条**：转写过程中实时显示当前阶段和完成百分比
- **会话导入/导出**：可将面试内容和 Prompt 导出为 txt 文件，方便在 ChatGPT/Claude 等其他 LLM 中复用

---

## 技术栈

| 组件 | 选型 |
|------|------|
| 前端 UI | Streamlit |
| 语音识别 | faster-whisper / WhisperX（Large V3） |
| 说话人分离 | pyannote.audio 3.1 |
| 公司搜索 | Serper.dev API |
| LLM | Google Gemini |
| 运行环境 | 本地 GPU（CUDA） |

### 支持的语言

- 日语（主要）
- 英语（主要）
- 中文暂不支持

---

## 目录结构

```
Interview_guide/
├── app.py                 # Streamlit 主应用入口
├── config.py              # API Key 和设备配置
├── requirements.txt       # Python 依赖
├── modules/
│   ├── transcriber.py     # STT + 说话人分离 + 进度回调
│   ├── searcher.py        # Serper 公司搜索
│   ├── analyzer.py        # Gemini 分析调用
│   └── prompts.py         # Prompt 模板与分析模式
├── howtorun.txt           # 启动备忘
└── CLAUDE.md              # 项目设计文档
```

---

## 环境准备

### 1. 安装依赖

建议使用 conda（推荐 miniforge）创建独立环境：

```bash
conda create -n interview_guide python=3.12 -c conda-forge
conda activate interview_guide

# 1) 先装 PyTorch 三件套（必须带 --index-url，否则找不到 cu121 wheel）
pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) 锁定 numpy 1.x（PyTorch 2.2.2 用 NumPy 1.x 编译，
#    避免后续装别的包时被升级到 2.x 触发 ABI 不兼容）
pip install numpy==1.26.4

# 3) 装项目依赖
pip install -r requirements.txt

# 4) 单独装 whisperx：其包元数据声明 numpy>=2.1.0，
#    但运行时实际兼容 numpy 1.26.4，用 --no-deps 跳过版本检查
pip install --no-deps whisperx==3.7.6
```

需要 NVIDIA GPU + CUDA 运行时（faster-whisper 和 pyannote 都依赖）。
PyTorch 2.2.2 + CUDA 12.1 是经过验证的组合，请勿随意升级。

### 2. 配置 API Key

需要以下 Key：

- **Gemini API Key**（必需）— 用于面试内容分析
- **HuggingFace Token**（必需）— 用于下载 pyannote 模型。首次使用需先在 HF 官网同意 `pyannote/speaker-diarization-3.1` 的使用协议
- **Serper API Key**（可选）— 用于搜索公司背景，免费额度 2500 次/月

Key 可以通过环境变量或 Streamlit 侧边栏输入框配置。

### 3. 手动放置 ffmpeg / ffprobe（**必需**）

项目通过 pydub 处理音频/视频解码，需要 `ffmpeg.exe` 和 `ffprobe.exe` 这两个二进制文件。
由于体积较大（合计约 150 MB），它们**不随仓库上传 GitHub**，需要手动下载并放入 `modules/` 目录。

**为什么不直接用 conda 装的 ffmpeg？**
> 经测试，conda（特别是 Anaconda 主源）安装的 ffmpeg 在 Windows 上常出现 DLL 冲突
> （报错 `gdk_pixbuf-2.0-0.dll` 找不到 `libintl_bind_textdomain_codeset` 入口点），
> 进而导致 pydub 调用 ffprobe 失败、抛 `JSONDecodeError`。
> 改用独立 ffmpeg 二进制是最稳妥的解决方案。

**下载步骤**：

1. 访问 https://www.gyan.dev/ffmpeg/builds/
2. 下载 **ffmpeg-release-essentials.zip**（或 7z 版本）
3. 解压后在 `bin/` 目录下找到 `ffmpeg.exe` 和 `ffprobe.exe`
4. 把这两个文件复制到本项目的 `modules/` 目录下：

```
Interview_guide/
└── modules/
    ├── ffmpeg.exe       ← 必需
    ├── ffprobe.exe      ← 必需
    ├── transcriber.py
    └── ...
```

`config.py` 启动时会自动检测并优先使用这两个文件，无需额外配置。

> 如果两个文件没有放入 `modules/`，运行到 **Start Transcription** 步骤时会报
> `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` 之类的解码错误。

---

## 启动方式

```bash
conda activate interview_guide
cd C:\Users\Admin\Documents\workspace\Interview_guide
streamlit run app.py
```

浏览器会自动打开 Streamlit 页面。

---

## 使用流程

### Step 1：上传面试录音
- 支持 mp4 / mp3 / wav / m4a / webm / ogg 格式
- 选择 Whisper 模型（默认 Large V3 + WhisperX，推荐）
- 选择面试主要语言（日语 / 英语）
- 填写面试人数（含你自己）。如果不确定，勾选 "I'm not sure" 让模型自动估计
- 点击 **Start Transcription**，进度条会实时显示当前阶段和百分比

### Step 2：选择"自己"
- 试听每个说话人的样本片段
- 点击"这是我"按钮标记出本人，其余自动标记为面试官

### Step 3：填写岗位信息
- 公司名称（可选）
- 岗位名称
- 岗位 JD
- 如果面试被拒，可点击 **I Was Rejected** 切换到失败模式

### Step 4：生成分析报告

完成岗位信息填写后，有 **两种方式** 获得面试分析，可任选其一。

#### 方式 A（推荐）：导出 Prompt，交给第三方顶级 LLM 分析

这是当前**最推荐**的使用方式，理由如下：
- **无需配置 API Key**，跳过繁琐的 Gemini API 申请和配额管理
- **可以直接使用最强的模型**，比如 Google Gemini Pro 3.1、Claude Opus、GPT-5 等网页版，分析质量和语言表达通常优于 API 默认调用的版本
- **更傻瓜式**：一次导出、随处可用，把 txt 粘进任意 LLM 对话框即可得到完整报告
- **可反复追问**：在 LLM 网页端可以针对报告继续追问、让它换角度分析，API 模式下不具备这种灵活性

操作步骤：
1. 在 Step 3 下方找到 **Export for Other LLMs** 区域
2. 点击 **Download as .txt** 下载完整 Prompt 文件（已自动拼好面试转录、岗位信息、公司背景和分析指令）
3. 打开你偏好的 LLM 网页端（如 Gemini Pro 3.1 / Claude / ChatGPT）
4. 将 txt 内容复制粘贴进对话框发送，即可获得完整面试分析报告
5. 下次需要再分析同一场面试时，直接通过 Step 1 的 **Import Previous Session** 页签重新载入 txt 文件即可，无需再次转写音频

#### 方式 B：使用内置 Gemini API 自动分析

如果已经配置了 Gemini API Key，也可以在应用内直接完成分析：
- 选择 Gemini 模型
- 点击 **Analyze Interview**
- 报告会分成几个可折叠区域展示：
  - Overall Impression（整体印象）
  - Q&A Breakdown（每个问答环节的点评）
  - Interviewer's Focus（面试官关注点）
  - Improvement Suggestions（改进建议）
  - What Went Well（失败模式下额外提供）

---

## 录音建议

项目目前只处理音频，不处理视频画面。推荐使用第三方工具录制线上面试：

- **OBS Studio**（推荐）：同时捕获麦克风和系统音频，一个文件搞定
- **VoiceMeeter Banana**：虚拟混音器，适合长期方案
- **Audacity + WASAPI loopback**：仅需纯录音时的简易方案

为了提升说话人分离的准确率，建议使用较为安静的环境录音。

---

## 开发阶段规划

- **第一阶段（当前 Demo）**：文本分析 pipeline 打通，验证 AI 输出的实用价值
- **第二阶段**：加入副语言特征分析（语音语调）
- **第三阶段**：支持视频输入，分析形象、动作、气质等全方位表现

---

## 已知限制

- 说话人分离在说话人声线相近时可能出错，建议提供明确的说话人数
- Whisper 对日英混合的技术词汇偶尔会把英文强行转成片假名
- 目前不支持实时/流式处理，需要等待整段音频处理完毕
- 仅本地 GPU 运行，未部署云服务
