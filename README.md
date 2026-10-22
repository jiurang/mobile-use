# mobile-use: automate your phone with natural language

<div align="center">

![mobile-use in Action](./doc/banner-v2.png)

</div>

<div align="center">

[![Discord](https://img.shields.io/discord/1403058278342201394?color=7289DA&label=Discord&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/6nSqmQ9pQs)
[![GitHub stars](https://img.shields.io/github/stars/minitap-ai/mobile-use?style=for-the-badge&color=e0a8dd)](https://github.com/minitap-ai/mobile-use/stargazers)

<h3>
    <a href="https://docs.minitap.ai/v2/mcp-server/introduction"><b>📚 Documentation</b></a> •
    <a href="https://arxiv.org/abs/2602.07787"><b>📃 Paper</b></a>

</h3>
<p align="center">
    <a href="https://discord.gg/6nSqmQ9pQs"><b>Discord</b></a> •
    <a href="https://x.com/minitap_ai?t=iRWtI497UhRGLeCKYQekig&s=09"><b>Twitter / X</b></a>
</p>

[![PyPI version](https://img.shields.io/pypi/v/minitap-mobile-use.svg?color=blue)](https://pypi.org/project/minitap-mobile-use/)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/minitap-ai/mobile-use/blob/main/LICENSE)

</div>

Mobile-use is a powerful, open-source AI agent that controls your Android or IOS device using natural language. It understands your commands and interacts with the UI to perform tasks, from sending messages to navigating complex apps.

> Mobile-use is quickly evolving. Your suggestions, ideas, and reported bugs will shape this project. Do not hesitate to join in the conversation on [Discord](https://discord.gg/6nSqmQ9pQs) or contribute directly, we will reply to everyone! ❤️

## ✨ Features

- 🗣️ **Natural Language Control**: Interact with your phone using your native language.
- 📱 **UI-Aware Automation**: Intelligently navigates through app interfaces (note: currently has limited effectiveness with games as they don't provide accessibility tree data).
- 📊 **Data Scraping**: Extract information from any app and structure it into your desired format (e.g., JSON) using a natural language description.
- 🔧 **Extensible & Customizable**: Easily configure different LLMs to power the agents that power mobile-use. Supports OpenAI, Google, xAI, OpenRouter, MiniMax, and more.

## Benchmarks

<p align="center">
  <a href="https://minitap.ai/benchmark">
    <img src="https://files.peachworlds.com/website/2b590171-669d-42ce-b4b5-ce6eae83a9d8/scorerank-140126.webp" alt="Project banner" />
  </a>
</p>

We stand as the top performers and the first to have completed 100% of the AndroidWorld benchmark.

Get more info about how we reached this milestone here: [Minitap Benchmark](https://minitap.ai/benchmark).

The official leaderboard is available [here](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0).

Check out our research paper [here](https://arxiv.org/abs/2602.07787).

## 🚀 Getting Started

Ready to automate your mobile experience? Follow these steps to get mobile-use up and running.

### 🛠️ Run locally

Mobile-use runs on your own Android or iOS device with your choice of supported LLM provider. Start with the [local quickstart](https://docs.minitap.ai/mobile-use-sdk/quickstart).

1.  **Set up Environment Variables:**
    Copy the example `.env.example` file to `.env` and add your API keys.

    ```bash
    cp .env.example .env
    ```

2.  **(Optional) Customize LLM Configuration:**
    To use different models or providers, create your own LLM configuration file.

    ```bash
    cp llm-config.override.template.jsonc llm-config.override.jsonc
    ```

    Then, edit `llm-config.override.jsonc` to fit your needs.

    You can also use local LLMs or any other openai-api compatible providers :

    1. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in your `.env`
    2. In your `llm-config.override.jsonc`, set `openai` as the provider for the agent nodes you want, and choose a model supported by your provider.

    **Using MiniMax:**

    [MiniMax](https://www.minimaxi.com/) provides high-performance, OpenAI-compatible LLMs for agentic and coding tasks. To use MiniMax as your provider:

    1. Set `MINIMAX_API_KEY` in your `.env`
    2. Copy the contents of the `minimax` preset object from `llm-config.defaults.jsonc` into your `llm-config.override.jsonc`, or set `"provider": "minimax"` on individual agent nodes.

    Available models: `MiniMax-M2.7` (200K context, recommended), `MiniMax-M2.7-highspeed` (200K context, faster inference).

    **Using DeepSeek:**

    [DeepSeek](https://platform.deepseek.com/) serves DeepSeek-V4 models over an OpenAI-compatible API at `https://api.deepseek.com`. To use it:

    1. Set `DEEPSEEK_API_KEY` in your `.env`
    2. Copy the contents of the `deepseek` preset object from `llm-config.defaults.jsonc` into your `llm-config.override.jsonc`, or set `"provider": "deepseek"` on individual agent nodes.

    Available models: `deepseek-v4-pro` (1M context, text only), `deepseek-flash` (1M context, faster/cheaper, accepts screenshots).

    > [!NOTE]
    > DeepSeek answers in **thinking mode** by default: it emits a chain of thought in `reasoning_content` before every answer, which costs extra latency and completion tokens on every agent call. Add `DEEPSEEK_THINKING="false"` to your `.env` to send `{"thinking": {"type": "disabled"}}` and trade that reasoning for speed.
    >
    > `temperature` only has an effect in non-thinking mode.

    **Using SiliconFlow (硅基流动):**

    [SiliconFlow](https://cloud.siliconflow.cn/) serves Qwen / DeepSeek / GLM / Kimi and other models over an OpenAI-compatible API at `https://api.siliconflow.cn/v1`. To use it:

    1. Set `SILICONFLOW_API_KEY` in your `.env` (optionally `SILICONFLOW_BASE_URL` to point at `https://api.siliconflow.com/v1` instead)
    2. Copy the contents of the `siliconflow` preset object from `llm-config.defaults.jsonc` into your `llm-config.override.jsonc`, or set `"provider": "siliconflow"` on individual agent nodes.

    > [!IMPORTANT]
    > **This platform has no model that can do all three at once** — see the screenshot, call a tool, *and* answer with a structured object. The capability matrix below was measured against the live API (2026-10-21):
    >
    > | model | sees images | tool calls | `json_schema` output |
    > |---|---|---|---|
    > | `zai-org/GLM-4.5V` | ✅ | ✅ | ❌ `400 Json mode is not supported` |
    > | `Qwen/Qwen3-VL-32B-Instruct` | ✅ | ❌ empty answer | ✅ |
    > | `Qwen/Qwen3-VL-8B-Instruct` | ✅ | ❌ empty answer | ✅ |
    > | `Qwen/Qwen2.5-72B-Instruct` | ❌ | ✅ | ✅ |
    > | `Qwen/Qwen2.5-7B-Instruct` | ❌ | ✅ | ✅ |
    >
    > So the vision roles have to be split: **`executor` → `GLM-4.5V`** (it clicks, so it needs eyes *and* tool calls), **`cortex` → `Qwen3-VL`** (it reads the screen and answers with a structured decision), everything else → a text model. Pointing both at the same model breaks one of them.
    >
    > Two more measured facts: the vision models reject a **forced** `tool_choice` (`400 Specifying functions for tool_choice is not yet supported`) — which is why structured output goes out as `response_format` first, with a tool-call retry as the second path; and `temperature` accepts 0..2, so `hopper`'s hardcoded `0` keeps working.

    **Using a local llama.cpp server (llama-server):**

    `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) serves any GGUF model over an OpenAI-compatible API. The `llamacpp` provider points mobile-use at it:

    1. Start the server, e.g. `llama serve -hf unsloth/Qwen3.5-0.8B-GGUF:Q6_K --ctx-size 65536`
    2. Set `"provider": "llamacpp"` and the model name **exactly as the server reports it** (via `GET /v1/models` — e.g. `unsloth/Qwen3.5-0.8B-GGUF:Q6_K`, including the `:Q6_K` suffix) on the agent nodes you want to run locally
    3. Optionally set `LLAMACPP_BASE_URL` (default `http://127.0.0.1:9931/v1`) and `LLAMACPP_API_KEY` (only needed when the server was started with `--api-key`)

    Three measured facts (llama-server b11782, 2026-10-22) — they are why this provider is not just "the `openai` provider with a different URL":

    - **Thinking has to be turned off.** The model answers in a thinking chain by default, and the chain eats the whole token budget — the answer body comes back empty and every structured output fails to parse. The provider therefore sends `chat_template_kwargs: {"enable_thinking": false}` per request, so **you do not have to restart the server**. Switch it back on per channel with `"thinking": true` (that is what the manager web UI's *thinking mode* switch writes).
    - **The generation cap must be bounded.** llama.cpp's `n_predict` defaults to `-1` (unlimited) and a thinking model keeps going — with no cap the request never returns (measured: 20s and 300s timeouts). The provider sends `max_tokens` (1024 with thinking off, 8192 with it on).
    - **Structured output goes out as native `response_format=json_schema`.** llama.cpp enforces it with a GBNF grammar, so the answer is valid JSON by construction; the "bind the schema as a tool" path (which the other providers use) is unreliable on small models — they often answer in prose instead of calling the tool.

    Check the endpoint with the probe before pointing a real task at it:

    ```bash
    python scripts/probe_llamacpp.py            # per-role parse rate + latency
    python scripts/probe_llamacpp.py --thinking # same, with the thinking chain on
    ```

    > [!NOTE]
    > The agent runs where you start it. With `run-local.ps1` (or `python -m minitap.mobile_use.main`) the LLM calls come from the PC, so `127.0.0.1:9931` reaches your local server. When the agent runs inside the Android app, the call originates on the phone — forward the port with `adb reverse tcp:9931 tcp:9931` first.

    **Using Anthropic Claude:**

    [Anthropic](https://www.anthropic.com/) provides Claude — a capable, safety-focused model family. To use it:

    1. Set `ANTHROPIC_API_KEY` in your `.env`
    2. Copy the contents of the `anthropic` preset object from `llm-config.defaults.jsonc` into your `llm-config.override.jsonc`, or set `"provider": "anthropic"` on individual agent nodes.

    Available models: `claude-sonnet-4-6` (200K context, recommended), `claude-haiku-4-5-20251001` (200K context, faster/cheaper).

    > [!NOTE]
    > If you want to use Google Vertex AI, you must either:
    >
    > - Have credentials configured for your environment (gcloud, workload identity, etc…)
    > - Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable
    >
    > More information: - [Credential types](https://cloud.google.com/docs/authentication/application-default-credentials#GAC) - [google.auth API reference](https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth)

### Quick Launch (Docker)

> [!NOTE]
> This quickstart, is only available for Android devices/emulators as of now, and you must have Docker installed.

First:

- Either plug your Android device and enable USB-debugging via the Developer Options
- Or launch an Android emulator

Then run in your terminal:

1. For Linux/macOS:

```bash
chmod +x mobile-use.sh
bash ./mobile-use.sh \
  "Open Gmail, find first 3 unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

2. For Windows (inside a Powershell terminal):

```powershell
powershell.exe -ExecutionPolicy Bypass -File mobile-use.ps1 `
  "Open Gmail, find first 3 unread emails, and list their sender and subject line" `
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If using your own device, make sure to accept the ADB-related connection requests that will pop up on your device.

#### 🧰 Troubleshooting

The script will try to connect to your device via IP.
Therefore, your device **must be connected to the same Wi-Fi network as your computer**.

##### 1. No device IP found

If the script fails with the following message:

```
Could not get device IP. Is a device connected via USB and on the same Wi-Fi network?
```

Then it couldn't find one of the common Wi-Fi interfaces on your device.
Therefore, you must determine what WLAN interface your phone is using via `adb shell ip addr show up`.
Then add the `--interface <YOUR_INTERFACE_NAME>` option to the script.

##### 2. Failed to connect to <DEVICE_IP>:5555 inside Docker

This is most probably an issue with your firewall blocking the connection. Therefore there is no clear fix for this.

##### 3. Failed to pull GHCR docker images (unauthorized)

Since UV docker images rely on a `ghcr.io` public repositories, you may have an expired token if you used `ghcr.io` before for private repositories.
Try running `docker logout ghcr.io` and then run the script again.

### Manual Launch (Development Mode)

For developers who want to set up the environment manually:

#### 1. Device Support

Mobile-use currently supports the following devices:

- **Physical Android Phones**: Connect via USB with USB debugging enabled.
- **Android Simulators**: Set up through Android Studio.
- **iOS Simulators**: Supported for macOS users.

> [!NOTE]
> Physical iOS devices are not yet supported.

#### 2. Prerequisites

**For Android:**

- **[Android Debug Bridge (ADB)](https://developer.android.com/studio/releases/platform-tools)**: A tool to connect to your device.

**For iOS (macOS only):**

- **[Xcode](https://developer.apple.com/xcode/)**: Apple's IDE for iOS development.
- **[fb-idb](https://fbidb.io/docs/installation/)**: Facebook's iOS Development Bridge for device automation.

  ```bash
  # Install via Homebrew (macOS)
  brew tap facebook/fb
  brew install idb-companion
  ```

  > [!NOTE]
  > `idb_companion` is required to communicate with iOS simulators. Make sure it's in your PATH after installation.

**Common requirements:**

Before you begin, ensure you have the following installed:

- **[uv](https://github.com/astral-sh/uv)**: A lightning-fast Python package manager.

#### 3. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/minitap-ai/mobile-use.git && cd mobile-use
    ```

2.  [**Setup environment variables**](#-getting-started)

3.  **Create & activate the virtual environment:**

    ```bash
    # This will create a .venv directory using the Python version in .python-version
    uv venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    # Sync with the locked dependencies for a consistent setup
    uv sync
    ```

5.  ** _(Optional)_ Enable OCR perception:**
    ```bash
    # CPU inference
    uv pip install -e ".[ocr]"

    # Or GPU inference (NVIDIA, CUDA 12 — see the section below)
    uv pip install -e ".[ocr-gpu]"
    ```

    See [⚡ OCR perception & GPU backends](#-ocr-perception--gpu-backends) for details.

## ⚡ OCR perception & GPU backends

Text drawn by the app itself (video captions, custom widgets) has no node in the
accessibility tree. OCR perception reads it with PaddleOCR and feeds the text —
and its measured coordinates — to the model, so those pixels become clickable.

### Choosing a backend

```bash
# CPU (default extra)
uv pip install -e ".[ocr]"

# NVIDIA GPU via the CUDA execution provider
uv pip install -e ".[ocr-gpu]"

# Windows GPU via DirectML (no CUDA toolkit needed)
uv pip install -e ".[ocr-dml]"
```

> [!IMPORTANT]
> These three extras are **mutually exclusive** — `onnxruntime`,
> `onnxruntime-gpu` and `onnxruntime-directml` all provide the same
> `onnxruntime` module and overwrite each other's files.

The backend defaults to `auto`, which probes for CUDA and silently stays on CPU
otherwise. Override it with an environment variable:

```bash
MOBILE_USE_OCR_DEVICE=cuda|cpu|dml|auto
MOBILE_USE_OCR_REC_BATCH=serial|pad|resize   # optional, see below
```

Or in code — an explicit field always wins over the environment variable:

```python
from minitap.mobile_use.context import PerceptionConfig
from minitap.mobile_use.services.ocr import OcrConfig

perception = PerceptionConfig(
    ocr_config=OcrConfig(
        device="cuda",
        device_id=0,        # which GPU, on multi-GPU machines
        rec_batch="pad",
    )
)
```

> [!TIP]
> Every knob is normalized before use: `"SERIAL"`, `"Serial"` and `" serial "` all
> behave as `serial`, and an unrecognised value is **logged as a warning** and
> falls back to the default rather than silently running a different mode.

### Measured numbers (RTX 4060 Laptop, driver 566.26, onnxruntime-gpu 1.22)

Per-screenshot OCR on a 1220x2712 capture, steady state (median of 5):

| Backend | Time |
| --- | --- |
| CPU | 264ms |
| CUDA (auto-selected) | 124ms |

**2.14x — but only because recognition is batched.** Each `run()` has a fixed
cost: ~6.2ms on CPU, ~40ms on CUDA (host↔device copies plus synchronization).
The pre-existing code called `rec` once per text box — 26 calls for one screen —
so running on a GPU *unchanged* makes it **4x slower than CPU**. Batching the
boxes into a single call is what turns the GPU into a win.

`rec_batch` controls this:

- `pad` (default) — every row keeps the geometry it had per-box, right-padded
  with black to the batch width. **Same pixel content as `serial`**, just fewer
  calls.
- `resize` — stretches every row to one common width. Fastest, but short boxes
  get horizontally stretched (1.45x) and long sentences get misread.
- `serial` — one call per box, byte-identical to the original behaviour, slowest.

### CUDA on Windows

`onnxruntime-gpu` **1.19–1.22** targets CUDA 12; 1.23+ requires CUDA 13 and will
report missing `cublasLt64_13.dll`. The runtime DLLs must be reachable — either
install the full CUDA Toolkit (they land on `PATH`) or point at extracted
components:

```bash
set MOBILE_USE_CUDA_DLL_DIR=C:\cuda\cudart\bin;C:\cuda\cublas\bin;C:\cuda\cufft\bin;C:\venv\Lib\site-packages\nvidia\cudnn\bin
```

> [!NOTE]
> On Windows, onnxruntime loads its provider with a native `LoadLibrary`, which
> does **not** honour `os.add_dll_directory` — that is why the code also prepends
> these directories to the process `PATH`.

When the requested backend cannot be initialised — wrong CUDA major version,
missing DLLs, or a `device_id` that does not exist on this machine — onnxruntime
falls back to CPU **silently**. mobile-use therefore verifies
`session.get_providers()` after every build and logs the backend it actually got.
Check `engine.device`, not `config.device`.

## 👨‍💻 Usage

To run mobile-use, simply pass your command as an argument.

**Example 1: Basic Command**

```bash
python ./minitap/mobile_use/main.py "Go to settings and tell me my current battery level"
```

**Example 2: Data Scraping**

Extract specific information and get it back in a structured format. For instance, to get a list of your unread emails:

```bash
python ./minitap/mobile_use/main.py \
  "Open Gmail, find all unread emails, and list their sender and subject line" \
  --output-description "A JSON list of objects, each with 'sender' and 'subject' keys"
```

> [!NOTE]
> If you haven't configured a specific model, mobile-use will prompt you to choose one from the available options.

### 🖥️ Interactive TUI

For hands-on work, `mobile-use-tui` opens a terminal UI instead of exiting after
one goal. It keeps a single agent session alive, so you can send several goals in
a row and watch each one being executed, with a live view of the device screen.

```bash
mobile-use-tui                 # interactive session on a connected device
mobile-use-tui --demo          # UI preview: fake device, no LLM key needed
```

Type a goal in the prompt and press enter. While a goal runs, the status bar
shows a spinner, the agent currently executing and the elapsed time. The panel
on the right reports the connected device (platform, id, screen size), token
spend, elapsed time and progress counters (steps and device actions). Press
`ctrl+t` to stop the running task, `ctrl+l` to clear the timeline, `ctrl+o` to
show the SDK's own logs (they are captured rather than printed over the screen —
warnings and errors are always visible), and `ctrl+c` to quit.

> [!NOTE]
> The token counter shows `n/a` when the configured provider does not report
> usage. Some OpenAI-compatible endpoints — including the MiniMax one this
> project defaults to — omit it from the response entirely, and reporting a
> zero would read as "this run was free".

| Option | Default | Description |
| --- | --- | --- |
| `--demo` | off | Run with a fake device and no LLM key (UI preview). |
| `-d`, `--device-type` | `local` | `local` for connected devices, `cloud` for Minitap cloud devices. |
| `--cloud-platform` | — | Cloud device platform: `android` or `ios`. Required with `--device-type cloud`. |
| `--with-video-recording-tools` | off | Enable AI agents to use video recording tools. |
| `--wda-url` | — | Override the WebDriverAgent URL (iOS). |
| `--wda-auto-start-wda` / `--no-wda-auto-start-wda` | — | Auto-build and run WDA if it is not responding (iOS). |
| `--wda-project-path` | — | Path to `WebDriverAgent.xcodeproj` (iOS). |
| `--idb-host` / `--idb-port` | — | IDB companion host and port (iOS simulators). |

> [!NOTE]
> The TUI needs an interactive terminal. For CI, non-TTY scripts or piped
> output, keep using `mobile-use "<goal>"`. The agents call their models with
> `ainvoke()` rather than streaming, so the TUI shows discrete agent/tool events
> (with durations) rather than a token-by-token stream.

## 🔎 Agentic System Overview

<div align="center">

![Graph Visualization](doc/graph.png)

_This diagram is automatically updated from the codebase. This is our current agentic system architecture._

</div>

## ❤️ Contributing

We love contributions! Whether you're fixing a bug, adding a feature, or improving documentation, your help is welcome. Please read our **[Contributing Guidelines](CONTRIBUTING.md)** to get started.

## ⭐ Star History

<p align="center">
  <a href="https://star-history.dera.page/#minitap-ai/mobile-use&Date">
    <img src="https://star-history.dera.page/svg?repos=minitap-ai/mobile-use&type=Date" alt="Star History Chart" />
  </a>
</p>

## 🏆 Attribution & Licensing

`mobile-use` is the first agentic framework to achieve **100% on the AndroidWorld benchmark**.

This project is licensed under the **Apache License 2.0**.

If you use this code, or are inspired by the architecture used to reach our benchmark results, we kindly request that you credit Minitap, Inc.

### How to Cite
If you use this work in research or a commercial product, please use the following:
> Pierre-Louis Favreau, Jean-Pierre Lo, Clement Guiguet, Charles Simon-Meunier,  
Nicolas Dehandschoewercker, Allen G. Roush, Judah Goldfeder, Ravid Shwartz-Ziv.  
_Do Multi-Agents Dream of Electric Screens? Achieving Perfect Accuracy on AndroidWorld Through Task Decomposition._  
arXiv preprint arXiv:2602.07787 (2026).  
https://arxiv.org/abs/2602.07787

#### Bibtex


```bibtex
@misc{favreau2026multiagentsdreamelectricscreens,
  title        = {Do Multi-Agents Dream of Electric Screens? Achieving Perfect Accuracy on AndroidWorld Through Task Decomposition},
  author       = {Pierre-Louis Favreau and Jean-Pierre Lo and Clement Guiguet and Charles Simon-Meunier and Nicolas Dehandschoewercker and Allen G. Roush and Judah Goldfeder and Ravid Shwartz-Ziv},
  year         = {2026},
  eprint       = {2602.07787},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2602.07787}
}
