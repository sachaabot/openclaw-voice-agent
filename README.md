# OpenClaw Voice Agent

Wake-word activated voice interface for OpenClaw sessions on PamirAI devices.

**Flow:** Listen for wake word → Capture speech → Transcribe with Whisper → Send to OpenClaw → Speak response

## Requirements

- Raspberry Pi CM5 (PamirAI Distiller)
- Microphone and speaker connected
- Python 3.10+ (Distiller SDK venv)
- OpenClaw local gateway running (see below)
- API keys: Picovoice (wake word), OpenAI (Whisper), AI provider for OpenClaw

## Setup

```bash
cd /home/distiller/Projects/openclaw-voice-agent

# Install dependencies into SDK venv
source /opt/distiller-sdk/activate.sh
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys and settings
```

### Porcupine Wake Word Setup

**Get Your Access Key (Free)**

1. Visit [Picovoice Console](https://console.picovoice.ai/) and sign up (no credit card needed)
2. Copy your **AccessKey** from the dashboard
3. Set it in `config.yaml`:
   ```yaml
   porcupine:
     access_key: "YOUR_PICOVOICE_ACCESS_KEY"
   ```

**Option 1: Built-in Keywords (Easiest)**

Use one of the pre-trained keywords:
- `"terminator"`, `"porcupine"`, `"bumblebee"`, `"americano"`, `"blueberry"`, and more

Works offline and is ready to go:
```yaml
porcupine:
  keywords:
    - "terminator"
  sensitivities:
    - 0.6
```

**Option 2: Custom Wake Word (Better UX)**

Train a custom wake word like "hey openclaw":

1. Go to [Picovoice Console](https://console.picovoice.ai/)
2. Click **"Create Custom Keyword"**
3. Enter your phrase (e.g., "hey openclaw")
4. Select your platform: **Raspberry Pi (32-bit ARM)** (for PamirAI/CM5)
5. Train the model (takes ~1 min)
6. Download the `.ppn` file
7. Copy it to your project directory (e.g., `models/hey-openclaw_en_raspberry-pi.ppn`)
8. Update `config.yaml`:
   ```yaml
   porcupine:
     keyword_paths:
       - "/home/distiller/Projects/openclaw-voice-agent/models/hey-openclaw_en_raspberry-pi.ppn"
   ```

**Fine-tune Detection**

Adjust sensitivity (0.0–1.0) if needed:
- Higher = catches quieter speech but more false positives
- Lower = requires louder/clearer speech
- Default (0.6) works well for most cases

### TTS Providers

| Provider | Quality | Speed | Offline | Setup |
|----------|---------|-------|---------|-------|
| `gtts` | OK | Slow | No | None (default) |
| `elevenlabs` | Excellent | Fast | No | API key required |
| `piper` | Good | Fast | Yes | Download model |

## Running

```bash
# Direct
source /opt/distiller-sdk/activate.sh
python3 openclaw_voice_agent.py

# Or with custom config path
OPENCLAW_CONFIG=/path/to/config.yaml python3 openclaw_voice_agent.py
```

## Systemd Service

```bash
# Install service
sudo cp openclaw-voice-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable openclaw-voice-agent
sudo systemctl start openclaw-voice-agent

# Management
sudo systemctl status openclaw-voice-agent
sudo systemctl stop openclaw-voice-agent
journalctl -u openclaw-voice-agent -f   # View logs
```

## OpenClaw Setup

OpenClaw is a **local, self-hosted Node.js gateway** that connects chat platforms to AI agents. It does NOT require an OpenClaw API key; instead it uses your AI provider credentials (Anthropic, OpenAI, etc.).

**Prerequisites:**
- Node.js 22+
- AI provider API key (Anthropic, OpenAI, etc.)

**Start OpenClaw:**

```bash
# Install (if not already done)
npm install -g openclaw

# Configure with your AI provider key
# Edit ~/.openclaw/openclaw.json with your credentials

# Start the gateway
openclaw gateway start

# Verify it's running
openclaw status
# Should show: Gateway is running on http://localhost:18789

# List available sessions
openclaw sessions list
```

**Documentation:**
- [OpenClaw Docs](https://docs.openclaw.ai)
- [GitHub](https://github.com/openclaw/openclaw)

## Configuration

See `config.example.yaml` for all options. Key settings:

- `porcupine.access_key` - Free from [Picovoice Console](https://console.picovoice.ai/)
- `whisper.api_key` - OpenAI API key
- `openclaw.base_url` - Local Gateway URL (default: `http://localhost:18789`)
- `openclaw.session_key` - Leave empty to auto-detect, or set to a specific session key from `openclaw sessions list`
- `tts.provider` - Choose `gtts`, `elevenlabs`, or `piper`
- `audio.silence_threshold` - Adjust if it cuts off too early or waits too long

**Important:** The voice agent does NOT need an OpenClaw API key. OpenClaw runs locally and uses your AI provider credentials (configured in `~/.openclaw/openclaw.json`).
