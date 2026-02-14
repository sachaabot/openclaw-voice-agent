# OpenClaw Voice Agent

Wake-word activated voice interface for OpenClaw sessions on PamirAI devices.

**Flow:** Listen for wake word → Capture speech → Transcribe with Whisper → Send to OpenClaw → Speak response

## Requirements

- Raspberry Pi CM5 (PamirAI Distiller)
- Microphone and speaker connected
- Python 3.10+ (Distiller SDK venv)
- API keys: Picovoice (wake word), OpenAI (Whisper), OpenClaw

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

### Custom Wake Word

The default config uses the built-in "terminator" keyword. To use a custom "hey openclaw" wake word:

1. Go to [Picovoice Console](https://console.picovoice.ai/)
2. Train a custom keyword for Raspberry Pi
3. Download the `.ppn` file
4. Update `config.yaml`:
   ```yaml
   porcupine:
     keyword_paths:
       - "/path/to/hey-openclaw_en_raspberry-pi.ppn"
   ```

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

## Configuration

See `config.example.yaml` for all options. Key settings:

- `porcupine.access_key` - Free from [Picovoice Console](https://console.picovoice.ai/)
- `whisper.api_key` - OpenAI API key
- `openclaw.base_url` - Local Gateway URL (default: `http://localhost:18789`)
- `openclaw.session_key` - Get from `openclaw sessions list` (no API key needed—OpenClaw runs locally)
- `tts.provider` - Choose `gtts`, `elevenlabs`, or `piper`
- `audio.silence_threshold` - Adjust if it cuts off too early or waits too long
