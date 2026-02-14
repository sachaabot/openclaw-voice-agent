#!/usr/bin/env python3
"""OpenClaw Voice Agent - Wake word activated voice interface for OpenClaw sessions."""

import io
import logging
import logging.handlers
import os
import signal
import struct
import sys
import tempfile
import time
import wave

import pyaudio
import pvporcupine
import requests
import yaml
from openai import OpenAI

logger = logging.getLogger("openclaw-voice-agent")


class LEDManager:
    """Control PamirAI LEDs via sysfs for user feedback."""

    # LED states
    OFF = 0
    BLUE = 200
    GREEN = 200
    RED = 200

    def __init__(self, config: dict = None, led_index: int = 0):
        """Initialize LED manager.
        
        Args:
            config: Optional config dict with 'led' settings
            led_index: Which LED to control (0-6)
        """
        self.led_index = led_index
        self.led_path = f"/sys/class/leds/pamir:led{led_index}/brightness"
        self.enabled = True
        self.current_color = self.OFF
        
        if config and "led" in config:
            self.enabled = config["led"].get("enabled", True)
            self.led_index = config["led"].get("index", led_index)
            self.led_path = f"/sys/class/leds/pamir:led{self.led_index}/brightness"

    def set_color(self, brightness: int) -> bool:
        """Set LED brightness. Returns True if successful."""
        if not self.enabled:
            return True
        
        if not os.path.exists(self.led_path):
            logger.warning("LED path not found: %s", self.led_path)
            self.enabled = False
            return False
        
        try:
            with open(self.led_path, "w") as f:
                f.write(str(brightness))
            self.current_color = brightness
            return True
        except (IOError, OSError) as e:
            logger.error("Failed to write to LED: %s", e)
            return False

    def set_blue(self):
        """Turn LED blue (wake word detected, capturing audio)."""
        self.set_color(self.BLUE)

    def set_green(self):
        """Turn LED green (processing, waiting for response)."""
        self.set_color(self.GREEN)

    def set_red(self):
        """Turn LED red (error state)."""
        self.set_color(self.RED)

    def turn_off(self):
        """Turn LED off."""
        self.set_color(self.OFF)


def get_active_session(base_url: str) -> str | None:
    """Query the local OpenClaw Gateway and return the most recent session key."""
    url = f"{base_url.rstrip('/')}/sessions"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("Failed to fetch sessions from %s: %s", url, e)
        return None

    sessions = data if isinstance(data, list) else data.get("sessions", [])
    if not sessions:
        logger.warning("No active sessions found at %s", url)
        return None

    # Pick the most recently created session
    sessions.sort(key=lambda s: s.get("createdAt", s.get("created_at", "")), reverse=True)
    key = sessions[0].get("sessionKey") or sessions[0].get("session_key") or sessions[0].get("key")
    if key:
        logger.info("Auto-detected session: %s", key)
    else:
        logger.warning("Session found but no key field recognized: %s", sessions[0])
    return key


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    syslog = logging.handlers.SysLogHandler(address="/dev/log")
    syslog.setFormatter(logging.Formatter("openclaw-voice-agent: %(levelname)s %(message)s"))
    logger.addHandler(syslog)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(console)


class AudioManager:
    """Handles microphone input and speaker output via PyAudio."""

    def __init__(self, config: dict):
        self.sample_rate = config["sample_rate"]
        self.frame_length = config["frame_length"]
        self.channels = config["channels"]
        self.hw_channels = config.get("hw_channels", 2)  # PamirAI hardware requires stereo
        self.record_seconds = config["record_seconds"]
        self.silence_threshold = config["silence_threshold"]
        self.silence_duration = config["silence_duration"]
        self.pa = pyaudio.PyAudio()

    def read_frames(self, stream, count: int) -> bytes:
        return stream.read(count, exception_on_overflow=False)

    def read_mono_frames(self, stream, count: int) -> tuple[bytes, tuple]:
        """Read frames from a (possibly stereo) stream and return mono PCM data.
        
        Returns:
            (raw_mono_bytes, mono_samples_tuple)
        """
        raw = stream.read(count, exception_on_overflow=False)
        if self.hw_channels <= 1:
            samples = struct.unpack_from(f"<{count}h", raw)
            return raw, samples

        # Stereo: extract left channel only (every other sample)
        stereo_samples = struct.unpack(f"<{count * self.hw_channels}h", raw)
        mono_samples = stereo_samples[::self.hw_channels]
        mono_bytes = struct.pack(f"<{len(mono_samples)}h", *mono_samples)
        return mono_bytes, mono_samples

    def open_input_stream(self):
        return self.pa.open(
            rate=self.sample_rate,
            channels=self.hw_channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.frame_length,
        )

    def capture_speech(self) -> bytes | None:
        """Record audio until silence detected or max duration reached. Returns WAV bytes."""
        stream = self.open_input_stream()
        frames = []
        silent_chunks = 0
        chunks_per_second = self.sample_rate / self.frame_length
        max_chunks = int(self.record_seconds * chunks_per_second)
        silence_chunks_needed = int(self.silence_duration * chunks_per_second)

        logger.info("Capturing speech...")
        try:
            for _ in range(max_chunks):
                mono_bytes, mono_samples = self.read_mono_frames(stream, self.frame_length)
                frames.append(mono_bytes)
                rms = (sum(s * s for s in mono_samples) / len(mono_samples)) ** 0.5
                if rms < self.silence_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed:
                        logger.debug("Silence detected, stopping capture")
                        break
                else:
                    silent_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)  # Always mono for Whisper
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def play_audio(self, audio_data: bytes):
        """Play WAV or MP3 audio bytes on the speaker."""
        # Write to temp file for playback
        suffix = ".wav" if audio_data[:4] == b"RIFF" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
        try:
            if suffix == ".wav":
                self._play_wav(tmp_path)
            else:
                # Use mpg123 or ffplay for MP3
                os.system(f"mpg123 -q {tmp_path} 2>/dev/null || ffplay -nodisp -autoexit {tmp_path} 2>/dev/null")
        finally:
            os.unlink(tmp_path)

    def _play_wav(self, path: str):
        with wave.open(path, "rb") as wf:
            stream = self.pa.open(
                format=self.pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            chunk = 1024
            data = wf.readframes(chunk)
            while data:
                stream.write(data)
                data = wf.readframes(chunk)
            stream.stop_stream()
            stream.close()

    def close(self):
        self.pa.terminate()


class WakeWordDetector:
    """Porcupine-based wake word detection."""

    def __init__(self, config: dict):
        kwargs = {
            "access_key": config["access_key"],
            "sensitivities": config.get("sensitivities", [0.5]),
        }
        if "keyword_paths" in config and config["keyword_paths"]:
            kwargs["keyword_paths"] = config["keyword_paths"]
        else:
            kwargs["keywords"] = config.get("keywords", ["porcupine"])
        self.porcupine = pvporcupine.create(**kwargs)
        self.frame_length = self.porcupine.frame_length
        self.sample_rate = self.porcupine.sample_rate

    def process(self, pcm_frame) -> int:
        """Process a frame of audio. Returns keyword index (>= 0) if detected, -1 otherwise."""
        return self.porcupine.process(pcm_frame)

    def close(self):
        self.porcupine.delete()


class Transcriber:
    """OpenAI Whisper transcription."""

    def __init__(self, config: dict):
        self.client = OpenAI(api_key=config["api_key"])
        self.model = config.get("model", "whisper-1")
        self.language = config.get("language", "en")

    def transcribe(self, wav_bytes: bytes) -> str:
        buf = io.BytesIO(wav_bytes)
        buf.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=buf,
            language=self.language,
        )
        return response.text.strip()


class OpenClawClient:
    """Client for OpenClaw local gateway (webhook-based agent trigger)."""

    def __init__(self, config: dict):
        self.base_url = config["base_url"].rstrip("/")
        self.session_key = config.get("session_key", "main")
        self.timeout = config.get("timeout", 30)
        self.hook_token = config.get("hook_token", "")

    def send_message(self, text: str) -> str:
        """
        Send a message to OpenClaw via the webhook endpoint.
        
        OpenClaw doesn't expose a direct REST API for sessions_send.
        Instead, we use the /hooks/agent endpoint to trigger an agent run.
        The agent runs and output is logged; for real response, use sessions_history.
        
        For simplicity, we enqueue the message and return a status message.
        """
        if not self.hook_token:
            logger.warning("hook_token not set; falling back to CLI-based send")
            return self._send_via_cli(text)

        url = f"{self.base_url}/hooks/agent"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.hook_token}"
        }
        payload = {
            "message": text,
            "sessionKey": self.session_key,
            "wakeMode": "now",
            "timeoutSeconds": self.timeout,
            "deliver": False  # Don't announce back to a channel
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Message sent to OpenClaw: %s", data.get("runId"))
            return "Message sent to OpenClaw session."
        except requests.RequestException as e:
            logger.error("Failed to send message via webhook: %s", e)
            # Fallback to CLI
            return self._send_via_cli(text)

    def _send_via_cli(self, text: str) -> str:
        """Fallback: use openclaw CLI to send message."""
        try:
            import subprocess
            cmd = ["openclaw", "sessions_send", "--session-key", self.session_key, "--message", text]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.returncode == 0:
                logger.info("Message sent via CLI")
                return "Message sent via OpenClaw CLI."
            else:
                logger.error("CLI send failed: %s", result.stderr)
                return "Failed to send message (hook and CLI both unavailable)"
        except Exception as e:
            logger.error("CLI fallback failed: %s", e)
            return "Failed to send message (hook and CLI both unavailable)"


class TTSEngine:
    """Text-to-speech with multiple provider support."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "gtts")
        self.config = config

    def synthesize(self, text: str) -> bytes:
        if self.provider == "elevenlabs":
            return self._elevenlabs(text)
        if self.provider == "piper":
            return self._piper(text)
        return self._gtts(text)

    def _gtts(self, text: str) -> bytes:
        from gtts import gTTS
        cfg = self.config.get("gtts", {})
        tts = gTTS(text=text, lang=cfg.get("lang", "en"), slow=cfg.get("slow", False))
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()

    def _elevenlabs(self, text: str) -> bytes:
        from elevenlabs import ElevenLabs
        cfg = self.config["elevenlabs"]
        client = ElevenLabs(api_key=cfg["api_key"])
        audio_iter = client.text_to_speech.convert(
            text=text,
            voice_id=cfg.get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
            model_id=cfg.get("model_id", "eleven_monolingual_v1"),
        )
        return b"".join(audio_iter)

    def _piper(self, text: str) -> bytes:
        """Use Piper TTS (local, fast on Raspberry Pi)."""
        model_path = self.config["piper"]["model_path"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            os.system(f'echo "{text}" | piper --model {model_path} --output_file {tmp_path}')
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class VoiceAgent:
    """Main voice agent orchestrator."""

    def __init__(self, config: dict):
        self.running = False
        self.audio = AudioManager(config["audio"])
        self.detector = WakeWordDetector(config["porcupine"])
        self.transcriber = Transcriber(config["whisper"])
        self.openclaw = OpenClawClient(config["openclaw"])
        self.tts = TTSEngine(config["tts"])
        self.led = LEDManager(config)

        # Override audio frame_length to match Porcupine requirements
        self.audio.frame_length = self.detector.frame_length
        self.audio.sample_rate = self.detector.sample_rate

    def run(self):
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Voice agent started, listening for wake word...")
        stream = self.audio.open_input_stream()

        try:
            while self.running:
                _mono_bytes, pcm_unpacked = self.audio.read_mono_frames(stream, self.detector.frame_length)

                keyword_index = self.detector.process(pcm_unpacked)
                if keyword_index >= 0:
                    logger.info("Wake word detected!")
                    self.led.set_blue()  # Turn on blue
                    stream.stop_stream()
                    try:
                        self._handle_interaction()
                    except Exception:
                        logger.exception("Error during interaction")
                        self.led.set_red()  # Turn on red for error
                    finally:
                        self.led.turn_off()  # Turn off when done
                    stream.start_stream()
        finally:
            stream.stop_stream()
            stream.close()
            self._cleanup()

    def _handle_interaction(self):
        # 1. Capture speech (blue LED)
        logger.info("Capturing speech...")
        self.led.set_blue()  # Keep blue while capturing
        wav_data = self.audio.capture_speech()
        if not wav_data:
            logger.warning("No audio captured")
            return

        # 2. Transcribe (still blue)
        logger.info("Transcribing...")
        self.led.set_blue()
        text = self.transcriber.transcribe(wav_data)
        if not text:
            logger.warning("Empty transcription")
            return
        logger.info("Transcription: %s", text)

        # 3. Send to OpenClaw (green LED - waiting for response)
        logger.info("Sending to OpenClaw...")
        self.led.set_green()
        response = self.openclaw.send_message(text)
        logger.info("OpenClaw response: %s", response[:200])

        # 4. Text-to-speech (still green)
        logger.info("Synthesizing speech...")
        self.led.set_green()
        audio_data = self.tts.synthesize(response)

        # 5. Play response (still green)
        logger.info("Playing response...")
        self.audio.play_audio(audio_data)

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False

    def _cleanup(self):
        logger.info("Cleaning up...")
        self.detector.close()
        self.audio.close()


def main():
    config_path = os.environ.get("OPENCLAW_CONFIG", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        print("Copy config.example.yaml to config.yaml and fill in your keys.", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    agent = VoiceAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
