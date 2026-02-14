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
        self.record_seconds = config["record_seconds"]
        self.silence_threshold = config["silence_threshold"]
        self.silence_duration = config["silence_duration"]
        self.pa = pyaudio.PyAudio()

    def read_frames(self, stream, count: int) -> bytes:
        return stream.read(count, exception_on_overflow=False)

    def open_input_stream(self):
        return self.pa.open(
            rate=self.sample_rate,
            channels=self.channels,
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
                data = self.read_frames(stream, self.frame_length)
                frames.append(data)
                samples = struct.unpack_from(f"<{self.frame_length}h", data)
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
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
            wf.setnchannels(self.channels)
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
    """Client for OpenClaw local session API."""

    def __init__(self, config: dict):
        self.base_url = config["base_url"].rstrip("/")
        self.session_key = config["session_key"]
        self.timeout = config.get("timeout", 30)

    def send_message(self, text: str) -> str:
        """Send a message to the OpenClaw session and return the response."""
        # Use sessions_send API: POST to gateway with sessionKey in body
        url = f"{self.base_url}/sessions/send"
        headers = {"Content-Type": "application/json"}
        payload = {
            "sessionKey": self.session_key,
            "message": text
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Response might be in 'response', 'message', or 'text' depending on OpenClaw version
        return data.get("response", data.get("message", data.get("text", str(data))))


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
                pcm = stream.read(self.detector.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from(f"<{self.detector.frame_length}h", pcm)

                keyword_index = self.detector.process(pcm_unpacked)
                if keyword_index >= 0:
                    logger.info("Wake word detected!")
                    stream.stop_stream()
                    try:
                        self._handle_interaction()
                    except Exception:
                        logger.exception("Error during interaction")
                    stream.start_stream()
        finally:
            stream.stop_stream()
            stream.close()
            self._cleanup()

    def _handle_interaction(self):
        # 1. Capture speech
        wav_data = self.audio.capture_speech()
        if not wav_data:
            logger.warning("No audio captured")
            return

        # 2. Transcribe
        logger.info("Transcribing...")
        text = self.transcriber.transcribe(wav_data)
        if not text:
            logger.warning("Empty transcription")
            return
        logger.info("Transcription: %s", text)

        # 3. Send to OpenClaw
        logger.info("Sending to OpenClaw...")
        response = self.openclaw.send_message(text)
        logger.info("OpenClaw response: %s", response[:200])

        # 4. Text-to-speech
        logger.info("Synthesizing speech...")
        audio_data = self.tts.synthesize(response)

        # 5. Play response
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
