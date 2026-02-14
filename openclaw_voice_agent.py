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
    """Control PamirAI RGB LEDs via sysfs for user feedback.

    Uses hardware-driven modes (static/fade/rainbow) for smooth animation
    instead of software threads. This avoids choppiness and thread-leak bugs.

    Sysfs layout per LED:
      /sys/class/leds/pamir:led{N}/{red,green,blue,brightness,mode,timing}

    Modes: static, blink, fade, rainbow
    Timing: animation period in ms (default 500)
    """

    def __init__(self, config: dict = None):
        self.enabled = True
        self.led_count = 7  # PamirAI has LEDs 0-6

        if config and "led" in config:
            self.enabled = config["led"].get("enabled", True)
            self.led_count = config["led"].get("count", 7)

        self.led_bases = [f"/sys/class/leds/pamir:led{i}" for i in range(self.led_count)]
        logger.info("LEDManager: enabled=%s, leds=%d", self.enabled, self.led_count)

    def _write(self, led_base: str, attr: str, value):
        """Write a value to an LED sysfs attribute."""
        path = f"{led_base}/{attr}"
        try:
            with open(path, "w") as f:
                f.write(str(value))
        except (IOError, OSError):
            import subprocess
            subprocess.run(
                ["sudo", "bash", "-c", f"echo {value} > {path}"],
                timeout=2, capture_output=True,
            )

    # Valid timing values supported by the SAM driver
    VALID_TIMINGS = [100, 200, 500, 1000]

    def _set_all(self, r: int, g: int, b: int, brightness: int = None,
                 mode: str = "static", timing: int = None):
        """Set all LEDs to given color, mode, and optional timing.

        Write order matters for smooth transitions:
        1. Set timing first (so driver has correct period before animating)
        2. Set RGB color (so driver fades the intended color)
        3. Set brightness
        4. Set mode last (starts animation with correct color+timing)
        """
        if not self.enabled:
            return
        if brightness is None:
            brightness = 255 if (r or g or b) else 0
        # Snap to nearest valid timing value
        if timing is not None:
            timing = min(self.VALID_TIMINGS, key=lambda x: abs(x - timing))
        for base in self.led_bases:
            if timing is not None:
                self._write(base, "timing", timing)
            self._write(base, "red", r)
            self._write(base, "green", g)
            self._write(base, "blue", b)
            self._write(base, "brightness", brightness)
            self._write(base, "mode", mode)

    def set_rgb(self, r: int, g: int, b: int):
        """Set ALL LEDs to a static RGB color."""
        logger.debug("LED rgb(%d, %d, %d)", r, g, b)
        self._set_all(r, g, b, mode="static")

    def start_animation(self):
        """Start a smooth rainbow color cycle using kernel-driven animation mode.

        Uses the SAM driver's 'rainbow' mode which cycles through all colors
        (including green→cyan→blue) entirely in firmware — no CPU overhead,
        no sudo, and perfectly smooth. Timing controls cycle speed.
        """
        if not self.enabled:
            return
        logger.debug("LED rainbow animation started (mode)")
        self._set_all(0, 0, 0, brightness=255, mode="rainbow", timing=100)

    def stop_animation(self):
        """Stop animation and turn off LEDs."""
        self.turn_off()
        logger.debug("LED rainbow animation stopped")

    def set_green(self):
        """Green: listening / capturing audio (static)."""
        self.set_rgb(0, 255, 0)

    def set_red(self):
        """Red: error state."""
        self.set_rgb(255, 0, 0)

    def turn_off(self):
        """Turn all LEDs off: clear triggers, stop animations, zero brightness."""
        if not self.enabled:
            return
        for base in self.led_bases:
            self._write(base, "trigger", "none")
            self._write(base, "mode", "static")
            self._write(base, "brightness", 0)


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
        self.min_capture_seconds = config.get("min_capture_seconds", 3.0)
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

    def calibrate_noise_floor(self, stream, duration: float = 0.5) -> float:
        """Measure ambient noise RMS over a short duration.
        
        Returns:
            Average ambient RMS (used to compute relative silence threshold).
        """
        chunks_per_second = self.sample_rate / self.frame_length
        num_chunks = int(duration * chunks_per_second)
        rms_values = []

        for _ in range(num_chunks):
            _mono_bytes, mono_samples = self.read_mono_frames(stream, self.frame_length)
            rms = (sum(s * s for s in mono_samples) / len(mono_samples)) ** 0.5
            rms_values.append(rms)

        if not rms_values:
            return 150.0  # fallback

        avg_rms = sum(rms_values) / len(rms_values)
        max_rms = max(rms_values)
        logger.info("Noise calibration: avg_rms=%.0f max_rms=%.0f (samples=%d)",
                     avg_rms, max_rms, len(rms_values))
        return avg_rms

    def capture_speech(self, stream=None) -> bytes | None:
        """Record audio until silence detected or max duration reached. Returns WAV bytes.
        
        Uses a rolling average approach: speech is detected when RMS rises
        significantly above the ambient noise floor. Silence is when RMS drops
        back to near-ambient levels for silence_duration seconds.
        
        Args:
            stream: Optional existing PyAudio stream to reuse (avoids device conflicts).
                    If None, opens a new stream (and closes it when done).
        """
        own_stream = stream is None
        if own_stream:
            stream = self.open_input_stream()

        # Calibrate ambient noise level
        ambient_rms = self.calibrate_noise_floor(stream)
        
        # Silence threshold: ambient + small fixed offset
        # From the logs: ambient ~148, speech ~240-264
        # So we set threshold at ambient + 30 (enough to catch the drop-off)
        silence_offset = max(ambient_rms * 0.15, 15)  # 15% above ambient or at least 15
        effective_threshold = ambient_rms + silence_offset
        logger.info("Silence detection: ambient=%.0f threshold=%.0f (ambient+%.0f)",
                     ambient_rms, effective_threshold, silence_offset)

        frames = []
        silent_chunks = 0
        chunks_per_second = self.sample_rate / self.frame_length
        max_chunks = int(self.record_seconds * chunks_per_second)
        silence_chunks_needed = int(self.silence_duration * chunks_per_second)

        # Don't check for silence until min_capture_seconds elapsed
        min_capture_chunks = int(self.min_capture_seconds * chunks_per_second)

        logger.info("Capturing speech (max %.0fs, silence after %.1fs of quiet, grace=%.1fs)...",
                     self.record_seconds, self.silence_duration, self.min_capture_seconds)
        try:
            for chunk_idx in range(max_chunks):
                mono_bytes, mono_samples = self.read_mono_frames(stream, self.frame_length)
                frames.append(mono_bytes)
                rms = (sum(s * s for s in mono_samples) / len(mono_samples)) ** 0.5

                # Log RMS every ~0.5s for debugging
                if chunk_idx % max(1, int(chunks_per_second * 0.5)) == 0:
                    elapsed = chunk_idx / chunks_per_second
                    logger.debug("t=%.1fs rms=%.0f threshold=%.0f silent=%d/%d",
                                 elapsed, rms, effective_threshold,
                                 silent_chunks, silence_chunks_needed)

                # Skip silence detection during grace period
                if chunk_idx < min_capture_chunks:
                    continue

                if rms < effective_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed:
                        logger.info("Silence detected after %.1fs (%.1fs of quiet), stopping",
                                    chunk_idx / chunks_per_second,
                                    silent_chunks / chunks_per_second)
                        break
                else:
                    silent_chunks = 0
        finally:
            if own_stream:
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
        """Play WAV or MP3 audio bytes on the speaker using aplay."""
        import subprocess

        is_wav = audio_data[:4] == b"RIFF"
        suffix = ".wav" if is_wav else ".mp3"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name

        try:
            if is_wav:
                # Play WAV directly with aplay
                subprocess.run(["aplay", "-D", "default", tmp_path],
                               capture_output=True, timeout=30)
            else:
                # Convert MP3 to WAV first, then play
                wav_path = tmp_path + ".wav"
                # Try sox, then mpg123, then python fallback
                converted = False

                # Try sox
                if not converted:
                    try:
                        result = subprocess.run(
                            ["sox", tmp_path, "-r", "16000", "-c", "2", wav_path],
                            capture_output=True, timeout=10)
                        if result.returncode == 0:
                            converted = True
                    except FileNotFoundError:
                        pass

                # Try mpg123 decode
                if not converted:
                    try:
                        result = subprocess.run(
                            ["mpg123", "-w", wav_path, tmp_path],
                            capture_output=True, timeout=10)
                        if result.returncode == 0:
                            converted = True
                    except FileNotFoundError:
                        pass

                # Python fallback: use pydub if available, or raw play
                if not converted:
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_mp3(tmp_path)
                        audio.export(wav_path, format="wav")
                        converted = True
                    except ImportError:
                        logger.error("Cannot play MP3: install sox, mpg123, or pydub")
                        return

                if converted and os.path.exists(wav_path):
                    subprocess.run(["aplay", "-D", "default", wav_path],
                                   capture_output=True, timeout=30)
                    os.unlink(wav_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

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
    """Client for OpenClaw local gateway via CLI."""

    DEFAULT_VOICE_PROMPT = (
        "[Voice conversation] Respond in 1-2 short sentences. "
        "Be concise and conversational -- this is spoken aloud, not read. "
        "No markdown, no bullet points, no lists."
    )

    def __init__(self, config: dict):
        self.base_url = config["base_url"].rstrip("/")
        self.session_id = config.get("session_id", "")
        self.agent_id = config.get("agent_id", "main")
        self.timeout = config.get("timeout", 60)
        self.voice_prompt = config.get("voice_prompt", self.DEFAULT_VOICE_PROMPT)

    def send_message(self, text: str) -> str:
        """Send a message to OpenClaw via the `openclaw agent` CLI and return the response."""
        import subprocess

        # Prepend voice prompt to guide concise, spoken-style responses
        prefixed_text = f"{self.voice_prompt}\n\n{text}" if self.voice_prompt else text
        cmd = ["openclaw", "agent", "--message", prefixed_text, "--agent", self.agent_id]

        # Route to specific session if set
        if self.session_id:
            cmd.extend(["--session-id", self.session_id])

        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10,  # Extra buffer over agent timeout
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.error("openclaw agent failed (rc=%d): %s", result.returncode, stderr)
                # Try to extract useful text from stderr
                return stderr or "Sorry, I couldn't process that."

            stdout = result.stdout.strip()
            if not stdout:
                return "No response received."
            return stdout

        except subprocess.TimeoutExpired:
            logger.error("openclaw agent timed out after %ds", self.timeout)
            return "Sorry, the request timed out."
        except FileNotFoundError:
            logger.error("openclaw CLI not found in PATH")
            return "OpenClaw CLI not installed."
        except Exception as e:
            logger.error("Failed to run openclaw agent: %s", e)
            return "Sorry, something went wrong."


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
                    self.led.set_green()  # Turn on green
                    try:
                        self._handle_interaction(stream)
                    except Exception:
                        logger.exception("Error during interaction")
                        self.led.set_red()  # Red for error
                        time.sleep(1)  # Show red briefly
                    finally:
                        self.led.turn_off()  # Turn off when done
        finally:
            stream.stop_stream()
            stream.close()
            self._cleanup()

    def _handle_interaction(self, stream):
        # 1. Capture speech (green LED)
        logger.info("Capturing speech...")
        self.led.set_green()
        wav_data = self.audio.capture_speech(stream=stream)
        if not wav_data:
            logger.warning("No audio captured")
            return

        # 2. Transcribe (still green)
        logger.info("Transcribing...")
        self.led.set_green()
        text = self.transcriber.transcribe(wav_data)
        if not text:
            logger.warning("Empty transcription")
            return
        logger.info("Transcription: %s", text)

        # 3-5. Start animation once and keep it running through the
        #       entire response cycle (OpenClaw → TTS → playback) to
        #       avoid flicker from repeated stop/start.
        self.led.start_animation()

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

        self.led.stop_animation()
        logger.info("Interaction complete.")

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
