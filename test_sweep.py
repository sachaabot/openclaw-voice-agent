#!/usr/bin/env python3
"""Test green-blue horizontal sweep animation on PamirAI LEDs."""
import math
import subprocess
import time
import signal
import sys

LED_COUNT = 7
running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def write_led(led_idx, attr, value):
    path = f"/sys/class/leds/pamir:led{led_idx}/{attr}"
    try:
        with open(path, "w") as f:
            f.write(str(value))
    except (IOError, OSError):
        subprocess.run(["bash", "-c", f"echo {value} > {path}"],
                       timeout=2, capture_output=True)

def set_led_rgb(led_idx, r, g, b):
    write_led(led_idx, "red", r)
    write_led(led_idx, "green", g)
    write_led(led_idx, "blue", b)
    write_led(led_idx, "brightness", 255 if (r or g or b) else 0)

def all_off():
    for i in range(LED_COUNT):
        write_led(i, "trigger", "none")
        set_led_rgb(i, 0, 0, 0)

# Oscillating sweep: a "wave" moves across LEDs 0-6, 
# transitioning each LED between green and blue based on
# a sine wave phase offset per LED position.
# 
# At any moment, each LED has a blend between green(0,255,0) and blue(0,0,255).
# The blend is determined by: t = (sin(phase + offset) + 1) / 2
# where offset spreads across LED positions.

print("Running green-blue sweep animation. Ctrl+C to stop.")

# Make sure triggers are set to none
for i in range(LED_COUNT):
    write_led(i, "trigger", "none")

step = 0
SPEED = 0.03  # seconds between frames
PHASE_SPREAD = math.pi * 2 / LED_COUNT  # full wave across all LEDs

try:
    while running:
        phase = step * 0.15  # controls speed of oscillation
        for i in range(LED_COUNT):
            # Sine wave determines blend: 0=full green, 1=full blue
            t = (math.sin(phase + i * PHASE_SPREAD) + 1.0) / 2.0
            g = int(255 * (1.0 - t))
            b = int(255 * t)
            set_led_rgb(i, 0, g, b)
        step += 1
        time.sleep(SPEED)
finally:
    all_off()
    print("\nLEDs off.")
