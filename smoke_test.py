"""
smoke_test.py — End-to-end pipeline verification script.

Run this before integrating pipeline.py into the web app to verify
the full pipeline works in your environment.

Usage:
    uv run python smoke_test.py
    python smoke_test.py

Requirements:
    - .env file with REPLICATE_API_TOKEN set
    - An input image file (default: input.webp)
    - prompts_generalized.yaml in the project root
"""

import os
import time

import requests
from dotenv import load_dotenv

import pipeline

load_dotenv()

# --- Configuration ---
IMAGE_PATH = "input.webp"           # swap for any JPEG/PNG/WebP you have
YAML_PATH  = "prompts_generalized.yaml"
POLL_INTERVAL_SECONDS = 5
TIMEOUT_SECONDS = 300

# --- Step 1: Submit the job ---
print("Loading image...")
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

print("Running pipeline (segmentation + Replicate submission)...")
print("  Note: first run downloads the SegFormer model (~100MB). This takes a minute.")

prediction_url = pipeline.run(image_bytes, YAML_PATH)
print(f"  Job submitted.\n  Polling URL: {prediction_url}\n")

# --- Step 2: Poll until done ---
headers = {"Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}"}
deadline = time.time() + TIMEOUT_SECONDS

print("Polling for result (FLUX.1 Fill usually takes 60–180s)...")
while time.time() < deadline:
    response = requests.get(prediction_url, headers=headers).json()
    status = response.get("status")
    print(f"  Status: {status}")

    if status == "succeeded":
        output_url = response["output"][0]
        print(f"\nDone! Output image URL:\n  {output_url}")
        break
    if status in ("failed", "canceled"):
        print(f"\nFailed: {response.get('error')}")
        break

    time.sleep(POLL_INTERVAL_SECONDS)
else:
    print(f"\nTimed out after {TIMEOUT_SECONDS}s. Check the prediction manually at:\n  {prediction_url}")
