"""
Wan 2.2 - Image to Video Generator
Generates emotion-based video clips for your museum hologram character
"""
 
import os
import subprocess
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
 
# ── Config ──────────────────────────────────────────────────────────────────
 
IMAGE_PATH   = "/home/moiz-shamshad/holoanime-tech-tonys/holoanime-tech-tonys/character.png"   # your magician character image
OUTPUT_DIR   = "videos"
MODEL_ID     = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"   # 5B = needs ~16GB VRAM (use this)
# MODEL_ID   = "Wan-AI/Wan2.2-I2V-A14B"  # 14B = needs ~80GB VRAM (skip)
 
FPS          = 24
WIDTH        = 832
HEIGHT       = 480
NUM_FRAMES   = 81    # ~3.4 seconds per clip (stitch 9 clips = ~30 seconds)
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── Emotion Prompts ──────────────────────────────────────────────────────────
# Each prompt generates one 3-4 second clip
# We generate multiple clips per emotion and stitch them into a 30sec loop
 
EMOTIONS = {
    "idle": [
        "3D cartoon magician in a dark neon suit, standing relaxed, breathing gently, blinking slowly, subtle idle animation, seamless loop, white background",
        "3D cartoon magician in a dark neon suit, standing still, small natural body sway, calm expression, white background",
        "3D cartoon magician in a dark neon suit, relaxed pose, gentle breathing, slight head movement, white background",
    ],
    "talking_happy": [
        "3D cartoon magician in a dark neon suit, talking happily, big smile, expressive hand gestures, nodding, white background",
        "3D cartoon magician in a dark neon suit, speaking with excitement, laughing, pointing forward, energetic, white background",
        "3D cartoon magician in a dark neon suit, joyful expression, talking enthusiastically, arms open wide, white background",
    ],
    "talking_sad": [
        "3D cartoon magician in a dark neon suit, talking sadly, head slightly down, slow movements, drooping shoulders, white background",
        "3D cartoon magician in a dark neon suit, speaking with sorrow, slow gestures, downcast eyes, white background",
        "3D cartoon magician in a dark neon suit, sad expression, speaking softly, hands clasped, white background",
    ],
    "talking_angry": [
        "3D cartoon magician in a dark neon suit, talking angrily, strong gestures, furrowed brows, pointing finger, white background",
        "3D cartoon magician in a dark neon suit, speaking with frustration, aggressive hand movement, intense expression, white background",
        "3D cartoon magician in a dark neon suit, angry talking, leaning forward, sharp gestures, white background",
    ],
    "thinking": [
        "3D cartoon magician in a dark neon suit, thinking pose, hand on chin, looking up, slow head tilt, white background",
        "3D cartoon magician in a dark neon suit, contemplating, rubbing beard, eyes looking sideways, white background",
        "3D cartoon magician in a dark neon suit, thoughtful expression, slow head movement, tapping chin, white background",
    ],
}
 
# ── Load Model ───────────────────────────────────────────────────────────────
 
def load_pipeline():
    print("Loading Wan 2.2 model... (first time downloads ~10GB)")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()   # saves VRAM
    pipe.enable_vae_slicing()         # saves more VRAM
    print("Model loaded!")
    return pipe
 
# ── Generate Clips ────────────────────────────────────────────────────────────
 
def generate_clip(pipe, image, prompt, output_path):
    print(f"  Generating: {output_path}")
    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, deformed, ugly, text, watermark",
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        guidance_scale=5.0,
        num_inference_steps=30,
    ).frames[0]
    export_to_video(frames, output_path, fps=FPS)
    print(f"  Saved: {output_path}")
 
# ── Stitch Clips into 30sec loop ─────────────────────────────────────────────
 
def stitch_videos(clip_paths, output_path):
    """Stitch multiple clips into one seamless 30sec video using ffmpeg"""
    list_file = output_path.replace(".mp4", "_list.txt")
    with open(list_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")
 
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)
    print(f"Stitched: {output_path}")
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Generation will be very slow on CPU.")
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {torch.cuda.get_device_name(0)} ({vram:.1f}GB VRAM)")
        if vram < 12:
            print("WARNING: Less than 12GB VRAM — may run out of memory.")
            print("Consider using Google Colab with free T4 GPU instead.")
 
    # Load image
    print(f"\nLoading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
 
    # Load model
    pipe = load_pipeline()
 
    # Generate clips for each emotion
    for emotion, prompts in EMOTIONS.items():
        print(f"\nGenerating emotion: {emotion}")
        clip_paths = []
 
        for i, prompt in enumerate(prompts):
            clip_path = os.path.join(OUTPUT_DIR, f"{emotion}_clip_{i+1}.mp4")
            generate_clip(pipe, image, prompt, clip_path)
            clip_paths.append(clip_path)
 
        # Stitch 3 clips = ~10 seconds per emotion
        final_path = os.path.join(OUTPUT_DIR, f"{emotion}.mp4")
        stitch_videos(clip_paths, final_path)
        print(f"Final video ready: {final_path}")
 
    print("\n✅ ALL VIDEOS GENERATED!")
    print("Files ready in:", OUTPUT_DIR)
    print("\nFinal videos:")
    for emotion in EMOTIONS:
        print(f"  {OUTPUT_DIR}/{emotion}.mp4")
 
if __name__ == "__main__":
    main()
 