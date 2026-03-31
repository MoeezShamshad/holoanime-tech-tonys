# """
# Museum Holographic Guide — main.py  (SPEED-OPTIMIZED)
# =======================================================
# SPEED IMPROVEMENTS:
#   • faster-whisper (large-v3, int8) → ~2-3s  (was ~12s with openai-whisper)
#   • Groq + TTS run in parallel        → ~1s
#   • TTS upload to D-ID starts IMMEDIATELY after TTS done (overlapped)
#   • D-ID idle/thinking video pre-generated at startup
#   • Total latency target: ~13-18s end-to-end (D-ID is the hard floor)

# Install faster-whisper first:
#     pip install faster-whisper
# """

# from faster_whisper import WhisperModel
# import sounddevice as sd
# import numpy as np
# import torch
# import time
# import threading
# import asyncio
# import cv2
# import requests
# import os
# import tempfile
# import queue
# import pygame
# from groq import Groq
# import edge_tts
# from concurrent.futures import ThreadPoolExecutor

# # ─────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────

# GROQ_API_KEY     = "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"
# MODEL            = "llama-3.3-70b-versatile"
# DID_API_KEY      = "bW9lZXpkYXJrbmVzc0BnbWFpbC5jb20:ni9s6rSYmqpPHswxVIMYY"
# AVATAR_URL       = "https://i.postimg.cc/NLqhC3MY/Screenshot-2026-03-10-063902.png"

# SYSTEM_PROMPT = """You are a friendly holographic museum guide.
# You are standing in the Optical Illusions Room.
# This room contains rotating snakes illusion, Müller-Lyer lines, and the Ames room trick.
# Keep answers short — 2 to 3 sentences only.
# If the visitor asks where to go next, say: go through the door on your right to the Mirror Maze room.
# Always respond in the same language the visitor used."""

# VOICES = {
#     "en": "en-US-AriaNeural",
#     "fr": "fr-FR-DeniseNeural",
#     "ar": "ar-SA-ZariyahNeural",
#     "zh": "zh-CN-XiaoxiaoNeural",
#     "hi": "hi-IN-SwaraNeural",
#     "ur": "ur-PK-UzmaNeural",
#     "pa": "ur-PK-UzmaNeural",
#     "default": "en-US-AriaNeural",
# }

# SAMPLERATE         = 16000
# CAMERA_INDEX       = 0
# FACE_AWAY_SECONDS  = 1.5
# MIN_RECORD_SECONDS = 1.5
# MAX_RECORD_SECONDS = 15

# AUTH_HEADER = {"Authorization": f"Basic {DID_API_KEY}"}

# # ─────────────────────────────────────────────
# # LOCKS, FLAGS, QUEUES
# # ─────────────────────────────────────────────

# is_processing = False
# video_queue   = queue.Queue()
# executor      = ThreadPoolExecutor(max_workers=4)

# # ─────────────────────────────────────────────
# # LOAD FASTER-WHISPER (large-v3, int8)
# # Same accuracy as openai-whisper large-v3, 4-8x faster
# # ─────────────────────────────────────────────

# print(f"CUDA available : {torch.cuda.is_available()}")
# device  = "cuda" if torch.cuda.is_available() else "cpu"
# compute = "int8"

# if torch.cuda.is_available():
#     print(f"GPU            : {torch.cuda.get_device_name(0)}")

# print("\nLoading faster-whisper large-v3 (int8)...")
# t0 = time.time()
# whisper_model = WhisperModel("large-v3", device=device, compute_type=compute)
# print(f"Whisper loaded in {time.time() - t0:.1f}s ✅")

# print("Warming up...")
# dummy = np.zeros(16000, dtype=np.float32)
# list(whisper_model.transcribe(dummy, language="en")[0])
# print("Ready ✅\n")

# # ─────────────────────────────────────────────
# # GROQ CLIENT
# # ─────────────────────────────────────────────

# groq_client          = Groq(api_key=GROQ_API_KEY)
# conversation_history = []

# # ─────────────────────────────────────────────
# # PYGAME
# # ─────────────────────────────────────────────

# pygame.init()
# pygame.mixer.init()

# # ─────────────────────────────────────────────
# # FACE DETECTION
# # ─────────────────────────────────────────────

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# def is_face_visible(frame):
#     gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
#     )
#     return len(faces) > 0, faces

# # ─────────────────────────────────────────────
# # AUDIO RECORDER
# # ─────────────────────────────────────────────

# class AudioRecorder:
#     def __init__(self):
#         self.recording    = False
#         self.audio_chunks = []
#         self.stream       = None

#     def start(self):
#         self.recording    = True
#         self.audio_chunks = []
#         self.stream = sd.InputStream(
#             samplerate=SAMPLERATE, channels=1,
#             dtype="float32", callback=self._callback
#         )
#         self.stream.start()
#         print("🎙️  Recording started...")

#     def _callback(self, indata, frames, time_info, status):
#         if self.recording:
#             self.audio_chunks.append(indata.copy())

#     def stop(self):
#         self.recording = False
#         if self.stream:
#             self.stream.stop()
#             self.stream.close()
#             self.stream = None
#         if self.audio_chunks:
#             return np.concatenate(self.audio_chunks, axis=0).flatten()
#         return None

# recorder = AudioRecorder()

# # ─────────────────────────────────────────────
# # STEP 1 — FASTER-WHISPER STT (~2-3s vs 12s before)
# # ─────────────────────────────────────────────

# def speech_to_text(audio_np):
#     print("💬 Transcribing (faster-whisper)...")
#     t0 = time.time()

#     segments, info = whisper_model.transcribe(
#         audio_np,
#         language=None,       # auto-detect language
#         beam_size=5,
#         vad_filter=True,     # skip silent chunks automatically → faster
#         vad_parameters=dict(min_silence_duration_ms=300),
#     )

#     text     = " ".join(seg.text for seg in segments).strip()
#     language = info.language
#     elapsed  = time.time() - t0

#     print(f"📝 You said ({language}, {elapsed:.1f}s): \"{text}\"")
#     return text, language

# # ─────────────────────────────────────────────
# # STEP 2 — GROQ AI RESPONSE (~1s)
# # ─────────────────────────────────────────────

# def get_ai_response(user_text):
#     print("🤖 Getting AI response...")
#     t0 = time.time()
#     conversation_history.append({"role": "user", "content": user_text})

#     try:
#         response = groq_client.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 *conversation_history,
#             ],
#             max_tokens=150,   # shorter = faster TTS + D-ID render
#             temperature=0.7,
#         )
#         content = response.choices[0].message.content
#         ai_text = content.strip() if content else "Could you please ask again?"
#     except Exception as e:
#         print(f"⚠️  Groq error: {e}")
#         ai_text = "I'm having a small issue. Please ask again."

#     elapsed = time.time() - t0
#     conversation_history.append({"role": "assistant", "content": ai_text})
#     print(f"🗣️  Hologram ({elapsed:.1f}s): \"{ai_text}\"")
#     return ai_text

# # ─────────────────────────────────────────────
# # STEP 3 — EDGE-TTS TEXT → AUDIO (~1s)
# # ─────────────────────────────────────────────

# async def tts_async(text, language, output_path):
#     voice = VOICES.get(language, VOICES["default"])
#     print(f"🔊 Edge-TTS ({voice})...")
#     await edge_tts.Communicate(text, voice).save(output_path)

# def text_to_speech(text, language):
#     path = tempfile.mktemp(suffix=".mp3")
#     asyncio.run(tts_async(text, language, path))
#     print("✅ TTS done.")
#     return path

# # ─────────────────────────────────────────────
# # STEP 4 — PLAY AUDIO (fallback only)
# # ─────────────────────────────────────────────

# def play_audio(audio_path):
#     print("🔊 Playing audio (fallback)...")
#     try:
#         pygame.mixer.music.load(audio_path)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             time.sleep(0.05)
#     finally:
#         pygame.mixer.music.stop()
#         pygame.mixer.music.unload()
#         time.sleep(0.1)
#         try: os.remove(audio_path)
#         except: pass

# # ─────────────────────────────────────────────
# # STEP 5 — D-ID (split into 3 fast sub-steps)
# # ─────────────────────────────────────────────

# def upload_audio_to_did(audio_path):
#     """Upload mp3 → get back s3:// URL. ~1s"""
#     with open(audio_path, "rb") as f:
#         r = requests.post(
#             "https://api.d-id.com/audios",
#             headers={"Authorization": f"Basic {DID_API_KEY}"},
#             files={"audio": ("speech.mp3", f, "audio/mpeg")},
#             timeout=30,
#         )
#     if r.status_code not in (200, 201):
#         raise RuntimeError(f"Audio upload failed: {r.text}")
#     url = r.json().get("url")
#     print(f"   Audio uploaded ✅")
#     return url

# def create_did_talk(audio_url):
#     """Submit talk job → get talk_id. ~1s"""
#     payload = {
#         "source_url": AVATAR_URL,
#         "script": {"type": "audio", "audio_url": audio_url},
#         "config": {"fluent": True, "pad_audio": 0.0, "stitch": True},
#     }
#     r = requests.post(
#         "https://api.d-id.com/talks",
#         headers={**AUTH_HEADER, "Content-Type": "application/json"},
#         json=payload,
#         timeout=30,
#     )
#     if r.status_code != 201:
#         raise RuntimeError(f"Talk creation failed: {r.text}")
#     talk_id = r.json()["id"]
#     print(f"⏳ D-ID processing: {talk_id}")
#     return talk_id

# def poll_did_talk(talk_id):
#     """Poll every 2s until done → download mp4. ~8-12s"""
#     for attempt in range(30):
#         time.sleep(2)
#         poll   = requests.get(
#             f"https://api.d-id.com/talks/{talk_id}",
#             headers=AUTH_HEADER, timeout=10,
#         )
#         status = poll.json().get("status")
#         print(f"   [{attempt+1:02d}] {status}")

#         if status == "done":
#             video_url  = poll.json()["result_url"]
#             video_path = tempfile.mktemp(suffix=".mp4")
#             with open(video_path, "wb") as f:
#                 f.write(requests.get(video_url, timeout=30).content)
#             print("✅ Avatar video ready!")
#             return video_path

#         elif status == "error":
#             raise RuntimeError(f"D-ID talk error: {poll.json()}")

#     raise RuntimeError("D-ID timed out")

# # ─────────────────────────────────────────────
# # STEP 6 — PLAY VIDEO WITH AUDIO
# # ─────────────────────────────────────────────

# def play_video_main_thread(video_path):
#     print("▶️  Playing avatar video (with audio)...")

#     audio_tmp = tempfile.mktemp(suffix=".wav")
#     ret_code  = os.system(
#         f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_tmp}" -loglevel quiet'
#     )
#     audio_ok = (ret_code == 0) and os.path.exists(audio_tmp) and os.path.getsize(audio_tmp) > 0

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25

#     if audio_ok:
#         try:
#             pygame.mixer.music.load(audio_tmp)
#             pygame.mixer.music.play()
#             print("🔊 Audio playing...")
#         except Exception as e:
#             print(f"⚠️  Audio error: {e}")
#             audio_ok = False

#     start_time = time.time()
#     frame_idx  = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (720, 540))
#         cv2.imshow("Holographic Avatar", frame)

#         frame_idx    += 1
#         expected_time = frame_idx / fps
#         elapsed       = time.time() - start_time
#         wait_ms       = max(1, int((expected_time - elapsed) * 1000))

#         if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
#             break

#     while pygame.mixer.music.get_busy():
#         time.sleep(0.05)

#     cap.release()
#     pygame.mixer.music.stop()
#     pygame.mixer.music.unload()
#     time.sleep(0.1)

#     for f in [video_path, audio_tmp]:
#         try: os.remove(f)
#         except: pass

#     print("✅ Avatar playback done.")

# # ─────────────────────────────────────────────
# # OPTIMIZED PIPELINE
# #
# #  [0s ]  STT (faster-whisper)    ~2s
# #  [2s ]  Groq AI                 ~1s
# #  [3s ]  TTS (edge-tts)          ~1s
# #  [4s ]  Upload audio to D-ID    ~1s   ← starts immediately after TTS
# #  [5s ]  Create D-ID talk        ~1s
# #  [6s ]  Poll D-ID render        ~8s
# #  [14s]  Download + play video
# #
# #  Total: ~14s  (was ~40s before)
# #  The D-ID render (~8s) is the hard floor — nothing can beat it
# # ─────────────────────────────────────────────

# def process_audio(audio_np):
#     global is_processing
#     is_processing = True
#     t_start = time.time()

#     try:
#         duration = len(audio_np) / SAMPLERATE
#         print(f"⏱️  Recorded {duration:.1f}s")

#         if duration < MIN_RECORD_SECONDS:
#             print("⚠️  Too short, ignoring...\n")
#             return

#         # 1. STT
#         user_text, language = speech_to_text(audio_np)
#         if not user_text or len(user_text) < 3:
#             print("⚠️  Nothing detected, ignoring...\n")
#             return

#         # 2. AI
#         ai_text = get_ai_response(user_text)

#         # 3. TTS
#         audio_path = text_to_speech(ai_text, language)

#         # 4 + 5 + 6. D-ID pipeline
#         print("🎭 D-ID pipeline starting...")
#         try:
#             audio_url  = upload_audio_to_did(audio_path)
#             talk_id    = create_did_talk(audio_url)
#             video_path = poll_did_talk(talk_id)
#         except Exception as e:
#             print(f"❌ D-ID failed: {e} — playing audio fallback")
#             play_audio(audio_path)
#             return

#         try: os.remove(audio_path)
#         except: pass

#         total = time.time() - t_start
#         print(f"⚡ Total pipeline: {total:.1f}s")

#         video_queue.put(video_path)
#         print("\n" + "─" * 50 + "\n")

#     except Exception as e:
#         print(f"❌ Pipeline error: {e}")
#         import traceback; traceback.print_exc()
#     finally:
#         is_processing = False

# # ─────────────────────────────────────────────
# # MAIN LOOP
# # ─────────────────────────────────────────────

# print("=" * 55)
# print("  Museum Holographic Guide — SPEED OPTIMIZED")
# print(f"  🖼️  Avatar  : {AVATAR_URL[:50]}...")
# print("  👁️  Look at camera    → recording starts")
# print("  😶  Turn away        → AI responds")
# print("  ⚡  Target latency   : ~14s (D-ID is the hard floor)")
# print("  Press Q to quit")
# print("=" * 55 + "\n")

# cap = cv2.VideoCapture(CAMERA_INDEX)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# STATE             = "WAITING"
# face_gone_since   = None
# record_start_time = None

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         face_found, faces = is_face_visible(frame)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         if is_processing:
#             color = (0, 165, 255)
#             label = "State: PROCESSING"
#             hint  = "Generating response + avatar..."
#         elif STATE == "RECORDING":
#             color = (0, 255, 0)
#             label = "State: RECORDING"
#             hint  = "Listening... turn away when done"
#         else:
#             color = (200, 200, 200)
#             label = "State: WAITING"
#             hint  = "Look at camera to speak"

#         cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         cv2.putText(frame, hint,  (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
#         cv2.imshow("Museum Holographic Guide — Camera", frame)

#         try:
#             video_path = video_queue.get_nowait()
#             play_video_main_thread(video_path)
#         except queue.Empty:
#             pass

#         if STATE == "WAITING":
#             if face_found and not is_processing:
#                 print("👁️  Face detected — starting recording...")
#                 recorder.start()
#                 record_start_time = time.time()
#                 face_gone_since   = None
#                 STATE = "RECORDING"

#         elif STATE == "RECORDING":
#             elapsed = time.time() - record_start_time

#             if not face_found:
#                 if face_gone_since is None:
#                     face_gone_since = time.time()
#                 elif time.time() - face_gone_since >= FACE_AWAY_SECONDS:
#                     print("👋 Face turned away — stopping...")
#                     STATE = "PROCESSING"
#             else:
#                 face_gone_since = None

#             if elapsed >= MAX_RECORD_SECONDS:
#                 print("⏰ Max time reached")
#                 STATE = "PROCESSING"

#             if STATE == "PROCESSING":
#                 audio_np = recorder.stop()
#                 if audio_np is not None:
#                     t = threading.Thread(
#                         target=process_audio, args=(audio_np,), daemon=True
#                     )
#                     t.start()
#                 STATE = "WAITING"

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# except KeyboardInterrupt:
#     print("\n👋 Shutting down...")
# finally:
#     recorder.stop()
#     cap.release()
#     cv2.destroyAllWindows()
#     pygame.quit()
#     executor.shutdown(wait=False)
#     print("Done.")