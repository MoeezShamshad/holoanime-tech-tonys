import os
import asyncio, io, json, queue, tempfile, threading, time
import cv2, numpy as np, sounddevice as sd
from flask import Flask, Response, send_file
from flask_cors import CORS
from groq import Groq
from mutagen.mp3 import MP3

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["DISPLAY"] = ":0"

try:
    import edge_tts
    USE_EDGE_TTS = True
except ImportError:
    USE_EDGE_TTS = False

GROQ_API_KEY       = "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"
MODEL              = "llama-3.1-8b-instant"
SAMPLERATE         = 16000
CAMERA_INDEX       = 0
MIN_RECORD_SECONDS = 1.0
MAX_RECORD_SECONDS = 15
PORT               = 5000

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
IDLE_VIDEO      = os.path.join(BASE_DIR, "idle.mp4")
LISTENING_VIDEO = os.path.join(BASE_DIR, "listening.mp4")

TALKING_VIDEOS = {
    1: os.path.join(BASE_DIR, "talk_1.mp4"),
    2: os.path.join(BASE_DIR, "talk_2.mp4"),
    3: os.path.join(BASE_DIR, "talk_3.mp4"),
    4: os.path.join(BASE_DIR, "talk_4.mp4"),
    5: os.path.join(BASE_DIR, "talk_5.mp4"),
}
TALKING_FALLBACK = os.path.join(BASE_DIR, "talking.mp4")

LATEST_AUDIO_PATH = None
LATEST_AUDIO_LOCK = threading.Lock()
is_speaking       = False

EDGE_VOICES = {
    "en": "en-US-GuyNeural",
    "nn": "en-US-GuyNeural",
    "ur": "ur-PK-UzmaNeural",
    "ar": "ar-SA-HamedNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "es": "es-ES-AlvaroNeural",
    "hi": "hi-IN-MadhurNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "ru": "ru-RU-DmitryNeural",
}

SYSTEM_PROMPT = """You are a friendly holographic museum guide in the Optical Illusions Room.
Room has: rotating snakes, Müller-Lyer lines, Ames room.
Answer 1-2 short sentences only.
If asked where to go next, say: go through the door on your right to Mirror Maze.
Respond in the visitor's language.

Always end your reply with one emotion tag:
[EMOTION:1] happy/welcoming/excited
[EMOTION:2] sad/sympathetic
[EMOTION:3] disappointed/correcting
[EMOTION:4] shocked/surprised
[EMOTION:5] neutral/calm

Examples:
Visitor: "Hello!" → Welcome! Happy to see you! [EMOTION:1]
Visitor: "I am lost" → No worries! Go right to the Mirror Maze room. [EMOTION:2]
Visitor: "Is this magic?" → Not magic, just science tricking your brain. [EMOTION:4]"""

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
sse_queue = queue.Queue()

groq_client          = Groq(api_key=GROQ_API_KEY)
conversation_history = []

# ── Helpers ───────────────────────────────────────────────────
def mp3_duration(path):
    try: return MP3(path).info.length
    except: return 5.0

def parse_emotion(text):
    import re
    match   = re.search(r'\[EMOTION:(\d)\]', text)
    emotion = int(match.group(1)) if match else 5
    clean   = re.sub(r'\s*\[EMOTION:\d\]', '', text).strip()
    return clean, emotion

def get_talking_video_path(emotion):
    path = TALKING_VIDEOS.get(emotion, TALKING_FALLBACK)
    if not os.path.exists(path):
        neutral = TALKING_VIDEOS.get(5, TALKING_FALLBACK)
        return neutral if os.path.exists(neutral) else TALKING_FALLBACK
    return path

# ── STT: Groq Whisper — fast (~1-2s), free, multilingual ─────
def speech_to_text(audio_np):
    import soundfile as sf
    try:
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, audio_np, SAMPLERATE)
        with open(tmp, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                file=("audio.wav", f),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        os.remove(tmp)
        text     = result.text.strip()
        language = getattr(result, "language", "en")
        print('[Groq STT] ({}): "{}"'.format(language, text))
        return text, language
    except Exception as e:
        print("Groq STT failed:", e)
        return "", "en"

# ── AI Response ───────────────────────────────────────────────
def get_ai_response(user_text):
    conversation_history.append({"role": "user", "content": user_text})
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation_history],
            max_tokens=60, temperature=0.7)
        ai_text = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_text = "I'm having a small issue. Please ask again. [EMOTION:3]"
        print("AI error:", e)
    conversation_history.append({"role": "assistant", "content": ai_text})
    print('AI: "{}"'.format(ai_text))
    return ai_text

# ── TTS ───────────────────────────────────────────────────────
def text_to_speech_file(text, language):
    global LATEST_AUDIO_PATH, is_speaking
    if not USE_EDGE_TTS: return None

    voice = EDGE_VOICES.get(language, "en-US-GuyNeural")

    async def _synth():
        buf  = io.BytesIO()
        comm = edge_tts.Communicate(text, voice)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    try:
        is_speaking = True
        audio_bytes = asyncio.run(_synth())
        if not audio_bytes:
            is_speaking = False
            return None
        tmp_path = os.path.join(BASE_DIR, "_tts_audio.mp3")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        with LATEST_AUDIO_LOCK:
            LATEST_AUDIO_PATH = tmp_path
        print("edge-tts done ({} bytes), voice={}".format(len(audio_bytes), voice))
        return tmp_path
    except Exception as e:
        print("edge-tts error:", e)
        is_speaking = False
        return None

# ── Audio Recorder with VAD ───────────────────────────────────
class AudioRecorder:
    SILENCE_THRESHOLD = 0.008   # RMS below this = silence
    SILENCE_SECONDS   = 1.8     # stop after 1.8s of silence
    BLOCK_SIZE        = 1600    # 0.1s per block at 16kHz

    def __init__(self):
        self.recording       = False
        self.chunks          = []
        self.stream          = None
        self.silence_frames  = 0
        self.speech_detected = False
        self.done_event      = threading.Event()

    def start(self):
        self.recording       = True
        self.chunks          = []
        self.silence_frames  = 0
        self.speech_detected = False
        self.done_event.clear()
        self.stream = sd.InputStream(
            samplerate=SAMPLERATE, channels=1,
            dtype="float32", callback=self._cb,
            blocksize=self.BLOCK_SIZE
        )
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        if not self.recording:
            return
        self.chunks.append(indata.copy())
        volume = np.abs(indata).mean()

        if volume > self.SILENCE_THRESHOLD:
            self.speech_detected = True
            self.silence_frames  = 0
        elif self.speech_detected:
            self.silence_frames += 1
            silence_so_far = self.silence_frames * (self.BLOCK_SIZE / SAMPLERATE)
            if silence_so_far >= self.SILENCE_SECONDS:
                self.recording = False
                self.done_event.set()
                print("VAD: silence detected — sending to STT")

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.chunks, axis=0).flatten() if self.chunks else None

recorder      = AudioRecorder()
is_processing = False

# ── Audio Pipeline ────────────────────────────────────────────
def process_audio(audio_np):
    global is_processing, is_speaking
    is_processing = True
    t0 = time.time()
    try:
        if len(audio_np) / SAMPLERATE < MIN_RECORD_SECONDS:
            print("Audio too short — ignoring")
            sse_queue.put(("state", "idle")); return

        sse_queue.put(("state", "thinking"))
        user_text, language = speech_to_text(audio_np)
        if not user_text or len(user_text) < 3:
            sse_queue.put(("state", "idle")); return

        sse_queue.put(("transcript", user_text))
        ai_raw           = get_ai_response(user_text)
        ai_text, emotion = parse_emotion(ai_raw)
        talk_path        = get_talking_video_path(emotion)
        audio_path       = text_to_speech_file(ai_text, language)

        emotion_names = {1:"happy", 2:"sad", 3:"disappointed", 4:"shocked", 5:"neutral"}
        print("Pipeline: {:.2f}s | emotion={} ({}) | video={}".format(
            time.time() - t0, emotion, emotion_names.get(emotion, "?"),
            os.path.basename(talk_path)))

        if audio_path:
            sse_queue.put(("speak", json.dumps({
                "text":          ai_text,
                "language":      language,
                "emotion":       emotion,
                "talking_video": "/video/talk/{}".format(emotion),
                "audio_url":     "/audio?t={}".format(int(time.time() * 1000))
            })))
            duration  = mp3_duration(audio_path)
            wait_time = duration + 1.0
            print("Waiting {:.1f}s before re-enabling mic".format(wait_time))
            time.sleep(wait_time)
        else:
            sse_queue.put(("speak_fallback", json.dumps({
                "text": ai_text, "language": language})))
            time.sleep(5)

        is_speaking = False
        sse_queue.put(("state", "idle"))
        print("Microphone re-enabled")

    except Exception as e:
        print("Pipeline error:", e)
        import traceback; traceback.print_exc()
        sse_queue.put(("state", "idle"))
        is_speaking = False
    finally:
        is_processing = False

# ── Camera Loop ───────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def camera_loop():
    global is_processing
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera not found!"); return

    cv2.namedWindow("Museum Guide — Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Museum Guide — Camera", 640, 480)

    STATE = "WAITING"; record_start = None; frame_count = 0
    print("Camera started")
    sse_queue.put(("state", "idle"))

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces      = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        face_found = len(faces) > 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if is_speaking:
            label, color = "AI SPEAKING — mic disabled", (0, 0, 255)
        elif is_processing:
            label, color = "PROCESSING...", (0, 165, 255)
        elif STATE == "RECORDING":
            label = "RECORDING {:.1f}s — stop speaking to send".format(
                time.time() - record_start); color = (0, 255, 0)
        else:
            label, color = "WAITING — look at camera to speak", (200, 200, 200)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Museum Guide — Camera", frame)
        cv2.waitKey(1)

        frame_count += 1
        if frame_count % 150 == 0:
            print("faces={} state={} processing={} speaking={}".format(
                len(faces), STATE, is_processing, is_speaking))

        if STATE == "WAITING":
            if face_found and not is_processing and not is_speaking:
                print("Face detected — recording...")
                recorder.start()
                record_start = time.time()
                STATE        = "RECORDING"
                sse_queue.put(("state", "listening"))

        elif STATE == "RECORDING":
            vad_done    = recorder.done_event.is_set()
            max_reached = time.time() - record_start >= MAX_RECORD_SECONDS
            face_left   = not face_found

            if vad_done or max_reached or face_left:
                reason = "VAD" if vad_done else ("max time" if max_reached else "face left")
                print("Recording ended ({}) — processing...".format(reason))
                audio_np = recorder.stop()
                if audio_np is not None:
                    threading.Thread(
                        target=process_audio, args=(audio_np,), daemon=True).start()
                STATE = "WAITING"

    cap.release(); cv2.destroyAllWindows()

# ── Flask Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    path = os.path.join(BASE_DIR, "index.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    except FileNotFoundError:
        return "<h2>index.html not found</h2>", 404

@app.route("/video/idle")
def serve_idle():
    if os.path.exists(IDLE_VIDEO):
        return send_file(IDLE_VIDEO, mimetype="video/mp4")
    return "idle.mp4 not found", 404

@app.route("/video/listening")
def serve_listening():
    if os.path.exists(LISTENING_VIDEO):
        return send_file(LISTENING_VIDEO, mimetype="video/mp4")
    if os.path.exists(IDLE_VIDEO):
        return send_file(IDLE_VIDEO, mimetype="video/mp4")
    return "listening.mp4 not found", 404

@app.route("/video/talk/<int:emotion>")
def serve_talk(emotion):
    path = get_talking_video_path(emotion)
    if os.path.exists(path):
        return send_file(path, mimetype="video/mp4")
    return "talking video not found", 404

@app.route("/audio")
def serve_audio():
    with LATEST_AUDIO_LOCK:
        path = LATEST_AUDIO_PATH
    if path and os.path.exists(path):
        return send_file(path, mimetype="audio/mpeg", conditional=False)
    return "No audio yet", 404

@app.route("/events")
def events():
    def generate():
        yield "data: {}\n\n".format(json.dumps({"type": "connected"}))
        while True:
            try:
                etype, payload = sse_queue.get(timeout=25)
                yield "data: {}\n\n".format(json.dumps({"type": etype, "data": payload}))
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
            except GeneratorExit:
                break
    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        "Connection": "keep-alive", "Access-Control-Allow-Origin": "*",
    })

# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    emotion_names = {1:"happy", 2:"sad", 3:"disappointed", 4:"shocked", 5:"neutral"}
    print("=" * 60)
    print("  Museum Holographic Guide")
    print("  Local:  http://localhost:{}".format(PORT))
    print("  Mobile: use your ngrok URL")
    print()
    print("  [{}] idle.mp4".format("OK" if os.path.exists(IDLE_VIDEO) else "MISSING"))
    print("  [{}] listening.mp4".format(
        "OK" if os.path.exists(LISTENING_VIDEO) else "MISSING — will use idle"))
    print()
    print("  Talking videos:")
    for n, name in emotion_names.items():
        path   = TALKING_VIDEOS[n]
        status = "OK" if os.path.exists(path) else "MISSING — will use fallback"
        print("    [{}] talk_{}.mp4  ({})".format(status, n, name))
    print()
    print("  TTS : edge-tts ✓" if USE_EDGE_TTS else "  TTS : MISSING")
    print("  STT : Groq Whisper (fast ~1-2s, free, multilingual)")
    print("  VAD : auto-trigger after 1.8s silence")
    print("  LLM : llama-3.1-8b-instant (fastest model)")
    print("=" * 60)
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)