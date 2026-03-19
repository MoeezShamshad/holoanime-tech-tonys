import asyncio, io, json, os, queue, tempfile, threading, time
import cv2, numpy as np, sounddevice as sd
from flask import Flask, Response, send_file
from flask_cors import CORS
from groq import Groq
from mutagen.mp3 import MP3

try:
    import edge_tts
    USE_EDGE_TTS = True
except ImportError:
    USE_EDGE_TTS = False

try:
    from faster_whisper import WhisperModel
    import torch
    USE_LOCAL_WHISPER = True
except ImportError:
    USE_LOCAL_WHISPER = False

GROQ_API_KEY       = "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"
MODEL              = "llama-3.3-70b-versatile"
SAMPLERATE         = 16000
CAMERA_INDEX       = 0
FACE_AWAY_SECONDS  = 1.5
MIN_RECORD_SECONDS = 1.5
MAX_RECORD_SECONDS = 15
PORT               = 5000

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
IDLE_VIDEO    = os.path.join(BASE_DIR, "idle.mp4")
TALKING_VIDEO = os.path.join(BASE_DIR, "talking.mp4")

LATEST_AUDIO_PATH = None
LATEST_AUDIO_LOCK = threading.Lock()

# blocks microphone while AI is playing audio
is_speaking = False

# All verified working male voices from edge-tts
EDGE_VOICES = {
    "en": "en-US-GuyNeural",
    "nn": "en-US-GuyNeural",
    "ur": "ur-PK-UzmaNeural",      # Urdu has no male voice
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

SYSTEM_PROMPT = """You are a friendly holographic museum guide.
You are standing in the Optical Illusions Room.
This room contains rotating snakes illusion, Müller-Lyer lines, and the Ames room trick.
Keep answers short — 2 to 3 sentences only.
If the visitor asks where to go next, say: go through the door on your right to the Mirror Maze room.
Always respond in the same language the visitor used and keep your answer short and to the point."""

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
sse_queue = queue.Queue()

groq_client          = Groq(api_key=GROQ_API_KEY)
conversation_history = []

if USE_LOCAL_WHISPER:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\nLoading faster-whisper ({})...".format(device))
        t0 = time.time()
        whisper_model = WhisperModel("large-v3", device=device, compute_type="int8")
        list(whisper_model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
        print("Whisper ready in {:.1f}s".format(time.time() - t0))
    except Exception as e:
        print("Whisper failed — using Groq API:", e)
        USE_LOCAL_WHISPER = False

def mp3_duration(path):
    try:
        audio = MP3(path)
        return audio.info.length
    except:
        return 5.0

def speech_to_text(audio_np):
    if USE_LOCAL_WHISPER:
        segments, info = whisper_model.transcribe(audio_np, language=None, beam_size=3, vad_filter=False)
        text     = " ".join(s.text for s in segments).strip()
        language = info.language
    else:
        import soundfile as sf
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, audio_np, SAMPLERATE)
        with open(tmp, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                file=("audio.wav", f), model="whisper-large-v3", response_format="verbose_json")
        os.remove(tmp)
        text     = result.text.strip()
        language = getattr(result, "language", "en")
    print('STT ({}): "{}"'.format(language, text))
    return text, language

def get_ai_response(user_text):
    conversation_history.append({"role": "user", "content": user_text})
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *conversation_history],
            max_tokens=60, temperature=0.7)
        ai_text = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_text = "I'm having a small issue. Please ask again."
        print("AI error:", e)
    conversation_history.append({"role": "assistant", "content": ai_text})
    print('AI: "{}"'.format(ai_text))
    return ai_text

def text_to_speech_file(text, language):
    global LATEST_AUDIO_PATH, is_speaking

    if not USE_EDGE_TTS:
        return None

    voice = EDGE_VOICES.get(language, "en-US-GuyNeural")
    t0    = time.time()

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
            print("edge-tts returned empty audio for voice:", voice)
            is_speaking = False
            return None

        tmp_path = os.path.join(BASE_DIR, "_tts_audio.mp3")
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        with LATEST_AUDIO_LOCK:
            LATEST_AUDIO_PATH = tmp_path

        print("edge-tts done in {:.2f}s ({} bytes), voice={}".format(
            time.time() - t0, len(audio_bytes), voice))
        return tmp_path

    except Exception as e:
        print("edge-tts error:", e)
        is_speaking = False
        return None

class AudioRecorder:
    def __init__(self):
        self.recording = False; self.chunks = []; self.stream = None

    def start(self):
        self.recording = True; self.chunks = []
        self.stream = sd.InputStream(samplerate=SAMPLERATE, channels=1,
                                     dtype="float32", callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        if self.recording: self.chunks.append(indata.copy())

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        return np.concatenate(self.chunks, axis=0).flatten() if self.chunks else None

recorder      = AudioRecorder()
is_processing = False

def process_audio(audio_np):
    global is_processing, is_speaking
    is_processing = True
    t0 = time.time()
    try:
        if len(audio_np) / SAMPLERATE < MIN_RECORD_SECONDS:
            sse_queue.put(("state", "idle")); return

        sse_queue.put(("state", "thinking"))
        user_text, language = speech_to_text(audio_np)
        if not user_text or len(user_text) < 3:
            sse_queue.put(("state", "idle")); return

        sse_queue.put(("transcript", user_text))
        ai_text    = get_ai_response(user_text)
        audio_path = text_to_speech_file(ai_text, language)
        print("Total pipeline: {:.2f}s".format(time.time() - t0))

        if audio_path:
            sse_queue.put(("speak", json.dumps({
                "text":      ai_text,
                "language":  language,
                "audio_url": "http://localhost:{}/audio?t={}".format(PORT, int(time.time()*1000))
            })))

            # Accurate MP3 duration + 1s cooldown after audio ends
            duration  = mp3_duration(audio_path)
            wait_time = duration + 1.0
            print("Audio duration ~{:.1f}s, waiting {:.1f}s before re-enabling mic".format(
                duration, wait_time))
            time.sleep(wait_time)

        else:
            sse_queue.put(("speak_fallback", json.dumps({
                "text": ai_text, "language": language})))
            time.sleep(5)

        is_speaking = False
        print("Microphone re-enabled")

    except Exception as e:
        print("Pipeline error:", e)
        import traceback; traceback.print_exc()
        sse_queue.put(("state", "idle"))
        is_speaking = False
    finally:
        is_processing = False

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def camera_loop():
    global is_processing
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera not found!"); return

    STATE = "WAITING"; face_gone_since = record_start = None; frame_count = 0
    print("Camera started")
    sse_queue.put(("state", "idle"))

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        face_found = len(faces) > 0
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if is_speaking:
            label, color = "AI SPEAKING — mic disabled", (0, 0, 255)
        elif is_processing:
            label, color = "PROCESSING...", (0, 165, 255)
        elif STATE == "RECORDING":
            label = "RECORDING {:.1f}s — turn away".format(time.time()-record_start); color=(0,255,0)
        else:
            label, color = "WAITING — look at camera", (200, 200, 200)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Museum Guide — Camera", frame)
        cv2.waitKey(1)

        frame_count += 1
        if frame_count % 100 == 0:
            print("faces={} state={} processing={} speaking={}".format(
                len(faces), STATE, is_processing, is_speaking))

        if STATE == "WAITING":
            if face_found and not is_processing and not is_speaking:
                print("Face detected — recording...")
                recorder.start(); record_start = time.time(); face_gone_since = None
                STATE = "RECORDING"; sse_queue.put(("state", "listening"))

        elif STATE == "RECORDING":
            if not face_found:
                if face_gone_since is None: face_gone_since = time.time()
                elif time.time() - face_gone_since >= FACE_AWAY_SECONDS: STATE = "PROCESS"
            else: face_gone_since = None
            if time.time() - record_start >= MAX_RECORD_SECONDS: STATE = "PROCESS"
            if STATE == "PROCESS":
                audio_np = recorder.stop()
                if audio_np is not None:
                    threading.Thread(target=process_audio, args=(audio_np,), daemon=True).start()
                STATE = "WAITING"

    cap.release(); cv2.destroyAllWindows()

@app.route("/")
def index():
    path = os.path.join(BASE_DIR, "index.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    except FileNotFoundError:
        return "<h2>index.html not found</h2>", 404

@app.route("/idle.mp4")
def serve_idle():
    return send_file(IDLE_VIDEO, mimetype="video/mp4")

@app.route("/talking.mp4")
def serve_talking():
    return send_file(TALKING_VIDEO, mimetype="video/mp4")

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

if __name__ == "__main__":
    print("=" * 60)
    print("  Museum Holographic Guide")
    print("  Open http://localhost:{} in Chrome".format(PORT))
    print()
    for name, path in [("idle.mp4", IDLE_VIDEO), ("talking.mp4", TALKING_VIDEO)]:
        print("  [{}] {}".format("OK" if os.path.exists(path) else "MISSING", name))
    print()
    print("  TTS: edge-tts ✓" if USE_EDGE_TTS else "  TTS: MISSING — pip install edge-tts")
    print("  Look at camera → speak → turn away")
    print("=" * 60)
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)