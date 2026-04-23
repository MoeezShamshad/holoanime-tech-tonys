import os
import asyncio, io, json, queue, threading, time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from groq import Groq
from mutagen.mp3 import MP3

load_dotenv()

try:
    import edge_tts
    USE_EDGE_TTS = True
except ImportError:
    USE_EDGE_TTS = False

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = os.getenv("MODEL", "llama-3.1-8b-instant")
PORT         = 5000

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
IDLE_VIDEO      = os.path.join(BASE_DIR, "idle.mp4")
LISTENING_VIDEO = os.path.join(BASE_DIR, "listening.mp4")
THINKING_VIDEO  = os.path.join(BASE_DIR, "thinking.mp4")

TALKING_VIDEOS = {
    1: os.path.join(BASE_DIR, "talk_1.mp4"),
    2: os.path.join(BASE_DIR, "talk_2.mp4"),
    3: os.path.join(BASE_DIR, "talk_3.mp4"),
    4: os.path.join(BASE_DIR, "talk_4.mp4"),
    5: os.path.join(BASE_DIR, "talk_5.mp4"),
}
TALKING_FALLBACK = os.path.join(BASE_DIR, "talking.mp4")

executor      = ThreadPoolExecutor(max_workers=6)
sessions      = {}
sessions_lock = threading.Lock()

# ── Default System Prompt ──────────────────────────────────────
# Locked suffix — always appended, never removable, kept minimal for speed
LOCKED_SUFFIX = (
    "Max 2 sentences. Match visitor tone: "
    "[EMOTION:1]friendly [EMOTION:2]sad [EMOTION:3]angry/rude [EMOTION:4]surprised [EMOTION:5]neutral. "
    "End reply with one silent tag. Never speak the tag."
)

# Base default prompt — kept short for speed
_BASE_PROMPT = (
    "Holographic museum guide, Optical Illusions Room (rotating snakes, Müller-Lyer lines, Ames room). "
    "Next: Mirror Maze, door right. Reply in visitor's language. "
    "Developer: Moiz Shamshad."
)

SYSTEM_PROMPT = _BASE_PROMPT + "\n" + LOCKED_SUFFIX

# ── Complete voice map ─────────────────────────────────────────
EDGE_VOICES = {
    "af": "af-ZA-AdriNeural",           "afrikaans":   "af-ZA-AdriNeural",
    "am": "am-ET-AmehaNeural",          "amharic":     "am-ET-AmehaNeural",
    "ar": "ar-SA-HamedNeural",          "arabic":      "ar-SA-HamedNeural",
    "az": "az-AZ-BabekNeural",          "azerbaijani": "az-AZ-BabekNeural",
    "bg": "bg-BG-BorislavNeural",       "bulgarian":   "bg-BG-BorislavNeural",
    "bn": "bn-BD-PradeepNeural",        "bengali":     "bn-BD-PradeepNeural",
                                         "bangla":      "bn-BD-PradeepNeural",
    "bs": "bs-BA-GoranNeural",          "bosnian":     "bs-BA-GoranNeural",
    "ca": "ca-ES-EnricNeural",          "catalan":     "ca-ES-EnricNeural",
    "cs": "cs-CZ-AntoninNeural",        "czech":       "cs-CZ-AntoninNeural",
    "cy": "cy-GB-AledNeural",           "welsh":       "cy-GB-AledNeural",
    "da": "da-DK-JeppeNeural",          "danish":      "da-DK-JeppeNeural",
    "de": "de-DE-ConradNeural",         "german":      "de-DE-ConradNeural",
    "el": "el-GR-NestorasNeural",       "greek":       "el-GR-NestorasNeural",
    "en": "en-US-GuyNeural",            "english":     "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",         "spanish":     "es-ES-AlvaroNeural",
    "et": "et-EE-KertNeural",           "estonian":    "et-EE-KertNeural",
    "eu": "eu-ES-AitorNeural",          "basque":      "eu-ES-AitorNeural",
    "fa": "fa-IR-FaridNeural",          "persian":     "fa-IR-FaridNeural",
                                         "farsi":       "fa-IR-FaridNeural",
    "fi": "fi-FI-HarriNeural",          "finnish":     "fi-FI-HarriNeural",
    "fil": "fil-PH-AngeloNeural",       "filipino":    "fil-PH-AngeloNeural",
                                         "tagalog":     "fil-PH-AngeloNeural",
    "fr": "fr-FR-HenriNeural",          "french":      "fr-FR-HenriNeural",
    "ga": "ga-IE-ColmNeural",           "irish":       "ga-IE-ColmNeural",
    "gl": "gl-ES-RoiNeural",            "galician":    "gl-ES-RoiNeural",
    "gu": "gu-IN-NiranjanNeural",       "gujarati":    "gu-IN-NiranjanNeural",
    "he": "he-IL-AvriNeural",           "hebrew":      "he-IL-AvriNeural",
    "hi": "hi-IN-MadhurNeural",         "hindi":       "hi-IN-MadhurNeural",
    "hr": "hr-HR-SreckoNeural",         "croatian":    "hr-HR-SreckoNeural",
    "hu": "hu-HU-TamasNeural",          "hungarian":   "hu-HU-TamasNeural",
    "hy": "hy-AM-HaykNeural",           "armenian":    "hy-AM-HaykNeural",
    "id": "id-ID-ArdiNeural",           "indonesian":  "id-ID-ArdiNeural",
    "is": "is-IS-GunnarNeural",         "icelandic":   "is-IS-GunnarNeural",
    "it": "it-IT-DiegoNeural",          "italian":     "it-IT-DiegoNeural",
    "ja": "ja-JP-KeitaNeural",          "japanese":    "ja-JP-KeitaNeural",
    "jv": "jv-ID-DimasNeural",          "javanese":    "jv-ID-DimasNeural",
    "ka": "ka-GE-GiorgiNeural",         "georgian":    "ka-GE-GiorgiNeural",
    "kk": "kk-KZ-DauletNeural",         "kazakh":      "kk-KZ-DauletNeural",
    "km": "km-KH-PisethNeural",         "khmer":       "km-KH-PisethNeural",
                                         "cambodian":   "km-KH-PisethNeural",
    "kn": "kn-IN-GaganNeural",          "kannada":     "kn-IN-GaganNeural",
    "ko": "ko-KR-InJoonNeural",         "korean":      "ko-KR-InJoonNeural",
    "lo": "lo-LA-ChanthavongNeural",    "lao":         "lo-LA-ChanthavongNeural",
    "lt": "lt-LT-LeonasNeural",         "lithuanian":  "lt-LT-LeonasNeural",
    "lv": "lv-LV-NilsNeural",           "latvian":     "lv-LV-NilsNeural",
    "mk": "mk-MK-AleksandarNeural",     "macedonian":  "mk-MK-AleksandarNeural",
    "ml": "ml-IN-MidhunNeural",         "malayalam":   "ml-IN-MidhunNeural",
    "mn": "mn-MN-BataaNeural",          "mongolian":   "mn-MN-BataaNeural",
    "mr": "mr-IN-ManoharNeural",        "marathi":     "mr-IN-ManoharNeural",
    "ms": "ms-MY-OsmanNeural",          "malay":       "ms-MY-OsmanNeural",
    "mt": "mt-MT-JosephNeural",         "maltese":     "mt-MT-JosephNeural",
    "my": "my-MM-ThihaNeural",          "burmese":     "my-MM-ThihaNeural",
                                         "myanmar":     "my-MM-ThihaNeural",
    "nb": "nb-NO-FinnNeural",           "norwegian":   "nb-NO-FinnNeural",
    "ne": "ne-NP-SagarNeural",          "nepali":      "ne-NP-SagarNeural",
    "nl": "nl-NL-MaartenNeural",        "dutch":       "nl-NL-MaartenNeural",
    "nn": "nb-NO-FinnNeural",
    "pa": "pa-IN-OjasNeural",           "punjabi":     "pa-IN-OjasNeural",
    "pl": "pl-PL-MarekNeural",          "polish":      "pl-PL-MarekNeural",
    "ps": "ps-AF-GulNawazNeural",       "pashto":      "ps-AF-GulNawazNeural",
    "pt": "pt-BR-AntonioNeural",        "portuguese":  "pt-BR-AntonioNeural",
    "ro": "ro-RO-EmilNeural",           "romanian":    "ro-RO-EmilNeural",
    "ru": "ru-RU-DmitryNeural",         "russian":     "ru-RU-DmitryNeural",
    "si": "si-LK-SameeraNeural",        "sinhala":     "si-LK-SameeraNeural",
    "sk": "sk-SK-LukasNeural",          "slovak":      "sk-SK-LukasNeural",
    "sl": "sl-SI-RokNeural",            "slovenian":   "sl-SI-RokNeural",
    "so": "so-SO-MuuseNeural",          "somali":      "so-SO-MuuseNeural",
    "sq": "sq-AL-IlirNeural",           "albanian":    "sq-AL-IlirNeural",
    "sr": "sr-RS-NicholasNeural",       "serbian":     "sr-RS-NicholasNeural",
    "su": "su-ID-JajangNeural",         "sundanese":   "su-ID-JajangNeural",
    "sv": "sv-SE-MattiasNeural",        "swedish":     "sv-SE-MattiasNeural",
    "sw": "sw-KE-RafikiNeural",         "swahili":     "sw-KE-RafikiNeural",
    "ta": "ta-IN-ValluvarNeural",       "tamil":       "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",          "telugu":      "te-IN-MohanNeural",
    "th": "th-TH-NiwatNeural",          "thai":        "th-TH-NiwatNeural",
    "tr": "tr-TR-AhmetNeural",          "turkish":     "tr-TR-AhmetNeural",
    "uk": "uk-UA-OstapNeural",          "ukrainian":   "uk-UA-OstapNeural",
    "ur": "ur-PK-SalmanNeural",         "urdu":        "ur-PK-SalmanNeural",
    "uz": "uz-UZ-SardorNeural",         "uzbek":       "uz-UZ-SardorNeural",
    "vi": "vi-VN-NamMinhNeural",        "vietnamese":  "vi-VN-NamMinhNeural",
    "zh": "zh-CN-YunxiNeural",          "chinese":     "zh-CN-YunxiNeural",
                                         "mandarin":    "zh-CN-YunxiNeural",
    "zu": "zu-ZA-ThembaNeural",         "zulu":        "zu-ZA-ThembaNeural",
}

def normalize_language(lang: str) -> str:
    if not lang:
        return "en"
    key = lang.strip().lower()
    if key in EDGE_VOICES:
        return key
    if "-" in key:
        prefix = key.split("-")[0]
        if prefix in EDGE_VOICES:
            return prefix
    if key[:2] in EDGE_VOICES:
        return key[:2]
    print("WARNING: unknown language '{}' — falling back to English".format(lang))
    return "en"

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Session helpers ────────────────────────────────────────────
def get_session(sid: str) -> dict:
    with sessions_lock:
        if sid not in sessions:
            sessions[sid] = {
                "history":       [],
                "audio_path":    None,
                "sse_queue":     queue.Queue(),
                "last_seen":     time.time(),
                "custom_prompt": None,        # ← NEW: per-session custom prompt
            }
        sessions[sid]["last_seen"] = time.time()
        return sessions[sid]

def cleanup_sessions():
    while True:
        time.sleep(300)
        cutoff = time.time() - 1800
        with sessions_lock:
            dead = [s for s, v in sessions.items() if v["last_seen"] < cutoff]
            for s in dead:
                audio = sessions[s].get("audio_path")
                if audio and os.path.exists(audio):
                    try: os.remove(audio)
                    except: pass
                sessions.pop(s, None)
                print("Cleaned up session:", s)

threading.Thread(target=cleanup_sessions, daemon=True).start()

# ── Helpers ────────────────────────────────────────────────────
def mp3_duration(data: bytes) -> float:
    try:    return MP3(io.BytesIO(data)).info.length
    except: return 5.0

def parse_emotion(text: str):
    import re
    match   = re.search(r'\[EMOTION:(\d)\]', text)
    emotion = int(match.group(1)) if match else 5

    # Remove the [EMOTION:X] tag
    clean = re.sub(r'\s*\[EMOTION:\d\]', '', text).strip()

    # Remove any stray emotion label words the AI might accidentally say
    # e.g. "...about you! [happy]" or "...neutral" at end of sentence
    clean = re.sub(
        r'\b(happy|sad|neutral|shocked|disappointed|welcoming|excited|sympathetic|correcting|calm)\b',
        '', clean, flags=re.IGNORECASE
    ).strip()

    # Clean up any double spaces or trailing punctuation left behind
    clean = re.sub(r'  +', ' ', clean).strip(' .,!;:')

    # Remove developer WhatsApp number from spoken text (silent info only)
    clean = re.sub(r'\+?923216452306', '', clean).strip()
    clean = re.sub(r'WhatsApp\s*:?\s*\+?\d+', '', clean, flags=re.IGNORECASE).strip()

    return clean, emotion

def get_talking_video_path(emotion: int) -> str:
    path = TALKING_VIDEOS.get(emotion, TALKING_FALLBACK)
    if not os.path.exists(path):
        neutral = TALKING_VIDEOS.get(5, TALKING_FALLBACK)
        return neutral if os.path.exists(neutral) else TALKING_FALLBACK
    return path

# ── STT ────────────────────────────────────────────────────────
def speech_to_text(audio_bytes: bytes, content_type: str = "audio/webm"):
    try:
        ext_map = {
            "audio/webm": ".webm", "audio/wav": ".wav",
            "audio/ogg":  ".ogg",  "audio/mp4": ".mp4",
            "audio/mpeg": ".mp3",
        }
        mime      = content_type.split(";")[0].strip()
        ext       = ext_map.get(mime, ".webm")
        audio_buf = io.BytesIO(audio_bytes)
        result    = groq_client.audio.transcriptions.create(
            file=("audio" + ext, audio_buf),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        text         = result.text.strip()
        raw_language = getattr(result, "language", "en")
        language     = normalize_language(raw_language)
        print('[STT] lang="{}" → "{}" | "{}"'.format(raw_language, language, text))
        return text, language
    except Exception as e:
        print("STT error:", e)
        return "", "en"

# ── LLM ────────────────────────────────────────────────────────
def get_ai_response(user_text: str, history: list, active_prompt: str) -> str:
    history.append({"role": "user", "content": user_text})
    try:
        resp    = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": active_prompt}, *history],
            max_tokens=60,
            temperature=0.7,
        )
        ai_text = resp.choices[0].message.content.strip()
    except Exception as e:
        ai_text = "I'm having a small issue. Please ask again. [EMOTION:3]"
        print("LLM error:", e)
    history.append({"role": "assistant", "content": ai_text})
    print('LLM: "{}"'.format(ai_text))
    return ai_text

# ── TTS ────────────────────────────────────────────────────────
async def text_to_speech(text: str, language: str, sid: str):
    if not USE_EDGE_TTS:
        return None, None

    voice = EDGE_VOICES.get(normalize_language(language), "en-US-GuyNeural")
    print("TTS: lang='{}' → voice='{}'".format(language, voice))

    try:
        buf  = io.BytesIO()
        comm = edge_tts.Communicate(text, voice)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio_bytes = buf.getvalue()
        if not audio_bytes:
            return None, None

        path = os.path.join(BASE_DIR, "_tts_{}.mp3".format(sid))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_file, path, audio_bytes)

        get_session(sid)["audio_path"] = path
        print("TTS done ({} bytes)".format(len(audio_bytes)))
        return audio_bytes, path

    except Exception as e:
        print("TTS error:", e)
        return None, None

def _write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

# ── Audio Pipeline ─────────────────────────────────────────────
async def process_audio_pipeline(audio_bytes: bytes, content_type: str, sid: str):
    sess = get_session(sid)
    q    = sess["sse_queue"]
    t0   = time.time()

    # ── Pick active prompt: custom (from browser) or default ──
    active_prompt = sess.get("custom_prompt") or SYSTEM_PROMPT

    try:
        q.put(("state", "thinking"))

        loop      = asyncio.get_event_loop()
        user_text, language = await loop.run_in_executor(
            executor, speech_to_text, audio_bytes, content_type
        )
        if not user_text or len(user_text) < 2:
            q.put(("state", "idle"))
            return

        q.put(("transcript", user_text))

        # Pass active_prompt into LLM
        ai_raw = await loop.run_in_executor(
            executor, get_ai_response, user_text, sess["history"], active_prompt
        )
        ai_text, emotion = parse_emotion(ai_raw)

        audio_data, audio_path = await text_to_speech(ai_text, language, sid)

        emotion_names = {1:"happy",2:"sad",3:"disappointed",4:"shocked",5:"neutral"}
        print("Pipeline {:.2f}s | emotion={} ({})".format(
            time.time() - t0, emotion, emotion_names.get(emotion, "?")))

        if audio_path:
            duration = mp3_duration(audio_data)
            q.put(("speak", json.dumps({
                "text":          ai_text,
                "language":      language,
                "emotion":       emotion,
                "talking_video": "/video/talk/{}".format(emotion),
                "audio_url":     "/audio/{}?t={}".format(sid, int(time.time() * 1000)),
                "duration":      duration,
            })))
        else:
            q.put(("speak_fallback", json.dumps({
                "text": ai_text, "language": language, "emotion": emotion,
                "talking_video": "/video/talk/{}".format(emotion),
            })))

    except Exception as e:
        print("Pipeline error:", e)
        import traceback; traceback.print_exc()
        q.put(("state", "idle"))

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return Response(content="<h2>index.html not found</h2>", status_code=404)

# ── NEW: endpoint to update system prompt for a session ───────
@app.post("/set-prompt/{sid}")
async def set_prompt(sid: str, request: Request):
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if prompt:
        sess = get_session(sid)
        # Always force-append locked lines — user cannot override these
        if LOCKED_SUFFIX not in prompt:
            prompt = prompt.rstrip() + "\n" + LOCKED_SUFFIX
        sess["custom_prompt"] = prompt
        print("[PROMPT] Session {} updated prompt ({} chars)".format(sid, len(prompt)))
        return JSONResponse({"ok": True, "chars": len(prompt)})
    return JSONResponse({"ok": False, "reason": "empty prompt"}, status_code=400)

# ── Transcribe ────────────────────────────────────────────────
@app.post("/transcribe/{sid}")
async def transcribe(sid: str, request: Request):
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"ok": False, "reason": "no audio data"}, status_code=400)

    content_type = request.headers.get("content-type", "audio/webm")
    asyncio.create_task(process_audio_pipeline(audio_bytes, content_type, sid))
    return JSONResponse({"ok": True})

# ── Done ──────────────────────────────────────────────────────
@app.post("/done/{sid}")
async def done(sid: str):
    get_session(sid)["sse_queue"].put(("state", "idle"))
    return JSONResponse({"ok": True})

# ── Videos ────────────────────────────────────────────────────
@app.get("/video/idle")
async def serve_idle():
    if os.path.exists(IDLE_VIDEO):
        return FileResponse(IDLE_VIDEO, media_type="video/mp4")
    return Response(content="idle.mp4 not found", status_code=404)

@app.get("/video/listening")
async def serve_listening():
    if os.path.exists(LISTENING_VIDEO):
        return FileResponse(LISTENING_VIDEO, media_type="video/mp4")
    if os.path.exists(IDLE_VIDEO):
        return FileResponse(IDLE_VIDEO, media_type="video/mp4")
    return Response(content="listening.mp4 not found", status_code=404)

@app.get("/video/thinking")
async def serve_thinking():
    if os.path.exists(THINKING_VIDEO):
        return FileResponse(THINKING_VIDEO, media_type="video/mp4")
    if os.path.exists(LISTENING_VIDEO):
        return FileResponse(LISTENING_VIDEO, media_type="video/mp4")
    if os.path.exists(IDLE_VIDEO):
        return FileResponse(IDLE_VIDEO, media_type="video/mp4")
    return Response(content="thinking.mp4 not found", status_code=404)

@app.get("/video/talk/{emotion}")
async def serve_talk(emotion: int):
    path = get_talking_video_path(emotion)
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return Response(content="talking video not found", status_code=404)

# ── Audio ─────────────────────────────────────────────────────
@app.get("/audio/{sid}")
async def serve_audio(sid: str):
    path = get_session(sid).get("audio_path")
    if path and os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    return Response(content="No audio yet", status_code=404)

# ── SSE ───────────────────────────────────────────────────────
@app.get("/events/{sid}")
async def events(sid: str):
    q = get_session(sid)["sse_queue"]

    async def generate():
        yield "data: {}\n\n".format(json.dumps({"type": "connected", "sid": sid}))
        while True:
            loop = asyncio.get_event_loop()
            try:
                etype, payload = await loop.run_in_executor(
                    None, lambda: q.get(timeout=25)
                )
                yield "data: {}\n\n".format(json.dumps({"type": etype, "data": payload}))
            except Exception:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Connection":                  "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )

# ── Static files ──────────────────────────────────────────────
@app.get("/{filename}")
async def static_files(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return Response(content="Not found", status_code=404)

# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    emotion_names = {1:"happy",2:"sad",3:"disappointed",4:"shocked",5:"neutral"}
    print("=" * 60)
    print("  Museum Holographic Guide  —  FastAPI / MULTI-USER")
    print("  Local:  http://localhost:{}".format(PORT))
    print()
    print("  [{}] idle.mp4".format("OK" if os.path.exists(IDLE_VIDEO) else "MISSING"))
    print("  [{}] listening.mp4".format(
        "OK" if os.path.exists(LISTENING_VIDEO) else "MISSING"))
    print("  [{}] thinking.mp4".format(
        "OK" if os.path.exists(THINKING_VIDEO) else "MISSING"))
    print()
    for n, name in emotion_names.items():
        path   = TALKING_VIDEOS[n]
        status = "OK" if os.path.exists(path) else "MISSING"
        print("  [{}] talk_{}.mp4  ({})".format(status, n, name))
    print()
    print("  TTS : edge-tts ✓" if USE_EDGE_TTS else "  TTS : MISSING — pip install edge-tts")
    print("  STT : whisper-large-v3-turbo (Groq)")
    print("  LLM : {}".format(MODEL))
    print("  Server: FastAPI + uvicorn")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT)