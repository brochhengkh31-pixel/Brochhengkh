import os, uuid, shutil, subprocess, tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
client = OpenAI(api_key=API_KEY) if API_KEY else None

ROOT = Path(__file__).parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
app = FastAPI(title="AI Khmer Drama API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_config():
    if client is None:
        raise HTTPException(503, "OPENAI_API_KEY is missing. Set it in backend/.env and restart the server.")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(503, "FFmpeg was not found. Install FFmpeg and make sure it is on PATH.")

def run_ffmpeg(args):
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2500:])
    return result

def srt_time(seconds: float) -> str:
    seconds = max(0, float(seconds))
    h = int(seconds // 3600); seconds -= h * 3600
    m = int(seconds // 60); seconds -= m * 60
    whole = int(seconds); ms = int((seconds - whole) * 1000)
    return f"{h:02}:{m:02}:{whole:02},{ms:03}"

def clean_srt_text(s: str) -> str:
    return s.replace("-->", "→").replace("\n", " ").strip()

@app.get("/api/health")
def health():
    return {"ok": True, "api_key_configured": bool(API_KEY), "ffmpeg_available": shutil.which("ffmpeg") is not None}

@app.post("/api/process")
async def process_video(
    video: UploadFile = File(...),
    source_language: str = Form("auto"),
    make_subtitles: bool = Form(True),
    make_voice: bool = Form(True),
):
    require_config()
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(400, "Unsupported video type. Upload MP4, MOV, MKV, WEBM, AVI, or M4V.")
    job = WORK / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=True)
    input_path = job / f"source{suffix}"
    audio_path = job / "source_audio.mp3"
    srt_path = job / "khmer_subtitles.srt"
    voice_path = job / "khmer_voice.mp3"
    output_path = job / "khmer_drama_export.mp4"

    size = 0
    try:
        with input_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk: break
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB upload limit.")
                out.write(chunk)

        run_ffmpeg(["-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)])
        with audio_path.open("rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=None if source_language == "auto" else source_language,
            )

        segments = getattr(transcription, "segments", None) or []
        if not segments:
            raise RuntimeError("The transcription provider returned no timed segments. Try a shorter/clearer clip.")

        translated = []
        # Translate in batches to reduce API calls; preserve each subtitle's timing.
        for seg in segments:
            original = getattr(seg, "text", "") or ""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate drama dialogue into natural, concise Khmer. Keep names and meaning. Return only the Khmer translation, no notes."},
                    {"role": "user", "content": original},
                ],
                temperature=0.2,
            )
            khmer = response.choices[0].message.content.strip()
            translated.append({
                "start": float(getattr(seg, "start", 0)),
                "end": float(getattr(seg, "end", 0.5)),
                "text": khmer,
            })

        if make_subtitles:
            with srt_path.open("w", encoding="utf-8") as f:
                for i, seg in enumerate(translated, 1):
                    f.write(f"{i}\n{srt_time(seg['start'])} --> {srt_time(max(seg['start'] + 0.2, seg['end']))}\n{clean_srt_text(seg['text'])}\n\n")

        result_files = []
        if make_voice:
            narration = "\n".join(seg["text"] for seg in translated)
            speech = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=narration[:12000],
                instructions="Speak clearly and naturally in Khmer, like a calm drama narrator.",
                response_format="mp3",
            )
            speech.stream_to_file(str(voice_path))
            # Keep the original video; mix in Khmer narration, fitting narration to video duration.
            if make_subtitles:
                run_ffmpeg([
                    "-i", str(input_path), "-i", str(voice_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", f"subtitles={str(srt_path).replace(chr(92), '/')}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-af", "apad", "-shortest", "-c:a", "aac", "-b:a", "128k",
                    str(output_path)
                ])
            else:
                run_ffmpeg([
                    "-i", str(input_path), "-i", str(voice_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-af", "apad", "-shortest", "-c:a", "aac",
                    str(output_path)
                ])
            result_files.append({"name": "MP4 with Khmer voice" + (" and subtitles" if make_subtitles else ""), "url": f"/api/download/{job.name}/video"})
        elif make_subtitles:
            run_ffmpeg([
                "-i", str(input_path), "-vf", f"subtitles={str(srt_path).replace(chr(92), '/')}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "copy",
                str(output_path)
            ])
            result_files.append({"name": "MP4 with Khmer subtitles", "url": f"/api/download/{job.name}/video"})

        if make_subtitles:
            result_files.append({"name": "Khmer subtitles (.srt)", "url": f"/api/download/{job.name}/srt"})
        if make_voice:
            result_files.append({"name": "Khmer narration (.mp3)", "url": f"/api/download/{job.name}/voice"})

        return {"ok": True, "message": "Processing completed. Review the result carefully before publishing.", "files": result_files}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        await video.close()

@app.get("/api/download/{job_id}/{kind}")
def download_result(job_id: str, kind: str):
    # Restrict paths to a job folder directly under WORK.
    if not job_id.isalnum() or len(job_id) != 32:
        raise HTTPException(400, "Invalid job id.")
    job = WORK / job_id
    names = {"video": "khmer_drama_export.mp4", "srt": "khmer_subtitles.srt", "voice": "khmer_voice.mp3"}
    if kind not in names:
        raise HTTPException(404, "Unknown file type.")
    path = job / names[kind]
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path, filename=path.name)


# Serve the built frontend from the same service in production.
from fastapi.staticfiles import StaticFiles
FRONTEND_DIST = ROOT.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
