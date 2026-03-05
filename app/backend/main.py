from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.backend.infer import Predictor

BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="P-SmokeNet Demo")

here = Path(__file__).parent

VIDEOS_DIR = BASE_DIR / "artifacts" / "videos"
pred = Predictor(base_dir=str(BASE_DIR))

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

templates = Jinja2Templates(directory=str(here / "templates"))
app.mount("/static", StaticFiles(directory=str(here / "static")), name="static")

pred = Predictor(base_dir=BASE_DIR)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/videos")
def api_videos(split: str = "test"):
    return JSONResponse({"split": split, "videos": pred.list_videos(split=split)})

@app.get("/api/clips")
def api_clips(video_id: str, split: str = "test", limit: int = 200):
    return JSONResponse({"video_id": video_id, "split": split, "clips": pred.list_clips(video_id=video_id, split=split, limit=limit)})

@app.get("/api/predict")
def api_predict(video_id: str, clip_start: int, clip_end: int, threshold: float = 0.5):
    out = pred.predict_clip(video_id=video_id, clip_start=clip_start, clip_end=clip_end, threshold=threshold)
    return JSONResponse(out)
    
@app.get("/api/predict_series")
def api_predict_series(video_id: str, t_start: float = 0.0, t_end: float = 360.0, step_s: float = 2.0, threshold: float = 0.5):
    fps = pred.fps
    T = pred.T

    clips = pred.list_clips(video_id=video_id, split="test", limit=100000)

    clips_sorted = sorted(clips, key=lambda x: float(x["t"]))
    times = [float(c["t"]) for c in clips_sorted]

    def nearest_idx(x):
        best_i = 0
        best_d = float("inf")
        for i, tt in enumerate(times):
            d = abs(tt - x)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    out = []
    t = float(t_start)
    while t <= float(t_end):
        if len(clips_sorted) > 0:
            c = clips_sorted[nearest_idx(t)]
            res = pred.predict_clip(video_id, int(c["clip_start"]), int(c["clip_end"]), threshold=threshold)
            out.append({
                "t": float(c["t"]),
                "clip_start": int(c["clip_start"]),
                "clip_end": int(c["clip_end"]),
                "prob": float(res["prob"]),
                "decision": int(res["decision"]),
                "ground_truth": res["ground_truth"],
            })
        else:
            end_f = int(round(t * fps))
            start_f = max(0, end_f - T)
            res = pred.predict_clip(video_id, start_f, end_f, threshold=threshold)
            out.append({
                "t": t,
                "clip_start": int(start_f),
                "clip_end": int(end_f),
                "prob": float(res["prob"]),
                "decision": int(res["decision"]),
                "ground_truth": res["ground_truth"],
            })
        t += float(step_s)

    return JSONResponse({"video_id": video_id, "t_start": t_start, "t_end": t_end, "step_s": step_s, "threshold": threshold, "series": out})