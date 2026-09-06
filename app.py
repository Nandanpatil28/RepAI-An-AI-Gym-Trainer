"""Flask web interface for the existing AI Fitness Trainer pose pipeline."""

from __future__ import annotations

import base64
import io
import os
import secrets
import socket
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import qrcode
from flask import Flask, abort, flash, g, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from exercise_engine import ALL_EXERCISES, WorkoutEngine
from video_analyzer import VideoAnalyzer

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "fitness.db"
MAX_FRAME_BYTES = 1_500_000
engines = {}
video_analyses = {}
mobile_camera_sessions = {}
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "webm"}

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", os.urandom(32)),
    DATABASE=str(DATABASE_PATH),
    MAX_CONTENT_LENGTH=200 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(app.config["DATABASE"])
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS workout_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            pushups INTEGER NOT NULL DEFAULT 0,
            pullups INTEGER NOT NULL DEFAULT 0,
            situps INTEGER NOT NULL DEFAULT 0,
            squats INTEGER NOT NULL DEFAULT 0,
            walking INTEGER NOT NULL DEFAULT 0,
            total_reps INTEGER NOT NULL DEFAULT 0,
            video_filename TEXT,
            duration_seconds REAL,
            source_type TEXT NOT NULL DEFAULT 'live',
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    columns = {row[1] for row in db.execute("PRAGMA table_info(workout_sessions)")}
    for name, definition in (("video_filename", "TEXT"), ("duration_seconds", "REAL"),
                             ("source_type", "TEXT NOT NULL DEFAULT 'live'")):
        if name not in columns:
            db.execute(f"ALTER TABLE workout_sessions ADD COLUMN {name} {definition}")
    db.commit()
    db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def active_engine():
    workout_id = session.get("workout_id")
    return engines.get(workout_id) if workout_id else None


def active_analysis(analysis_id):
    analysis = video_analyses.get(analysis_id)
    if analysis is None or analysis["user_id"] != session.get("user_id"):
        return None
    return analysis


def allowed_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def local_network_ip():
    """Best-effort LAN IP lookup; MOBILE_CAMERA_HOST can override it."""
    configured = os.environ.get("MOBILE_CAMERA_HOST")
    if configured:
        return configured
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def mobile_camera_scheme():
    return "https" if os.environ.get("FLASK_USE_HTTPS") == "1" else "http"


def clean_mobile_sessions():
    now = datetime.now(timezone.utc)
    for token, item in list(mobile_camera_sessions.items()):
        if item["expires_at"] <= now:
            item["engine"].close()
            del mobile_camera_sessions[token]


def owned_mobile_session(token):
    clean_mobile_sessions()
    item = mobile_camera_sessions.get(token)
    if item is None or item["user_id"] != session.get("user_id"):
        return None
    return item


def mobile_session(token):
    clean_mobile_sessions()
    return mobile_camera_sessions.get(token)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.route("/")
def index():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirmation = request.form.get("confirm_password", "")
        if not name or len(name) > 80:
            flash("Please enter a name up to 80 characters.", "error")
        elif not email or len(email) > 254 or "@" not in email:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirmation:
            flash("Passwords do not match.", "error")
        else:
            try:
                db = get_db()
                db.execute("INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                           (name, email, generate_password_hash(password), utcnow()))
                db.commit()
                flash("Account created. Please sign in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    end_engine(save=False)
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    totals = db.execute("""SELECT COALESCE(SUM(pushups), 0) pushups, COALESCE(SUM(pullups), 0) pullups,
        COALESCE(SUM(situps), 0) situps, COALESCE(SUM(squats), 0) squats,
        COALESCE(SUM(walking), 0) walking, COALESCE(SUM(total_reps), 0) total_reps
        FROM workout_sessions WHERE user_id = ?""", (session["user_id"],)).fetchone()
    recent = db.execute("SELECT * FROM workout_sessions WHERE user_id = ? ORDER BY ended_at DESC LIMIT 5",
                        (session["user_id"],)).fetchall()
    return render_template("dashboard.html", totals=totals, recent=recent)


@app.route("/workout")
@login_required
def workout():
    return render_template("workout.html")


@app.route("/mobile-camera/connect")
@login_required
def mobile_camera_connection():
    """Desktop page; a QR session is created only after this page is opened."""
    return render_template("mobile_connection.html")


@app.post("/api/mobile-camera/create")
@login_required
def create_mobile_camera():
    clean_mobile_sessions()
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    mobile_camera_sessions[token] = {
        "user_id": session["user_id"], "engine": WorkoutEngine(), "image": None,
        "connected": False, "connected_ip": None, "last_seen": None,
        "expires_at": expires_at, "saved": False,
        "started_at": utcnow(),
    }
    host = local_network_ip()
    port = int(os.environ.get("FLASK_PORT", 5000))
    mobile_url = f"{mobile_camera_scheme()}://{host}:{port}/mobile-camera/{token}"
    qr_image = qrcode.make(mobile_url)
    buffer = io.BytesIO()
    qr_image.save(buffer, format="PNG")
    return jsonify(token=token, mobile_url=mobile_url, host=host, port=port,
                   qr_code="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                   expires_at=expires_at.isoformat())


@app.route("/mobile-camera/<token>")
def mobile_camera(token):
    item = mobile_session(token)
    if item is None:
        return render_template("mobile_camera.html", token=None, error="expired"), 404
    # Check if a different phone is already actively streaming
    if item["connected_ip"] and item["connected_ip"] != request.remote_addr:
        last_seen = datetime.fromisoformat(item["last_seen"]) if item["last_seen"] else None
        if last_seen and datetime.now(timezone.utc) - last_seen < timedelta(seconds=8):
            return render_template("mobile_camera.html", token=None, error="in_use"), 409
    # Don't mark connected until first frame arrives
    return render_template("mobile_camera.html", token=token)


@app.post("/api/mobile-camera/<token>/frame")
def mobile_camera_frame(token):
    item = mobile_session(token)
    if item is None:
        return jsonify(error="This camera link has expired."), 404
    # Enforce single-phone exclusivity
    caller_ip = request.remote_addr
    if item["connected_ip"] and item["connected_ip"] != caller_ip:
        last_seen = datetime.fromisoformat(item["last_seen"]) if item["last_seen"] else None
        if last_seen and datetime.now(timezone.utc) - last_seen < timedelta(seconds=8):
            return jsonify(error="Camera session already in use by another device."), 409
    encoded = (request.get_json(silent=True) or {}).get("image", "")
    if not isinstance(encoded, str) or "," not in encoded:
        return jsonify(error="A JPEG camera frame is required."), 400
    try:
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if len(raw) > MAX_FRAME_BYTES:
            raise ValueError("Frame too large")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        data, annotated = item["engine"].process_with_overlay(frame)
        ok, output = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            item["image"] = "data:image/jpeg;base64," + base64.b64encode(output.tobytes()).decode("ascii")
        item["connected"] = True
        item["connected_ip"] = caller_ip
        item["last_seen"] = utcnow()
        return jsonify(data)
    except (ValueError, TypeError):
        return jsonify(error="Invalid camera frame."), 400


@app.get("/api/mobile-camera/<token>/status")
@login_required
def mobile_camera_status(token):
    item = owned_mobile_session(token)
    if item is None:
        return jsonify(error="Mobile camera session not found."), 404
    last_seen = datetime.fromisoformat(item["last_seen"]) if item["last_seen"] else None
    connected = bool(last_seen and datetime.now(timezone.utc) - last_seen < timedelta(seconds=5))
    data = item["engine"].snapshot(item["image"] is not None)
    data.update({"connected": connected, "image": item["image"],
                 "last_seen": item["last_seen"]})
    return jsonify(data)


@app.post("/api/mobile-camera/<token>/disconnect")
@login_required
def disconnect_mobile_camera(token):
    """Disconnect the current phone so a new one can connect."""
    item = owned_mobile_session(token)
    if item is None:
        return jsonify(error="Mobile camera session not found."), 404
    item["connected"] = False
    item["connected_ip"] = None
    item["last_seen"] = None
    item["image"] = None
    return jsonify(status="disconnected", message="Phone disconnected. A new phone can now connect.")


@app.get("/api/mobile-camera/<token>/info")
@login_required
def mobile_camera_info(token):
    """Return network diagnostics for the PC dashboard."""
    item = owned_mobile_session(token)
    if item is None:
        return jsonify(error="Mobile camera session not found."), 404
    host = local_network_ip()
    port = int(os.environ.get("FLASK_PORT", 5000))
    scheme = mobile_camera_scheme()
    last_seen = datetime.fromisoformat(item["last_seen"]) if item["last_seen"] else None
    connected = bool(last_seen and datetime.now(timezone.utc) - last_seen < timedelta(seconds=5))
    return jsonify(
        host=host, port=port, scheme=scheme, token=token,
        connection_url=f"{scheme}://{host}:{port}/mobile-camera/{token}",
        server_running=True, connected=connected,
        connected_ip=item["connected_ip"],
        last_seen=item["last_seen"],
        expires_at=item["expires_at"].isoformat(),
        https_enabled=scheme == "https",
    )


@app.post("/api/mobile-camera/<token>/end")
@login_required
def end_mobile_camera(token):
    item = owned_mobile_session(token)
    if item is None:
        return jsonify(error="Mobile camera session not found."), 404
    data = item["engine"].snapshot(item["image"] is not None)
    if not item["saved"]:
        counts, total = data["counters"], sum(data["counters"].values())
        get_db().execute("""INSERT INTO workout_sessions
            (user_id, started_at, ended_at, pushups, pullups, situps, squats, walking, total_reps, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'mobile')""",
            (session["user_id"], item["started_at"], utcnow(), counts["pushup"], counts["pullup"],
             counts["situp"], counts["squat"], counts["walk"], total))
        get_db().commit()
        item["saved"] = True
    item["engine"].close()
    del mobile_camera_sessions[token]
    return jsonify(data)


@app.route("/upload")
@login_required
def upload_video():
    return render_template("upload.html")


@app.post("/api/video/upload")
@login_required
def upload_video_api():
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify(error="Choose a workout video first."), 400
    if not allowed_video(file.filename):
        return jsonify(error="Unsupported format. Upload MP4, AVI, MOV, or WebM."), 400
    original_name = secure_filename(file.filename)
    if not original_name:
        return jsonify(error="Invalid video filename."), 400
    analysis_id = str(uuid.uuid4())
    stored_name = f"{analysis_id}_{original_name}"
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / stored_name
    try:
        file.save(file_path)
        analyzer = VideoAnalyzer(file_path)
    except (OSError, ValueError) as exc:
        file_path.unlink(missing_ok=True)
        return jsonify(error=f"Unable to read this video: {exc}"), 400
    video_analyses[analysis_id] = {
        "user_id": session["user_id"], "filename": original_name,
        "stored_name": stored_name, "analyzer": analyzer,
    }
    return jsonify(analysis_id=analysis_id, filename=original_name,
                   frame_count=analyzer.frame_count, duration_seconds=analyzer.duration_seconds,
                   source_fps=analyzer.fps, ai_fps=analyzer.ai_fps)


@app.errorhandler(413)
def upload_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="Video is too large. Maximum upload size is 200 MB."), 413
    return "Upload too large", 413


@app.route("/video-analysis/<analysis_id>")
@login_required
def video_analysis(analysis_id):
    analysis = active_analysis(analysis_id)
    if analysis is None:
        flash("That video analysis is unavailable.", "error")
        return redirect(url_for("upload_video"))
    return render_template("video_analysis.html", analysis_id=analysis_id, filename=analysis["filename"])


@app.route("/video-analysis/<analysis_id>/source")
@login_required
def uploaded_video_source(analysis_id):
    analysis = active_analysis(analysis_id)
    if analysis is None:
        abort(404)
    return send_from_directory(str(UPLOAD_DIR), analysis["stored_name"], as_attachment=False)


@app.post("/api/video/<analysis_id>/frame")
@login_required
def analyze_browser_video_frame(analysis_id):
    """Accept one browser-sampled native-video frame for independent AI analysis."""
    analysis = active_analysis(analysis_id)
    if analysis is None:
        return jsonify(error="Video analysis not found."), 404
    analyzer = analysis["analyzer"]
    if analyzer.complete:
        return jsonify(analyzer.payload())
    payload = request.get_json(silent=True) or {}
    encoded, timestamp = payload.get("image", ""), payload.get("timestamp", 0)
    if not isinstance(encoded, str) or "," not in encoded:
        return jsonify(error="A JPEG video frame is required."), 400
    try:
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if len(raw) > MAX_FRAME_BYTES:
            raise ValueError("Frame too large")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        return jsonify(analyzer.process_browser_frame(frame, timestamp))
    except (ValueError, TypeError):
        return jsonify(error="Invalid video frame."), 400


@app.post("/api/video/<analysis_id>/complete")
@login_required
def complete_browser_video_analysis(analysis_id):
    analysis = active_analysis(analysis_id)
    if analysis is None:
        return jsonify(error="Video analysis not found."), 404
    analyzer = analysis["analyzer"]
    analyzer.frame_index = analyzer.frame_count
    analyzer.complete = True
    return jsonify(analyzer.payload())


@app.post("/api/video/<analysis_id>/restart")
@login_required
def restart_video_analysis(analysis_id):
    analysis = active_analysis(analysis_id)
    if analysis is None:
        return jsonify(error="Video analysis not found."), 404
    analysis["analyzer"].close()
    try:
        analysis["analyzer"] = VideoAnalyzer(UPLOAD_DIR / analysis["stored_name"])
        return jsonify(analysis["analyzer"].payload())
    except ValueError as exc:
        return jsonify(error=f"Unable to restart analysis: {exc}"), 400


@app.post("/api/video/<analysis_id>/save")
@login_required
def save_video_analysis(analysis_id):
    analysis = active_analysis(analysis_id)
    if analysis is None:
        return jsonify(error="Video analysis not found."), 404
    analyzer = analysis["analyzer"]
    if not analyzer.complete:
        return jsonify(error="Finish analysis before saving."), 409
    if analyzer.saved:
        return jsonify(error="This analysis was already saved."), 409
    counts = analyzer.engine.snapshot()["counters"]
    total = sum(counts.values())
    get_db().execute("""INSERT INTO workout_sessions
        (user_id, started_at, ended_at, pushups, pullups, situps, squats, walking, total_reps,
         video_filename, duration_seconds, source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'video')""",
        (session["user_id"], utcnow(), utcnow(), counts["pushup"], counts["pullup"],
         counts["situp"], counts["squat"], counts["walk"], total, analysis["filename"], analyzer.duration_seconds))
    get_db().commit()
    analyzer.saved = True
    return jsonify(message="Workout saved to history.", total_reps=total)


@app.post("/api/workout/start")
@login_required
def start_workout():
    end_engine(save=False)
    workout_id = str(uuid.uuid4())
    engines[workout_id] = WorkoutEngine()
    session["workout_id"] = workout_id
    session["workout_started_at"] = utcnow()
    return jsonify(engines[workout_id].snapshot())


@app.post("/api/workout/frame")
@login_required
def process_frame():
    engine = active_engine()
    if engine is None:
        return jsonify(error="Start the camera before sending frames."), 409
    encoded = (request.get_json(silent=True) or {}).get("image", "")
    if not isinstance(encoded, str) or "," not in encoded:
        return jsonify(error="A JPEG camera frame is required."), 400
    try:
        raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
        if len(raw) > MAX_FRAME_BYTES:
            raise ValueError("Frame is too large")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        return jsonify(engine.process(frame))
    except (ValueError, TypeError):
        return jsonify(error="Invalid camera frame."), 400


@app.post("/api/workout/control")
@login_required
def workout_control():
    engine = active_engine()
    if engine is None:
        return jsonify(error="No active workout."), 409
    payload = request.get_json(silent=True) or {}
    action, exercise = payload.get("action"), payload.get("exercise")
    if action not in {"manual", "auto", "reset", "clear"}:
        return jsonify(error="Invalid workout command."), 400
    if action == "manual" and exercise not in ALL_EXERCISES:
        return jsonify(error="Invalid exercise."), 400
    return jsonify(engine.control(action, exercise))


def end_engine(save=True):
    workout_id = session.pop("workout_id", None)
    started_at = session.pop("workout_started_at", None)
    engine = engines.pop(workout_id, None) if workout_id else None
    if engine is None:
        return None
    data = engine.snapshot()
    engine.close()
    if save and "user_id" in session:
        counts = data["counters"]
        total = sum(counts.values())
        get_db().execute("""INSERT INTO workout_sessions
            (user_id, started_at, ended_at, pushups, pullups, situps, squats, walking, total_reps, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'live')""",
            (session["user_id"], started_at or utcnow(), utcnow(), counts["pushup"], counts["pullup"],
             counts["situp"], counts["squat"], counts["walk"], total))
        get_db().commit()
    return data


@app.post("/api/workout/end")
@login_required
def end_workout():
    data = end_engine(save=True)
    if data is None:
        return jsonify(error="No active workout."), 409
    return jsonify(data)


@app.route("/history")
@login_required
def history():
    workouts = get_db().execute("SELECT * FROM workout_sessions WHERE user_id = ? ORDER BY ended_at DESC",
                                (session["user_id"],)).fetchall()
    return render_template("history.html", workouts=workouts)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("FLASK_PORT", 5000))
    ssl_context = "adhoc" if os.environ.get("FLASK_USE_HTTPS") == "1" else None
    app.run(host="0.0.0.0", port=port, debug=True, ssl_context=ssl_context)
