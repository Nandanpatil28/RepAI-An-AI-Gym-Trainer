"""Flask web interface for the existing AI Fitness Trainer pose pipeline."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import secrets
import socket
import sqlite3
import smtplib
import tempfile
import uuid
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np
import qrcode
from flask import Flask, abort, flash, g, jsonify, redirect, render_template, request, send_file, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from apscheduler.schedulers.background import BackgroundScheduler

from exercise_engine import ALL_EXERCISES, WorkoutEngine
from food_model import FoodModelUnavailable, classify_food
from nutrition_lookup import NutritionLookupError, calculate_totals, lookup_nutrition
from video_analyzer import VideoAnalyzer

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "fitness.db"
MAX_FRAME_BYTES = 1_500_000
engines = {}
video_analyses = {}
mobile_camera_sessions = {}
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "webm"}
ALLOWED_FOOD_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
FOOD_IMAGE_MAX_BYTES = 8 * 1024 * 1024
FOOD_IMAGE_MAX_PIXELS = 12_000_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

EXERCISE_LIBRARY = [
    {
        "key": "push-up", "name": "Push-Up", "category": "Chest", "muscles": "Chest, shoulders, triceps, core",
        "difficulty": "Beginner", "equipment": "Bodyweight", "supported": True,
        "description": "A horizontal pressing movement that builds upper-body pushing strength.",
        "instructions": ["Start in a high plank with hands under shoulders.", "Lower your chest while keeping your body straight.", "Press through the palms until elbows are extended."],
        "demo": "Exercise_videos/push-up.mp4",
    },
    {
        "key": "pull-up", "name": "Pull-Up", "category": "Back", "muscles": "Lats, upper back, biceps, core",
        "difficulty": "Advanced", "equipment": "Pull-up bar", "supported": True,
        "description": "A vertical pull that develops back and arm strength.",
        "instructions": ["Hang from a bar with hands slightly wider than shoulders.", "Pull until your chin clears the bar.", "Lower with control to a full hang."],
        "demo": "Exercise_videos/pull-up.mp4",
    },
    {
        "key": "sit-up", "name": "Sit-Up", "category": "Core", "muscles": "Abdominals, hip flexors",
        "difficulty": "Beginner", "equipment": "Mat optional", "supported": True,
        "description": "A trunk-flexion exercise for abdominal endurance.",
        "instructions": ["Lie on your back with knees bent.", "Brace your core and lift your torso toward your knees.", "Lower under control without dropping back."],
        "demo": "Exercise_videos/sit-up.mp4",
    },
    {
        "key": "squat", "name": "Squat", "category": "Legs", "muscles": "Quads, glutes, hamstrings, core",
        "difficulty": "Beginner", "equipment": "Bodyweight", "supported": True,
        "description": "A lower-body movement for leg strength and mobility.",
        "instructions": ["Stand with feet about shoulder-width apart.", "Sit hips back and bend knees while keeping chest lifted.", "Drive through your feet to stand tall."],
        "demo": "Exercise_videos/squat.mp4",
    },
    {
        "key": "walk", "name": "Walking", "category": "Cardio", "muscles": "Calves, quads, hamstrings, glutes",
        "difficulty": "Beginner", "equipment": "None", "supported": True,
        "description": "Low-impact cardio for aerobic capacity and daily movement.",
        "instructions": ["Stand tall with relaxed shoulders.", "Walk at a sustainable pace.", "Keep strides smooth and breathe steadily."],
        "demo": "Exercise_videos/walk.mp4",
    },
    {
        "key": "plank", "name": "Plank", "category": "Core", "muscles": "Core, shoulders, glutes",
        "difficulty": "Intermediate", "equipment": "Mat optional", "supported": False,
        "description": "An isometric core exercise for trunk stability.",
        "instructions": ["Place elbows under shoulders.", "Extend legs and brace your body in a straight line.", "Hold without sagging hips."],
        "demo": "",
    },
    {
        "key": "lunge", "name": "Lunge", "category": "Legs", "muscles": "Quads, glutes, hamstrings",
        "difficulty": "Intermediate", "equipment": "Bodyweight or dumbbells", "supported": False,
        "description": "A single-leg pattern for lower-body strength and balance.",
        "instructions": ["Step forward into a split stance.", "Lower until both knees bend comfortably.", "Push back to standing and alternate sides."],
        "demo": "",
    },
]

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

MEAL_LIBRARY = [
    {
        "name": "Oats with banana and peanut butter",
        "slot": "breakfast",
        "diet": {"vegetarian", "vegan", "non-vegetarian"},
        "ingredients": ["rolled oats", "banana", "peanut butter", "soy milk", "chia seeds"],
        "instructions": "Cook oats in soy milk, top with sliced banana, chia seeds, and peanut butter.",
        "portion": "1 bowl: 60 g oats, 1 banana, 1 tbsp peanut butter",
        "kcal": 520, "protein": 19, "carbs": 74, "fat": 17,
        "substitutions": "Use sunflower seed butter instead of peanut butter if nuts are avoided.",
    },
    {
        "name": "Vegetable poha with curd",
        "slot": "breakfast",
        "diet": {"vegetarian", "non-vegetarian"},
        "ingredients": ["flattened rice", "peas", "carrot", "onion", "curd", "mustard seeds"],
        "instructions": "Cook rinsed poha with vegetables and spices; serve with curd.",
        "portion": "1.5 cups poha with 150 g curd",
        "kcal": 430, "protein": 16, "carbs": 70, "fat": 9,
        "substitutions": "Use tofu curd for a vegan version.",
    },
    {
        "name": "Egg veggie toast plate",
        "slot": "breakfast",
        "diet": {"eggetarian", "non-vegetarian"},
        "ingredients": ["eggs", "whole wheat toast", "spinach", "tomato", "olive oil"],
        "instructions": "Scramble eggs with spinach and tomato; serve with toasted bread.",
        "portion": "2 eggs, 2 toast slices, 1 cup vegetables",
        "kcal": 460, "protein": 25, "carbs": 42, "fat": 20,
        "substitutions": "Swap eggs with paneer or tofu scramble.",
    },
    {
        "name": "Paneer quinoa bowl",
        "slot": "lunch",
        "diet": {"vegetarian", "non-vegetarian"},
        "ingredients": ["paneer", "quinoa", "capsicum", "cucumber", "yogurt dressing"],
        "instructions": "Grill paneer, cook quinoa, and combine with vegetables and yogurt dressing.",
        "portion": "120 g paneer, 1 cup cooked quinoa, 2 cups vegetables",
        "kcal": 650, "protein": 38, "carbs": 62, "fat": 27,
        "substitutions": "Use tofu instead of paneer for vegan preference.",
    },
    {
        "name": "Dal, brown rice, and salad",
        "slot": "lunch",
        "diet": {"vegetarian", "vegan", "non-vegetarian"},
        "ingredients": ["lentils", "brown rice", "cucumber", "tomato", "lemon"],
        "instructions": "Simmer dal with spices; serve with brown rice and a lemon salad.",
        "portion": "1.5 cups dal, 1 cup cooked rice, 1 bowl salad",
        "kcal": 620, "protein": 28, "carbs": 103, "fat": 9,
        "substitutions": "Use millet or roti instead of rice.",
    },
    {
        "name": "Chicken rice power bowl",
        "slot": "lunch",
        "diet": {"non-vegetarian"},
        "ingredients": ["chicken breast", "rice", "beans", "mixed vegetables", "curd"],
        "instructions": "Grill chicken, cook rice and beans, and serve with vegetables and curd.",
        "portion": "150 g chicken, 1 cup rice, 0.5 cup beans, 2 cups vegetables",
        "kcal": 680, "protein": 52, "carbs": 78, "fat": 14,
        "substitutions": "Swap chicken with tofu, paneer, or extra beans.",
    },
    {
        "name": "Tofu millet bowl",
        "slot": "lunch",
        "diet": {"vegan", "vegetarian", "non-vegetarian"},
        "ingredients": ["tofu", "millet", "broccoli", "carrot", "sesame"],
        "instructions": "Pan-sear tofu, cook millet, and toss with vegetables and sesame.",
        "portion": "150 g tofu, 1 cup cooked millet, 2 cups vegetables",
        "kcal": 590, "protein": 31, "carbs": 70, "fat": 20,
        "substitutions": "Use chickpeas or paneer if preferred.",
    },
    {
        "name": "Rajma roti plate",
        "slot": "dinner",
        "diet": {"vegetarian", "vegan", "non-vegetarian"},
        "ingredients": ["kidney beans", "whole wheat roti", "onion", "tomato", "spinach"],
        "instructions": "Cook rajma in tomato gravy and serve with rotis and sauteed spinach.",
        "portion": "1.5 cups rajma, 2 rotis, 1 cup spinach",
        "kcal": 610, "protein": 26, "carbs": 98, "fat": 11,
        "substitutions": "Use chickpeas or black beans instead of rajma.",
    },
    {
        "name": "Fish curry with rice",
        "slot": "dinner",
        "diet": {"pescatarian", "non-vegetarian"},
        "ingredients": ["fish", "rice", "coconut milk", "tomato", "green beans"],
        "instructions": "Simmer fish in light curry and serve with rice and steamed beans.",
        "portion": "140 g fish, 1 cup rice, 1 cup beans",
        "kcal": 580, "protein": 38, "carbs": 62, "fat": 18,
        "substitutions": "Use tofu or lentil curry if fish is avoided.",
    },
    {
        "name": "Paneer tikka roti wrap",
        "slot": "dinner",
        "diet": {"vegetarian", "non-vegetarian"},
        "ingredients": ["paneer", "whole wheat roti", "capsicum", "onion", "mint chutney"],
        "instructions": "Grill paneer and vegetables, wrap in rotis with mint chutney.",
        "portion": "120 g paneer with 2 rotis and vegetables",
        "kcal": 620, "protein": 34, "carbs": 58, "fat": 27,
        "substitutions": "Use tofu tikka for a vegan wrap.",
    },
    {
        "name": "Fruit and yogurt bowl",
        "slot": "snack",
        "diet": {"vegetarian", "non-vegetarian"},
        "ingredients": ["curd", "apple", "berries", "pumpkin seeds"],
        "instructions": "Mix curd with chopped fruit and top with pumpkin seeds.",
        "portion": "200 g curd, 1 cup fruit, 1 tbsp seeds",
        "kcal": 260, "protein": 13, "carbs": 36, "fat": 8,
        "substitutions": "Use coconut yogurt or soy yogurt for dairy-free preference.",
    },
    {
        "name": "Roasted chana and fruit",
        "slot": "snack",
        "diet": {"vegetarian", "vegan", "non-vegetarian"},
        "ingredients": ["roasted chana", "orange", "lemon water"],
        "instructions": "Pair roasted chana with fruit and water.",
        "portion": "50 g roasted chana and 1 fruit",
        "kcal": 250, "protein": 12, "carbs": 40, "fat": 4,
        "substitutions": "Use makhana or sprouts if chana is avoided.",
    },
    {
        "name": "Protein smoothie",
        "slot": "snack",
        "diet": {"vegetarian", "vegan", "non-vegetarian"},
        "ingredients": ["plant protein", "banana", "oats", "soy milk"],
        "instructions": "Blend all ingredients until smooth.",
        "portion": "1 scoop protein, 1 banana, 25 g oats, 250 ml soy milk",
        "kcal": 360, "protein": 30, "carbs": 48, "fat": 6,
        "substitutions": "Use curd or dairy milk if preferred and tolerated.",
    },
]

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", os.urandom(32)),
    DATABASE=str(DATABASE_PATH),
    MAX_CONTENT_LENGTH=200 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Configure logging for nutrition lookup diagnostics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("nutrition_lookup").setLevel(logging.DEBUG)


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
            created_at TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1
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
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            height_cm REAL,
            weight_kg REAL,
            fitness_goal TEXT,
            activity_level TEXT,
            dietary_preference TEXT,
            allergies TEXT,
            liked_foods TEXT,
            disliked_foods TEXT,
            meals_per_day INTEGER,
            preferred_meal_times TEXT,
            budget TEXT,
            regional_preference TEXT,
            medical_notes TEXT,
            measurement_units TEXT NOT NULL DEFAULT 'metric',
            timezone TEXT NOT NULL DEFAULT 'Asia/Calcutta',
            notify_in_app INTEGER NOT NULL DEFAULT 1,
            notify_email INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS meal_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            week_start TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            target_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS meal_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            meal_plan_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            day_index INTEGER NOT NULL,
            meal_index INTEGER NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            actual_intake TEXT,
            actual_calories REAL,
            actual_protein REAL,
            actual_carbs REAL,
            actual_fat REAL,
            logged_at TEXT NOT NULL,
            UNIQUE(user_id, meal_plan_id, day_index, meal_index),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (meal_plan_id) REFERENCES meal_plans(id)
        );
        CREATE TABLE IF NOT EXISTS food_scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dish_name TEXT NOT NULL,
            selected_label TEXT,
            serving_grams REAL NOT NULL,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            source TEXT,
            fdc_id INTEGER,
            confidence REAL,
            logged_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS workout_routines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            items_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS workout_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            routine_id INTEGER NOT NULL,
            scheduled_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            completed_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (routine_id) REFERENCES workout_routines(id)
        );
        CREATE TABLE IF NOT EXISTS fitness_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal_type TEXT NOT NULL,
            title TEXT NOT NULL,
            target_value REAL NOT NULL,
            exercise_key TEXT,
            deadline TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS weight_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            weight_kg REAL NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, entry_date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            earned_at TEXT NOT NULL,
            UNIQUE(user_id, code),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS workout_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            reminder_date TEXT NOT NULL,
            reminder_time TEXT NOT NULL,
            timezone TEXT NOT NULL,
            note TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            last_notified_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS reminder_email_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder_id INTEGER NOT NULL UNIQUE,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            attempted_at TEXT,
            sent_at TEXT,
            failure_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (reminder_id) REFERENCES workout_reminders(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_reminder_email_deliveries_user ON reminder_email_deliveries(user_id);
        CREATE TABLE IF NOT EXISTS custom_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            muscles TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            equipment TEXT,
            description TEXT NOT NULL,
            instructions TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );
    """)
    user_columns = {row[1] for row in db.execute("PRAGMA table_info(users)")}
    for name, definition in (
        ("is_admin", "INTEGER NOT NULL DEFAULT 0"),
        ("active", "INTEGER NOT NULL DEFAULT 1"),
    ):
        if name not in user_columns:
            db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")
    columns = {row[1] for row in db.execute("PRAGMA table_info(workout_sessions)")}
    for name, definition in (("video_filename", "TEXT"), ("duration_seconds", "REAL"),
                             ("source_type", "TEXT NOT NULL DEFAULT 'live'")):
        if name not in columns:
            db.execute(f"ALTER TABLE workout_sessions ADD COLUMN {name} {definition}")
    profile_columns = {row[1] for row in db.execute("PRAGMA table_info(user_profiles)")}
    for name, definition in (
        ("medical_notes", "TEXT"),
        ("budget", "TEXT"),
        ("regional_preference", "TEXT"),
        ("preferred_meal_times", "TEXT"),
        ("measurement_units", "TEXT NOT NULL DEFAULT 'metric'"),
        ("timezone", "TEXT NOT NULL DEFAULT 'Asia/Calcutta'"),
        ("notify_in_app", "INTEGER NOT NULL DEFAULT 1"),
        ("notify_email", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if name not in profile_columns:
            db.execute(f"ALTER TABLE user_profiles ADD COLUMN {name} {definition}")
    admin_emails = [item.strip().lower() for item in os.environ.get("ADMIN_EMAILS", "").split(",") if item.strip()]
    if admin_emails:
        placeholders = ",".join("?" for _ in admin_emails)
        db.execute(f"UPDATE users SET is_admin = 1 WHERE lower(email) IN ({placeholders})", admin_emails)
    db.commit()
    db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    if "user_id" not in session:
        return None
    return get_db().execute(
        "SELECT id, name, email, created_at, is_admin, active FROM users WHERE id = ?",
        (session["user_id"],),
    ).fetchone()


def valid_email(value):
    """Accept normal provider addresses while rejecting malformed/header-injection input."""
    return bool(value and len(value) <= 254 and "\n" not in value and "\r" not in value and EMAIL_PATTERN.fullmatch(value))


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            return redirect(url_for("login"))
        if not user["is_admin"]:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_navigation_state():
    user = current_user()
    reminders = []
    if user is not None:
        reminders = due_reminders(limit=3)
    return {"current_user": user, "due_reminders": reminders}


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


def allowed_food_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_FOOD_IMAGE_EXTENSIONS


def image_size(file_storage):
    stream = file_storage.stream
    position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(position)
    return size


def validate_food_image(path):
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ValueError("Image validation dependency Pillow is not installed.") from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width * height > FOOD_IMAGE_MAX_PIXELS:
                raise ValueError("Food photo is too large in resolution. Upload an image up to 12 megapixels.")
            image.verify()
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Unsupported image format. Upload JPG, PNG, or WebP.")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Unable to read this image. Upload a valid food photo.") from exc


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


def today_iso():
    return date.today().isoformat()


def as_int(value, default=None):
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def current_profile():
    return get_db().execute("SELECT * FROM user_profiles WHERE user_id = ?", (session["user_id"],)).fetchone()


def default_timezone():
    profile = current_profile() if "user_id" in session else None
    return profile["timezone"] if profile and profile["timezone"] else "Asia/Calcutta"


def valid_timezone(value):
    try:
        ZoneInfo(value)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


def user_local_now(tz_name=None):
    try:
        tz = ZoneInfo(tz_name or default_timezone())
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


def custom_exercises():
    try:
        rows = get_db().execute("SELECT * FROM custom_exercises WHERE active = 1 ORDER BY name").fetchall()
    except sqlite3.OperationalError:
        return []
    items = []
    for row in rows:
        key = f"custom-{row['id']}"
        instructions = [item.strip() for item in (row["instructions"] or "").splitlines() if item.strip()]
        items.append({
            "key": key,
            "name": row["name"],
            "category": row["category"],
            "muscles": row["muscles"],
            "difficulty": row["difficulty"],
            "equipment": row["equipment"] or "Optional",
            "supported": False,
            "description": row["description"],
            "instructions": instructions or ["Follow a controlled, pain-free range of motion."],
            "demo": "",
            "custom_id": row["id"],
        })
    return items


def exercise_library_items(include_inactive=False):
    items = [dict(item) for item in EXERCISE_LIBRARY]
    if include_inactive:
        rows = get_db().execute("SELECT * FROM custom_exercises ORDER BY active DESC, name").fetchall()
        for row in rows:
            item = {
                "key": f"custom-{row['id']}",
                "name": row["name"],
                "category": row["category"],
                "muscles": row["muscles"],
                "difficulty": row["difficulty"],
                "equipment": row["equipment"] or "Optional",
                "supported": False,
                "description": row["description"],
                "instructions": [line.strip() for line in (row["instructions"] or "").splitlines() if line.strip()],
                "demo": "",
                "custom_id": row["id"],
                "active": row["active"],
            }
            items.append(item)
        return items
    return items + custom_exercises()


def parse_terms(text):
    return {item.strip().lower() for item in (text or "").replace(";", ",").split(",") if item.strip()}


def expanded_food_terms(text):
    terms = parse_terms(text)
    aliases = {
        "lactose": {"milk", "curd", "yogurt", "paneer", "cheese", "dairy"},
        "dairy": {"milk", "curd", "yogurt", "paneer", "cheese", "lactose"},
        "nuts": {"peanut", "peanuts", "peanut butter", "almond", "cashew", "walnut", "nut butter"},
        "nut": {"peanut", "peanuts", "peanut butter", "almond", "cashew", "walnut", "nut butter"},
        "peanut": {"peanuts", "peanut butter"},
        "peanuts": {"peanut", "peanut butter"},
        "egg": {"eggs"},
        "eggs": {"egg"},
        "gluten": {"wheat", "roti", "toast", "bread"},
        "shellfish": {"prawn", "shrimp", "crab"},
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(aliases.get(term, set()))
    return expanded


def profile_ready(profile):
    required = ("age", "height_cm", "weight_kg", "fitness_goal", "activity_level", "dietary_preference")
    return bool(profile and all(profile[key] not in (None, "") for key in required))


def profile_form_values(form):
    units = form.get("measurement_units", "metric").strip() or "metric"
    tz_name = form.get("timezone", default_timezone()).strip() or "UTC"
    return {
        "user_id": session["user_id"],
        "age": as_int(form.get("age")),
        "height_cm": as_float(form.get("height_cm")),
        "weight_kg": as_float(form.get("weight_kg")),
        "fitness_goal": form.get("fitness_goal", "").strip(),
        "activity_level": form.get("activity_level", "").strip(),
        "dietary_preference": form.get("dietary_preference", "").strip(),
        "allergies": form.get("allergies", "").strip(),
        "liked_foods": form.get("liked_foods", "").strip(),
        "disliked_foods": form.get("disliked_foods", "").strip(),
        "meals_per_day": as_int(form.get("meals_per_day"), 3),
        "preferred_meal_times": form.get("preferred_meal_times", "").strip(),
        "budget": form.get("budget", "").strip(),
        "regional_preference": form.get("regional_preference", "").strip(),
        "medical_notes": form.get("medical_notes", "").strip(),
        "measurement_units": units if units in {"metric", "imperial"} else "metric",
        "timezone": tz_name if valid_timezone(tz_name) else "UTC",
        "notify_in_app": 1 if form.get("notify_in_app") == "on" else 0,
        "notify_email": 1 if form.get("notify_email") == "on" else 0,
        "updated_at": utcnow(),
    }


def validate_profile_values(values):
    errors = []
    if values["age"] is not None and not 8 <= values["age"] <= 100:
        errors.append("Age must be between 8 and 100.")
    if values["height_cm"] is not None and not 80 <= values["height_cm"] <= 250:
        errors.append("Height must be between 80 and 250 cm.")
    if values["weight_kg"] is not None and not 20 <= values["weight_kg"] <= 300:
        errors.append("Weight must be between 20 and 300 kg.")
    if values["meals_per_day"] is not None and not 3 <= values["meals_per_day"] <= 6:
        errors.append("Meals per day must be between 3 and 6.")
    if values["fitness_goal"] and values["fitness_goal"] not in {"weight_loss", "muscle_gain", "maintenance", "general_fitness"}:
        errors.append("Choose a valid fitness goal.")
    if values["activity_level"] and values["activity_level"] not in ACTIVITY_MULTIPLIERS:
        errors.append("Choose a valid activity level.")
    if values["timezone"] and not valid_timezone(values["timezone"]):
        errors.append("Choose a valid timezone.")
    return errors


def save_profile_values(values):
    get_db().execute("""
        INSERT INTO user_profiles
        (user_id, age, height_cm, weight_kg, fitness_goal, activity_level, dietary_preference,
         allergies, liked_foods, disliked_foods, meals_per_day, preferred_meal_times, budget,
         regional_preference, medical_notes, measurement_units, timezone, notify_in_app,
         notify_email, updated_at)
        VALUES (:user_id, :age, :height_cm, :weight_kg, :fitness_goal, :activity_level,
                :dietary_preference, :allergies, :liked_foods, :disliked_foods, :meals_per_day,
                :preferred_meal_times, :budget, :regional_preference, :medical_notes,
                :measurement_units, :timezone, :notify_in_app, :notify_email, :updated_at)
        ON CONFLICT(user_id) DO UPDATE SET
            age=excluded.age, height_cm=excluded.height_cm, weight_kg=excluded.weight_kg,
            fitness_goal=excluded.fitness_goal, activity_level=excluded.activity_level,
            dietary_preference=excluded.dietary_preference, allergies=excluded.allergies,
            liked_foods=excluded.liked_foods, disliked_foods=excluded.disliked_foods,
            meals_per_day=excluded.meals_per_day, preferred_meal_times=excluded.preferred_meal_times,
            budget=excluded.budget, regional_preference=excluded.regional_preference,
            medical_notes=excluded.medical_notes, measurement_units=excluded.measurement_units,
            timezone=excluded.timezone, notify_in_app=excluded.notify_in_app,
            notify_email=excluded.notify_email, updated_at=excluded.updated_at
    """, values)


def calculate_targets(profile):
    age = int(profile["age"])
    height = float(profile["height_cm"])
    weight = float(profile["weight_kg"])
    activity = ACTIVITY_MULTIPLIERS.get(profile["activity_level"], 1.375)
    maintenance = (10 * weight + 6.25 * height - 5 * age) * activity
    goal = profile["fitness_goal"]
    safety_note = ""
    if age < 18:
        calories = maintenance
        safety_note = "Minor profile: no calorie deficit is applied; use a guardian and qualified professional for restrictive goals."
    elif goal == "weight_loss":
        calories = max(1600, maintenance * 0.85)
    elif goal == "muscle_gain":
        calories = maintenance * 1.1
    else:
        calories = maintenance
    protein_factor = 1.8 if goal in {"muscle_gain", "weight_loss"} else 1.4
    protein = weight * protein_factor
    fat = max(45, calories * 0.25 / 9)
    carbs = max(0, (calories - protein * 4 - fat * 9) / 4)
    return {
        "calories": round(calories),
        "protein": round(protein),
        "carbs": round(carbs),
        "fat": round(fat),
        "maintenance_calories": round(maintenance),
        "method": "Estimated with Mifflin-St Jeor-style BMR without sex adjustment, activity multiplier, and conservative goal adjustment.",
        "safety_note": safety_note,
    }


def compatible_meals(profile, slot):
    preference = (profile["dietary_preference"] or "non-vegetarian").lower()
    allergies = expanded_food_terms(profile["allergies"])
    avoided = expanded_food_terms(profile["disliked_foods"])
    regional = parse_terms(profile["regional_preference"])
    liked = parse_terms(profile["liked_foods"])
    budget = (profile["budget"] or "").lower()
    blocked = allergies | avoided
    meals = []
    for meal in MEAL_LIBRARY:
        allowed_diets = set(meal["diet"])
        if preference == "other":
            allowed_diets.update({"other"})
        if meal["slot"] != slot or (preference != "other" and preference not in allowed_diets):
            continue
        meal_terms = {item.lower() for item in meal["ingredients"]} | {meal["name"].lower()}
        meal_terms.add(meal["substitutions"].lower())
        if any(term and any(term in candidate for candidate in meal_terms) for term in blocked):
            continue
        score = 0
        if any(term and any(term in candidate for candidate in meal_terms) for term in liked):
            score += 2
        if any(term and any(term in candidate for candidate in meal_terms) for term in regional):
            score += 1
        if any(term in budget for term in ("low", "budget", "cheap")):
            if any(item in " ".join(meal_terms) for item in ("dal", "rice", "chana", "rajma", "lentil", "beans", "poha")):
                score += 1
        meals.append((score, meal))
    if not meals:
        fallback = {
            "name": f"Custom {slot} bowl",
            "slot": slot,
            "ingredients": ["safe protein source", "whole grain", "seasonal vegetables", "healthy fat"],
            "instructions": "Assemble foods you tolerate from each group; avoid listed allergens and disliked foods.",
            "portion": "Adjust portions to the listed estimated target.",
            "kcal": 450 if slot == "snack" else 600,
            "protein": 20 if slot == "snack" else 32,
            "carbs": 45 if slot == "snack" else 75,
            "fat": 12 if slot == "snack" else 18,
            "substitutions": "Choose tolerated local foods from the same food group.",
        }
        return [fallback]
    return [meal for _score, meal in sorted(meals, key=lambda item: item[0], reverse=True)]


def default_meal_times(count, preferred):
    entered = [item.strip() for item in (preferred or "").replace(";", ",").split(",") if item.strip()]
    defaults = ["08:00", "13:00", "19:30", "16:30", "21:00", "11:00"]
    merged = entered + [item for item in defaults if item not in entered]
    return merged[:count]


def scale_meal(meal, factor):
    scaled = dict(meal)
    scaled["kcal"] = round(meal["kcal"] * factor)
    scaled["protein"] = round(meal["protein"] * factor, 1)
    scaled["carbs"] = round(meal["carbs"] * factor, 1)
    scaled["fat"] = round(meal["fat"] * factor, 1)
    if factor >= 1.15:
        scaled["portion"] = meal["portion"] + " plus an extra half portion of the main carb or protein."
    elif factor <= 0.85:
        scaled["portion"] = "About three-quarter portion of: " + meal["portion"]
    return scaled


def generate_meal_plan(profile):
    targets = calculate_targets(profile)
    meal_count = min(max(as_int(profile["meals_per_day"], 3), 3), 6)
    slots = ["breakfast", "lunch", "dinner"] + ["snack"] * (meal_count - 3)
    times = default_meal_times(meal_count, profile["preferred_meal_times"])
    weights = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10}
    if meal_count > 4:
        weights["snack"] = 0.10 / (meal_count - 3)
    start = date.today()
    days = []
    for day_index in range(7):
        meals = []
        totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        for meal_index, slot in enumerate(slots):
            choices = compatible_meals(profile, slot)
            choice = choices[(day_index + meal_index) % len(choices)]
            desired = targets["calories"] * weights[slot]
            factor = min(1.35, max(0.70, desired / choice["kcal"]))
            scaled = scale_meal(choice, factor)
            item = {
                "slot": slot,
                "time": times[meal_index],
                "name": scaled["name"],
                "ingredients": scaled["ingredients"],
                "portion": scaled["portion"],
                "instructions": scaled["instructions"],
                "substitutions": scaled["substitutions"],
                "nutrition": {
                    "calories": scaled["kcal"], "protein": scaled["protein"],
                    "carbs": scaled["carbs"], "fat": scaled["fat"],
                },
            }
            for key, value in item["nutrition"].items():
                totals[key] += value
            meals.append(item)
        days.append({
            "date": (start + timedelta(days=day_index)).isoformat(),
            "label": (start + timedelta(days=day_index)).strftime("%A"),
            "meals": meals,
            "totals": {key: round(value, 1) for key, value in totals.items()},
        })
    return {"days": days, "generated_at": utcnow()}, targets


def latest_meal_plan():
    return get_db().execute(
        "SELECT * FROM meal_plans WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (session["user_id"],),
    ).fetchone()


def meal_logs_for_plan(plan_id):
    rows = get_db().execute(
        "SELECT * FROM meal_logs WHERE user_id = ? AND meal_plan_id = ?",
        (session["user_id"], plan_id),
    ).fetchall()
    return {(row["day_index"], row["meal_index"]): row for row in rows}


def exercise_by_key(key):
    return next((item for item in exercise_library_items() if item["key"] == key), None)


def parse_date(value, default=None):
    if not value:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError:
        return default


def workout_rows(start_date=None, end_date=None):
    sql = "SELECT * FROM workout_sessions WHERE user_id = ?"
    params = [session["user_id"]]
    if start_date:
        sql += " AND date(ended_at) >= ?"
        params.append(start_date.isoformat())
    if end_date:
        sql += " AND date(ended_at) <= ?"
        params.append(end_date.isoformat())
    sql += " ORDER BY ended_at DESC"
    return get_db().execute(sql, params).fetchall()


def workout_duration(row):
    if row["duration_seconds"] is not None:
        return float(row["duration_seconds"])
    try:
        started = datetime.fromisoformat(row["started_at"])
        ended = datetime.fromisoformat(row["ended_at"])
        return max(0, (ended - started).total_seconds())
    except (TypeError, ValueError):
        return 0


def exercise_counts(row):
    return {
        "push-up": row["pushups"],
        "pull-up": row["pullups"],
        "sit-up": row["situps"],
        "squat": row["squats"],
        "walk": row["walking"],
    }


def workout_date(row):
    return datetime.fromisoformat(row["ended_at"]).date()


def current_streak(rows):
    dates = {workout_date(row) for row in rows}
    today = date.today()
    cursor = today if today in dates else today - timedelta(days=1)
    streak = 0
    while cursor in dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def analytics_summary(start_date=None, end_date=None):
    rows = workout_rows(start_date, end_date)
    total_duration = sum(workout_duration(row) for row in rows)
    per_exercise = {item["key"]: 0 for item in exercise_library_items() if item["supported"]}
    by_day = {}
    for row in rows:
        day = workout_date(row).isoformat()
        by_day.setdefault(day, {"date": day, "workouts": 0, "reps": 0, "duration": 0})
        by_day[day]["workouts"] += 1
        by_day[day]["reps"] += row["total_reps"]
        by_day[day]["duration"] += workout_duration(row)
        for key, value in exercise_counts(row).items():
            per_exercise[key] = per_exercise.get(key, 0) + value
    return {
        "rows": rows,
        "total_workouts": len(rows),
        "total_duration": round(total_duration),
        "total_reps": sum(row["total_reps"] for row in rows),
        "per_exercise": per_exercise,
        "by_day": [by_day[key] for key in sorted(by_day)],
        "streak": current_streak(rows),
    }


def goal_progress(goal):
    created = parse_date(goal["created_at"][:10], date.min)
    deadline = parse_date(goal["deadline"])
    rows = workout_rows(created, deadline)
    value = 0
    if goal["goal_type"] == "frequency":
        value = len(rows)
    elif goal["goal_type"] == "consistency":
        value = current_streak(rows)
    elif goal["goal_type"] == "rep_target":
        key = goal["exercise_key"]
        value = sum(exercise_counts(row).get(key, row["total_reps"] if not key else 0) for row in rows)
    elif goal["goal_type"] == "duration":
        value = round(sum(workout_duration(row) for row in rows) / 60)
    elif goal["goal_type"] == "performance":
        key = goal["exercise_key"]
        value = max([exercise_counts(row).get(key, row["total_reps"]) for row in rows] or [0])
    elif goal["goal_type"] == "weight":
        latest = get_db().execute(
            "SELECT weight_kg FROM weight_entries WHERE user_id = ? ORDER BY entry_date DESC LIMIT 1",
            (session["user_id"],),
        ).fetchone()
        value = latest["weight_kg"] if latest else 0
    pct = min(100, round((float(value) / float(goal["target_value"])) * 100)) if goal["target_value"] else 0
    return {"value": round(value, 1), "percent": pct, "complete": pct >= 100}


def personal_records():
    rows = workout_rows()
    best_exercise = {}
    for item in exercise_library_items():
        if not item["supported"]:
            continue
        key = item["key"]
        best_row = max(rows, key=lambda row: exercise_counts(row).get(key, 0), default=None)
        best_exercise[key] = {
            "name": item["name"],
            "value": exercise_counts(best_row).get(key, 0) if best_row else 0,
            "date": best_row["ended_at"][:10] if best_row else None,
        }
    longest = max(rows, key=workout_duration, default=None)
    highest = max(rows, key=lambda row: row["total_reps"], default=None)
    return {
        "highest_reps": {"value": highest["total_reps"] if highest else 0, "date": highest["ended_at"][:10] if highest else None},
        "longest_duration": {"value": round(workout_duration(longest) / 60, 1) if longest else 0, "date": longest["ended_at"][:10] if longest else None},
        "exercise": best_exercise,
    }


def award_achievement(code, title, description):
    get_db().execute(
        "INSERT OR IGNORE INTO achievements (user_id, code, title, description, earned_at) VALUES (?, ?, ?, ?, ?)",
        (session["user_id"], code, title, description, utcnow()),
    )


def refresh_achievements():
    summary = analytics_summary()
    records = personal_records()
    if summary["total_workouts"] >= 1:
        award_achievement("first_workout", "First Workout", "Completed and saved your first workout.")
    if summary["total_workouts"] >= 10:
        award_achievement("ten_workouts", "10 Workouts", "Completed 10 saved workouts.")
    if summary["streak"] >= 3:
        award_achievement("three_day_streak", "3 Day Streak", "Worked out for three days in a row.")
    if records["highest_reps"]["value"] >= 50:
        award_achievement("fifty_reps", "50 Rep Session", "Reached 50 valid reps in one workout.")
    completed_scheduled = get_db().execute(
        "SELECT COUNT(*) count FROM workout_schedules WHERE user_id = ? AND status = 'completed'",
        (session["user_id"],),
    ).fetchone()["count"]
    if completed_scheduled >= 1:
        award_achievement("scheduled_complete", "Planned Workout Done", "Completed a scheduled workout.")
    active_goals = get_db().execute("SELECT * FROM fitness_goals WHERE user_id = ?", (session["user_id"],)).fetchall()
    if any(goal_progress(goal)["complete"] for goal in active_goals):
        award_achievement("goal_reached", "Goal Reached", "Reached at least one tracked fitness goal.")
    get_db().commit()


def routine_owned(routine_id):
    return get_db().execute(
        "SELECT * FROM workout_routines WHERE id = ? AND user_id = ?", (routine_id, session["user_id"])
    ).fetchone()


def reminder_due(row, now=None):
    if not row["enabled"] or row["last_notified_at"]:
        return False
    now = now or user_local_now(row["timezone"])
    try:
        scheduled = datetime.fromisoformat(f"{row['reminder_date']}T{row['reminder_time']}").replace(
            tzinfo=ZoneInfo(row["timezone"])
        )
    except (ValueError, ZoneInfoNotFoundError):
        return False
    return scheduled <= now


def due_reminders(limit=None):
    profile = current_profile()
    if profile and not profile["notify_in_app"]:
        return []
    rows = get_db().execute(
        "SELECT * FROM workout_reminders WHERE user_id = ? AND enabled = 1 ORDER BY reminder_date, reminder_time",
        (session["user_id"],),
    ).fetchall()
    due = [row for row in rows if reminder_due(row)]
    return due[:limit] if limit else due


def reminder_owned(reminder_id):
    return get_db().execute(
        "SELECT * FROM workout_reminders WHERE id = ? AND user_id = ?", (reminder_id, session["user_id"])
    ).fetchone()


def reminder_is_due_for_email(row, now_utc):
    """Evaluate a stored local reminder time without needing a logged-in request."""
    if not row["enabled"]:
        return False
    try:
        scheduled = datetime.fromisoformat(f"{row['reminder_date']}T{row['reminder_time']}").replace(
            tzinfo=ZoneInfo(row["timezone"])
        )
    except (ValueError, ZoneInfoNotFoundError):
        return False
    return scheduled.astimezone(timezone.utc) <= now_utc


def smtp_configured():
    return all(os.environ.get(key, "").strip() for key in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"))


def send_reminder_email(row):
    """Send one reminder using environment-only SMTP credentials.

    The recipient is deliberately read from the joined user row, never from request data
    or a sender configuration value.
    """
    recipient = row["email"].strip().lower()
    if not valid_email(recipient):
        return False, "invalid_recipient"
    if not smtp_configured():
        return False, "smtp_not_configured"

    sender = os.environ["SMTP_FROM"].strip()
    if not valid_email(sender):
        return False, "smtp_not_configured"
    message = EmailMessage()
    message["Subject"] = f"Fitness reminder: {row['title']}"
    message["From"] = sender
    message["To"] = recipient
    scheduled = f"{row['reminder_date']} at {row['reminder_time']} ({row['timezone']})"
    note = f"\n\nNote: {row['note']}" if row["note"] else ""
    message.set_content(
        f"Hi {row['name']},\n\nIt's time for your scheduled workout: {row['title']}.\n"
        f"Scheduled for: {scheduled}.\nKeep moving—you've got this!{note}\n\nRepAI Fitness Trainer"
    )
    try:
        host = os.environ["SMTP_HOST"].strip()
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ.get("SMTP_USERNAME", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "")
        if os.environ.get("SMTP_USE_SSL", "").strip() == "1":
            client = smtplib.SMTP_SSL(host, port, timeout=20)
        else:
            client = smtplib.SMTP(host, port, timeout=20)
        with client:
            if os.environ.get("SMTP_USE_TLS", "1").strip() != "0" and os.environ.get("SMTP_USE_SSL", "").strip() != "1":
                client.starttls()
            if username:
                client.login(username, password)
            client.send_message(message)
        return True, None
    except (OSError, ValueError, smtplib.SMTPException):
        # Do not log SMTP exceptions: they can include authentication or recipient data.
        return False, "smtp_error"


def dispatch_due_reminder_emails(now_utc=None):
    """Poll due reminders and atomically claim each reminder before sending it.

    This function uses its own connection because it is called by a background thread,
    outside Flask request and session contexts.
    """
    if not smtp_configured():
        return 0
    now_utc = now_utc or datetime.now(timezone.utc)
    now_text = now_utc.isoformat()
    db = sqlite3.connect(app.config["DATABASE"], timeout=15)
    db.row_factory = sqlite3.Row
    sent = 0
    try:
        rows = db.execute("""
            SELECT r.*, u.name, u.email
            FROM workout_reminders r
            JOIN users u ON u.id = r.user_id AND u.active = 1
            JOIN user_profiles p ON p.user_id = u.id AND p.notify_email = 1
            LEFT JOIN reminder_email_deliveries d ON d.reminder_id = r.id
            WHERE r.enabled = 1 AND d.id IS NULL
            ORDER BY r.reminder_date, r.reminder_time
        """).fetchall()
        for row in rows:
            if not reminder_is_due_for_email(row, now_utc):
                continue
            # The UNIQUE reminder_id insert is the cross-process duplicate guard.
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute("""INSERT INTO reminder_email_deliveries
                    (reminder_id, user_id, status, created_at, updated_at)
                    VALUES (?, ?, 'sending', ?, ?)""", (row["id"], row["user_id"], now_text, now_text))
                db.commit()
            except sqlite3.IntegrityError:
                db.rollback()
                continue
            except sqlite3.OperationalError:
                db.rollback()
                continue

            ok, reason = send_reminder_email(row)
            status = "sent" if ok else ("skipped_invalid_recipient" if reason == "invalid_recipient" else "failed")
            db.execute("""UPDATE reminder_email_deliveries
                SET status = ?, attempted_at = ?, sent_at = ?, failure_reason = ?, updated_at = ?
                WHERE reminder_id = ? AND user_id = ? AND status = 'sending'""",
                (status, now_text, now_text if ok else None, reason, now_text, row["id"], row["user_id"]))
            db.commit()
            sent += int(ok)
    finally:
        db.close()
    return sent


def parse_report_range():
    start = parse_date(request.values.get("start"))
    end = parse_date(request.values.get("end"))
    if start and end and start > end:
        start, end = end, start
    return start, end


def goal_progress_rows():
    goals = get_db().execute("SELECT * FROM fitness_goals WHERE user_id = ? ORDER BY deadline", (session["user_id"],)).fetchall()
    return [(goal, goal_progress(goal)) for goal in goals]


def generate_report_pdf(start_date=None, end_date=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RepTitle", parent=styles["Title"], textColor=colors.HexColor("#111827")))
    styles.add(ParagraphStyle(name="RepSmall", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#4b5563")))
    story = []
    user = current_user()
    summary = analytics_summary(start_date, end_date)
    records = personal_records()
    range_label = f"{start_date or 'All'} to {end_date or 'All'}"
    story.append(Paragraph("RepAI Fitness Report", styles["RepTitle"]))
    story.append(Paragraph(f"Account: {user['name']} ({user['email']})", styles["RepSmall"]))
    story.append(Paragraph(f"Date range: {range_label}", styles["RepSmall"]))
    story.append(Paragraph(f"Generated: {utcnow()}", styles["RepSmall"]))
    story.append(Spacer(1, 0.18 * inch))
    overview = [
        ["Metric", "Value"],
        ["Workouts", summary["total_workouts"]],
        ["Total reps", summary["total_reps"]],
        ["Duration minutes", round(summary["total_duration"] / 60, 1)],
        ["Current streak days", summary["streak"]],
    ]
    story.append(Table(overview, hAlign="LEFT", colWidths=[2.2 * inch, 2.0 * inch], style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#bef264")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Spacer(1, 0.18 * inch))
    exercise_rows = [["Exercise", "Reps"]] + [[key, value] for key, value in summary["per_exercise"].items()]
    story.append(Paragraph("Exercise Counts", styles["Heading2"]))
    story.append(Table(exercise_rows, hAlign="LEFT", colWidths=[2.2 * inch, 2.0 * inch], style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Personal Records", styles["Heading2"]))
    record_rows = [
        ["Record", "Value", "Date"],
        ["Highest reps", records["highest_reps"]["value"], records["highest_reps"]["date"] or "No workouts"],
        ["Longest workout", f"{records['longest_duration']['value']} minutes", records["longest_duration"]["date"] or "No workouts"],
    ]
    for item in records["exercise"].values():
        record_rows.append([item["name"], item["value"], item["date"] or "No record"])
    story.append(Table(record_rows, hAlign="LEFT", colWidths=[2.2 * inch, 1.5 * inch, 1.5 * inch], style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Goal Progress", styles["Heading2"]))
    goals = goal_progress_rows()
    if goals:
        goal_rows = [["Goal", "Type", "Progress"]]
        for goal, progress in goals:
            goal_rows.append([goal["title"], goal["goal_type"].replace("_", " "), f"{progress['value']} / {goal['target_value']} ({progress['percent']}%)"])
    else:
        goal_rows = [["Goal", "Type", "Progress"], ["No goals saved", "-", "-"]]
    story.append(Table(goal_rows, hAlign="LEFT", colWidths=[2.3 * inch, 1.4 * inch, 2.1 * inch], style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Workout History", styles["Heading2"]))
    workouts = summary["rows"][:25]
    if workouts:
        workout_rows_pdf = [["Date", "Source", "Reps", "Duration min"]]
        for row in workouts:
            workout_rows_pdf.append([row["ended_at"][:10], row["source_type"], row["total_reps"], round(workout_duration(row) / 60, 1)])
    else:
        workout_rows_pdf = [["Date", "Source", "Reps", "Duration min"], ["No workouts in this range", "-", "-", "-"]]
    story.append(Table(workout_rows_pdf, hAlign="LEFT", colWidths=[1.35 * inch, 1.4 * inch, 1.0 * inch, 1.35 * inch], style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    doc.build(story)
    buffer.seek(0)
    return buffer


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
        elif not valid_email(email):
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirmation:
            flash("Passwords do not match.", "error")
        else:
            try:
                db = get_db()
                admin_emails = {item.strip().lower() for item in os.environ.get("ADMIN_EMAILS", "").split(",") if item.strip()}
                db.execute("INSERT INTO users (name, email, password_hash, created_at, is_admin) VALUES (?, ?, ?, ?, ?)",
                           (name, email, generate_password_hash(password), utcnow(), 1 if email in admin_emails else 0))
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
        elif not user["active"]:
            flash("This account is disabled.", "error")
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


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    db = get_db()
    profile_row = current_profile()
    if request.method == "POST":
        values = profile_form_values(request.form)
        if "notify_in_app" not in request.form:
            values["notify_in_app"] = profile_row["notify_in_app"] if profile_row else 1
        if "notify_email" not in request.form:
            values["notify_email"] = profile_row["notify_email"] if profile_row else 0
        if "measurement_units" not in request.form:
            values["measurement_units"] = profile_row["measurement_units"] if profile_row else "metric"
        if "timezone" not in request.form:
            values["timezone"] = profile_row["timezone"] if profile_row else "Asia/Calcutta"
        errors = validate_profile_values(values)
        if errors:
            for message in errors:
                flash(message, "error")
        else:
            save_profile_values(values)
            db.commit()
            flash("Profile saved.", "success")
            return redirect(url_for("profile"))
    profile_row = current_profile()
    return render_template(
        "profile.html",
        profile=profile_row,
        goals=["weight_loss", "muscle_gain", "maintenance", "general_fitness"],
        activities=list(ACTIVITY_MULTIPLIERS.keys()),
        preferences=["vegetarian", "vegan", "non-vegetarian", "pescatarian", "eggetarian", "other"],
    )


@app.route("/settings")
@login_required
def settings():
    return render_template(
        "settings.html",
        user=current_user(),
        profile=current_profile(),
        goals=["weight_loss", "muscle_gain", "maintenance", "general_fitness"],
        activities=list(ACTIVITY_MULTIPLIERS.keys()),
        preferences=["vegetarian", "vegan", "non-vegetarian", "pescatarian", "eggetarian", "other"],
        common_timezones=["Asia/Calcutta", "UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "Europe/London"],
    )


@app.post("/settings/account")
@login_required
def update_account():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    if not name or len(name) > 80:
        flash("Please enter a name up to 80 characters.", "error")
        return redirect(url_for("settings"))
    if not valid_email(email):
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("settings"))
    try:
        db = get_db()
        db.execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (name, email, session["user_id"]))
        # A corrected address may safely receive any reminder that previously failed.
        db.execute("DELETE FROM reminder_email_deliveries WHERE user_id = ? AND status != 'sent'", (session["user_id"],))
        db.commit()
        session["user_name"] = name
        flash("Account details updated.", "success")
    except sqlite3.IntegrityError:
        flash("That email address is already in use.", "error")
    return redirect(url_for("settings"))


@app.post("/settings/preferences")
@login_required
def update_preferences():
    profile_row = current_profile()
    values = profile_form_values(request.form)
    if profile_row:
        for key in ("age", "height_cm", "weight_kg", "fitness_goal", "activity_level", "dietary_preference",
                    "allergies", "liked_foods", "disliked_foods", "meals_per_day", "preferred_meal_times",
                    "budget", "regional_preference", "medical_notes"):
            if key not in request.form:
                values[key] = profile_row[key]
    errors = validate_profile_values(values)
    if errors:
        for message in errors:
            flash(message, "error")
    else:
        save_profile_values(values)
        get_db().commit()
        flash("Preferences saved.", "success")
    return redirect(url_for("settings"))


@app.post("/settings/password")
@login_required
def change_password():
    current = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirmation = request.form.get("confirm_password", "")
    user = get_db().execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], current):
        flash("Current password is incorrect.", "error")
    elif len(new_password) < 8:
        flash("New password must be at least 8 characters.", "error")
    elif new_password != confirmation:
        flash("New passwords do not match.", "error")
    else:
        get_db().execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new_password), session["user_id"]))
        get_db().commit()
        flash("Password changed securely.", "success")
    return redirect(url_for("settings"))


@app.post("/settings/delete-account")
@login_required
def delete_account():
    confirmation = request.form.get("confirm_delete", "").strip()
    user = current_user()
    if confirmation != user["email"]:
        flash("Type your email address exactly to delete your account.", "error")
        return redirect(url_for("settings"))
    db = get_db()
    for table in ("meal_logs", "meal_plans", "workout_schedules", "workout_routines", "fitness_goals",
                  "weight_entries", "achievements", "reminder_email_deliveries", "workout_reminders", "workout_sessions", "user_profiles"):
        db.execute(f"DELETE FROM {table} WHERE user_id = ?", (session["user_id"],))
    db.execute("UPDATE users SET active = 0, email = ?, name = 'Deleted account' WHERE id = ?",
               (f"deleted-{session['user_id']}@local.invalid", session["user_id"]))
    db.commit()
    session.clear()
    flash("Your account and fitness data were deleted from this app.", "success")
    return redirect(url_for("login"))


@app.route("/reminders")
@login_required
def reminders():
    rows = get_db().execute(
        """SELECT r.*, d.status AS email_status, d.sent_at AS email_sent_at, d.failure_reason AS email_failure_reason
           FROM workout_reminders r
           LEFT JOIN reminder_email_deliveries d ON d.reminder_id = r.id AND d.user_id = r.user_id
           WHERE r.user_id = ? ORDER BY r.reminder_date, r.reminder_time""",
        (session["user_id"],),
    ).fetchall()
    return render_template("reminders.html", reminders=rows, timezone_name=default_timezone(), today=today_iso())


@app.post("/reminders")
@login_required
def save_reminder():
    reminder_id = as_int(request.form.get("reminder_id"))
    title = request.form.get("title", "").strip()
    reminder_date = parse_date(request.form.get("reminder_date"))
    reminder_time = request.form.get("reminder_time", "").strip()
    note = request.form.get("note", "").strip()
    tz_name = request.form.get("timezone", default_timezone()).strip() or default_timezone()
    enabled = 1 if request.form.get("enabled") == "on" else 0
    if not title or len(title) > 120 or reminder_date is None or len(reminder_time) != 5 or not valid_timezone(tz_name):
        flash("Reminder title, date, time, and timezone are required.", "error")
        return redirect(url_for("reminders"))
    db = get_db()
    if reminder_id:
        if reminder_owned(reminder_id) is None:
            abort(404)
        db.execute("""UPDATE workout_reminders SET title = ?, reminder_date = ?, reminder_time = ?,
                      timezone = ?, note = ?, enabled = ?, last_notified_at = NULL, updated_at = ?
                      WHERE id = ? AND user_id = ?""",
                   (title, reminder_date.isoformat(), reminder_time, tz_name, note, enabled, utcnow(),
                    reminder_id, session["user_id"]))
        db.execute("DELETE FROM reminder_email_deliveries WHERE reminder_id = ? AND user_id = ?",
                   (reminder_id, session["user_id"]))
        flash("Reminder updated.", "success")
    else:
        db.execute("""INSERT INTO workout_reminders
                      (user_id, title, reminder_date, reminder_time, timezone, note, enabled, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (session["user_id"], title, reminder_date.isoformat(), reminder_time, tz_name, note, enabled, utcnow(), utcnow()))
        flash("Reminder created.", "success")
    db.commit()
    return redirect(url_for("reminders"))


@app.post("/reminders/<int:reminder_id>/<action>")
@login_required
def update_reminder(reminder_id, action):
    row = reminder_owned(reminder_id)
    if row is None:
        abort(404)
    if action == "enable":
        get_db().execute("UPDATE workout_reminders SET enabled = 1, updated_at = ? WHERE id = ? AND user_id = ?", (utcnow(), reminder_id, session["user_id"]))
    elif action == "disable":
        get_db().execute("UPDATE workout_reminders SET enabled = 0, updated_at = ? WHERE id = ? AND user_id = ?", (utcnow(), reminder_id, session["user_id"]))
    elif action == "dismiss":
        get_db().execute("UPDATE workout_reminders SET last_notified_at = ? WHERE id = ? AND user_id = ?", (utcnow(), reminder_id, session["user_id"]))
    elif action == "delete":
        get_db().execute("DELETE FROM workout_reminders WHERE id = ? AND user_id = ?", (reminder_id, session["user_id"]))
    else:
        abort(404)
    get_db().commit()
    flash("Reminder updated.", "success")
    return redirect(request.referrer or url_for("reminders"))


@app.route("/reports")
@login_required
def reports():
    start, end = parse_report_range()
    return render_template("reports.html", start=start.isoformat() if start else "", end=end.isoformat() if end else "",
                           summary=analytics_summary(start, end), records=personal_records(), goals=goal_progress_rows())


@app.get("/reports/pdf")
@login_required
def download_report_pdf():
    start, end = parse_report_range()
    pdf = generate_report_pdf(start, end)
    filename = f"repai-report-{start or 'all'}-{end or 'all'}.pdf"
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name=filename)


@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()
    stats = {
        "users": db.execute("SELECT COUNT(*) count FROM users WHERE active = 1").fetchone()["count"],
        "admins": db.execute("SELECT COUNT(*) count FROM users WHERE active = 1 AND is_admin = 1").fetchone()["count"],
        "workouts": db.execute("SELECT COUNT(*) count FROM workout_sessions").fetchone()["count"],
        "reps": db.execute("SELECT COALESCE(SUM(total_reps), 0) total FROM workout_sessions").fetchone()["total"],
        "reminders": db.execute("SELECT COUNT(*) count FROM workout_reminders WHERE enabled = 1").fetchone()["count"],
    }
    users = db.execute("""
        SELECT u.id, u.name, u.email, u.created_at, u.is_admin, u.active,
               COUNT(w.id) workout_count, COALESCE(SUM(w.total_reps), 0) total_reps
        FROM users u
        LEFT JOIN workout_sessions w ON w.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    exercises = db.execute("SELECT * FROM custom_exercises ORDER BY active DESC, name").fetchall()
    return render_template("admin.html", stats=stats, users=users, exercises=exercises)


@app.post("/admin/users/<int:user_id>/<action>")
@admin_required
def admin_update_user(user_id, action):
    if user_id == session["user_id"] and action in {"disable", "demote"}:
        flash("You cannot remove your own admin access or disable your own account.", "error")
        return redirect(url_for("admin_dashboard"))
    if action == "promote":
        get_db().execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
    elif action == "demote":
        get_db().execute("UPDATE users SET is_admin = 0 WHERE id = ?", (user_id,))
    elif action == "enable":
        get_db().execute("UPDATE users SET active = 1 WHERE id = ?", (user_id,))
    elif action == "disable":
        get_db().execute("UPDATE users SET active = 0 WHERE id = ?", (user_id,))
    else:
        abort(404)
    get_db().commit()
    flash("User account updated.", "success")
    return redirect(url_for("admin_dashboard"))


@app.post("/admin/exercises")
@admin_required
def admin_save_exercise():
    exercise_id = as_int(request.form.get("exercise_id"))
    values = {
        "name": request.form.get("name", "").strip(),
        "category": request.form.get("category", "").strip(),
        "muscles": request.form.get("muscles", "").strip(),
        "difficulty": request.form.get("difficulty", "").strip(),
        "equipment": request.form.get("equipment", "").strip(),
        "description": request.form.get("description", "").strip(),
        "instructions": request.form.get("instructions", "").strip(),
        "active": 1 if request.form.get("active") == "on" else 0,
        "updated_at": utcnow(),
    }
    if not values["name"] or not values["category"] or not values["muscles"] or not values["difficulty"] or not values["description"]:
        flash("Exercise name, category, muscles, difficulty, and description are required.", "error")
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    if exercise_id:
        existing = db.execute("SELECT id FROM custom_exercises WHERE id = ?", (exercise_id,)).fetchone()
        if existing is None:
            abort(404)
        db.execute("""UPDATE custom_exercises SET name = :name, category = :category, muscles = :muscles,
                      difficulty = :difficulty, equipment = :equipment, description = :description,
                      instructions = :instructions, active = :active, updated_at = :updated_at
                      WHERE id = :id""", {**values, "id": exercise_id})
        flash("Exercise updated.", "success")
    else:
        db.execute("""INSERT INTO custom_exercises
                      (name, category, muscles, difficulty, equipment, description, instructions, active,
                       created_by, created_at, updated_at)
                      VALUES (:name, :category, :muscles, :difficulty, :equipment, :description,
                              :instructions, :active, :created_by, :created_at, :updated_at)""",
                   {**values, "created_by": session["user_id"], "created_at": utcnow()})
        flash("Exercise added to the library.", "success")
    db.commit()
    return redirect(url_for("admin_dashboard"))


@app.post("/admin/exercises/<int:exercise_id>/delete")
@admin_required
def admin_delete_exercise(exercise_id):
    get_db().execute("DELETE FROM custom_exercises WHERE id = ?", (exercise_id,))
    get_db().commit()
    flash("Exercise deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/food-scanner")
@login_required
def food_scanner():
    recent_logs = get_db().execute(
        "SELECT * FROM food_scan_logs WHERE user_id = ? ORDER BY logged_at DESC, id DESC LIMIT 10",
        (session["user_id"],),
    ).fetchall()
    return render_template("food_scanner.html", recent_logs=recent_logs)


@app.post("/api/food/scan")
@login_required
def scan_food_image():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify(error="Choose a food photo first."), 400
    if not allowed_food_image(file.filename):
        return jsonify(error="Unsupported format. Upload JPG, PNG, or WebP."), 400
    if image_size(file) > FOOD_IMAGE_MAX_BYTES:
        return jsonify(error="Food photo is too large. Maximum upload size is 8 MB."), 400

    original_name = secure_filename(file.filename)
    if not original_name:
        return jsonify(error="Invalid image filename."), 400

    suffix = Path(original_name).suffix.lower()
    UPLOAD_DIR.mkdir(exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
        file.save(temp_path)
        validate_food_image(temp_path)
        scan = classify_food(temp_path)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except FoodModelUnavailable as exc:
        return jsonify(error=str(exc)), 503
    except OSError as exc:
        return jsonify(error=f"Unable to save this image: {exc}"), 400
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    top_prediction = scan["predictions"][0] if scan["predictions"] else None
    nutrition = None
    nutrition_error = None
    if top_prediction:
        try:
            nutrition = lookup_nutrition(top_prediction["display_name"], 100).as_dict()
        except NutritionLookupError as exc:
            nutrition_error = str(exc)

    return jsonify({
        "model": scan["model"],
        "device": scan["device"],
        "predictions": scan["predictions"],
        "nutrition": nutrition,
        "nutrition_error": nutrition_error,
    })


@app.post("/api/food/nutrition")
@login_required
def food_nutrition_lookup():
    data = request.get_json(silent=True) or {}
    food_name = (data.get("food_name") or "").strip()
    serving_grams = as_float(data.get("serving_grams"), 100)
    if not food_name:
        return jsonify(error="Food name is required."), 400
    if serving_grams is None or serving_grams <= 0 or serving_grams > 5000:
        return jsonify(error="Serving size must be between 1 and 5000 grams."), 400
    try:
        nutrition = lookup_nutrition(food_name, serving_grams).as_dict()
    except NutritionLookupError as exc:
        return jsonify(error=str(exc)), 502
    return jsonify(nutrition=nutrition)


@app.post("/api/food/log")
@login_required
def log_scanned_food():
    data = request.get_json(silent=True) or {}
    nutrition = data.get("nutrition") or {}
    per_100g = nutrition.get("per_100g") or {}
    dish_name = (data.get("dish_name") or nutrition.get("name") or "").strip()
    serving_grams = as_float(nutrition.get("serving_grams"), 100)
    if not dish_name:
        return jsonify(error="Food name is required before saving."), 400
    if serving_grams is None or serving_grams <= 0 or serving_grams > 5000:
        return jsonify(error="Serving size must be between 1 and 5000 grams."), 400
    try:
        normalized_per_100g = {
            key: max(0, float(per_100g.get(key) or 0))
            for key in ("calories", "protein", "carbs", "fat")
        }
    except (TypeError, ValueError):
        return jsonify(error="Nutrition values must be numeric."), 400
    totals = calculate_totals(normalized_per_100g, serving_grams)
    selected_label = (data.get("selected_label") or "").strip()[:100] or None
    source = (nutrition.get("source") or "Food scanner estimate").strip()[:100]
    fdc_id = as_int(nutrition.get("fdc_id"))
    confidence = as_float(data.get("confidence"))

    duplicate = get_db().execute(
        """
        SELECT id FROM food_scan_logs
        WHERE user_id = ? AND lower(dish_name) = lower(?) AND logged_at >= ?
        ORDER BY logged_at DESC LIMIT 1
        """,
        (session["user_id"], dish_name, (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(timespec="seconds")),
    ).fetchone()
    if duplicate:
        return jsonify(error="This scanned meal was just saved. Wait a moment before saving it again."), 409

    get_db().execute(
        """
        INSERT INTO food_scan_logs
        (user_id, dish_name, selected_label, serving_grams, calories, protein, carbs, fat,
         source, fdc_id, confidence, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"], dish_name[:120], selected_label, serving_grams,
            totals["calories"], totals["protein"], totals["carbs"], totals["fat"],
            source, fdc_id, confidence, utcnow(),
        ),
    )
    get_db().commit()
    return jsonify(message="Scanned meal saved to your food log.", totals=totals)


@app.route("/meal-planner")
@login_required
def meal_planner():
    profile_row = current_profile()
    plan = latest_meal_plan()
    plan_data = json.loads(plan["plan_json"]) if plan else None
    targets = json.loads(plan["target_json"]) if plan else None
    logs = meal_logs_for_plan(plan["id"]) if plan else {}
    today = today_iso()
    daily_progress = None
    if plan_data:
        day = next((item for item in plan_data["days"] if item["date"] == today), plan_data["days"][0])
        consumed = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
        planned = day["totals"]
        completed = 0
        for meal_index, meal in enumerate(day["meals"]):
            log = logs.get((plan_data["days"].index(day), meal_index))
            if log and log["completed"]:
                completed += 1
                nutrition = meal["nutrition"]
                consumed["calories"] += log["actual_calories"] if log["actual_calories"] is not None else nutrition["calories"]
                consumed["protein"] += log["actual_protein"] if log["actual_protein"] is not None else nutrition["protein"]
                consumed["carbs"] += log["actual_carbs"] if log["actual_carbs"] is not None else nutrition["carbs"]
                consumed["fat"] += log["actual_fat"] if log["actual_fat"] is not None else nutrition["fat"]
        daily_progress = {
            "date": day["date"],
            "planned": planned,
            "consumed": {key: round(value, 1) for key, value in consumed.items()},
            "completed": completed,
            "total_meals": len(day["meals"]),
        }
    return render_template(
        "meal_planner.html", profile=profile_row, plan=plan, plan_data=plan_data,
        targets=targets, logs=logs, daily_progress=daily_progress, today=today,
    )


@app.post("/meal-planner/generate")
@login_required
def generate_meal_plan_route():
    profile_row = current_profile()
    if not profile_ready(profile_row):
        flash("Complete age, height, weight, goal, activity level, and dietary preference before generating a meal plan.", "error")
        return redirect(url_for("profile"))
    if profile_row["medical_notes"]:
        flash("Medical notes were saved. This planner is not a medical diet; please use qualified dietary guidance for medical conditions or pregnancy.", "error")
    plan_data, targets = generate_meal_plan(profile_row)
    db = get_db()
    db.execute(
        "INSERT INTO meal_plans (user_id, week_start, plan_json, target_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (session["user_id"], plan_data["days"][0]["date"], json.dumps(plan_data), json.dumps(targets), utcnow()),
    )
    db.commit()
    flash("Meal plan generated with estimated nutrition.", "success")
    return redirect(url_for("meal_planner"))


@app.post("/meal-planner/log")
@login_required
def log_meal():
    plan_id = as_int(request.form.get("plan_id"))
    day_index = as_int(request.form.get("day_index"))
    meal_index = as_int(request.form.get("meal_index"))
    plan = get_db().execute(
        "SELECT * FROM meal_plans WHERE id = ? AND user_id = ?", (plan_id, session["user_id"])
    ).fetchone()
    if plan is None or day_index is None or meal_index is None or day_index < 0 or meal_index < 0:
        flash("Meal log could not be saved.", "error")
        return redirect(url_for("meal_planner"))
    plan_data = json.loads(plan["plan_json"])
    try:
        day = plan_data["days"][day_index]
        meal_date = day["date"]
        day["meals"][meal_index]
    except (IndexError, KeyError):
        flash("Meal log could not be saved.", "error")
        return redirect(url_for("meal_planner"))
    completed = 1 if request.form.get("completed") == "on" else 0
    values = (
        session["user_id"], plan_id, meal_date, day_index, meal_index, completed,
        request.form.get("actual_intake", "").strip(),
        as_float(request.form.get("actual_calories")),
        as_float(request.form.get("actual_protein")),
        as_float(request.form.get("actual_carbs")),
        as_float(request.form.get("actual_fat")),
        utcnow(),
    )
    get_db().execute("""
        INSERT INTO meal_logs
        (user_id, meal_plan_id, log_date, day_index, meal_index, completed, actual_intake,
         actual_calories, actual_protein, actual_carbs, actual_fat, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, meal_plan_id, day_index, meal_index) DO UPDATE SET
            completed=excluded.completed, actual_intake=excluded.actual_intake,
            actual_calories=excluded.actual_calories, actual_protein=excluded.actual_protein,
            actual_carbs=excluded.actual_carbs, actual_fat=excluded.actual_fat,
            logged_at=excluded.logged_at
    """, values)
    get_db().commit()
    flash("Meal log updated.", "success")
    return redirect(url_for("meal_planner"))


@app.route("/exercises")
@login_required
def exercise_library():
    query = request.args.get("q", "").strip().lower()
    category = request.args.get("category", "").strip()
    difficulty = request.args.get("difficulty", "").strip()
    items = []
    for item in exercise_library_items():
        haystack = " ".join([item["name"], item["description"], item["muscles"], item["category"]]).lower()
        if query and query not in haystack:
            continue
        if category and item["category"] != category:
            continue
        if difficulty and item["difficulty"] != difficulty:
            continue
        items.append(item)
    categories = sorted({item["category"] for item in exercise_library_items()})
    difficulties = sorted({item["difficulty"] for item in exercise_library_items()})
    return render_template("exercise_library.html", exercises=items, categories=categories,
                           difficulties=difficulties, selected_category=category,
                           selected_difficulty=difficulty, query=query)


@app.route("/exercise-demo/<path:filename>")
@login_required
def exercise_demo(filename):
    safe = secure_filename(filename)
    demo_path = BASE_DIR / "Exercise_videos" / safe
    if not demo_path.exists():
        abort(404)
    return send_from_directory(str(BASE_DIR / "Exercise_videos"), safe, as_attachment=False)


def parse_routine_items(form):
    items = []
    names = form.getlist("exercise_key")
    sets = form.getlist("sets")
    reps = form.getlist("reps")
    durations = form.getlist("duration_seconds")
    rests = form.getlist("rest_seconds")
    for index, key in enumerate(names):
        if not key:
            continue
        exercise = exercise_by_key(key)
        if exercise is None:
            continue
        item = {
            "exercise_key": key,
            "exercise_name": exercise["name"],
            "sets": max(1, min(as_int(sets[index] if index < len(sets) else None, 1), 20)),
            "reps": max(0, min(as_int(reps[index] if index < len(reps) else None, 0), 1000)),
            "duration_seconds": max(0, min(as_int(durations[index] if index < len(durations) else None, 0), 7200)),
            "rest_seconds": max(0, min(as_int(rests[index] if index < len(rests) else None, 60), 1800)),
        }
        items.append(item)
    return items


@app.route("/planner")
@login_required
def workout_planner():
    db = get_db()
    routines = db.execute("SELECT * FROM workout_routines WHERE user_id = ? ORDER BY updated_at DESC",
                          (session["user_id"],)).fetchall()
    routine_views = [(routine, json.loads(routine["items_json"])) for routine in routines]
    schedules = db.execute("""
        SELECT s.*, r.name routine_name, r.items_json FROM workout_schedules s
        JOIN workout_routines r ON r.id = s.routine_id
        WHERE s.user_id = ? ORDER BY s.scheduled_date ASC, s.id ASC
    """, (session["user_id"],)).fetchall()
    today = today_iso()
    return render_template("workout_planner.html", routines=routines, routine_views=routine_views, schedules=schedules,
                           exercises=exercise_library_items(), today=today)


@app.post("/planner/routine")
@login_required
def save_routine():
    routine_id = as_int(request.form.get("routine_id"))
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    items = parse_routine_items(request.form)
    if not name or len(name) > 100:
        flash("Routine name is required and must be under 100 characters.", "error")
        return redirect(url_for("workout_planner"))
    if not items:
        flash("Add at least one valid exercise to the routine.", "error")
        return redirect(url_for("workout_planner"))
    db = get_db()
    if routine_id:
        if routine_owned(routine_id) is None:
            abort(404)
        db.execute("""UPDATE workout_routines SET name = ?, description = ?, items_json = ?, updated_at = ?
                      WHERE id = ? AND user_id = ?""",
                   (name, description, json.dumps(items), utcnow(), routine_id, session["user_id"]))
        flash("Routine updated.", "success")
    else:
        db.execute("""INSERT INTO workout_routines (user_id, name, description, items_json, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, ?)""",
                   (session["user_id"], name, description, json.dumps(items), utcnow(), utcnow()))
        flash("Routine created.", "success")
    db.commit()
    return redirect(url_for("workout_planner"))


@app.post("/planner/routine/<int:routine_id>/duplicate")
@login_required
def duplicate_routine(routine_id):
    routine = routine_owned(routine_id)
    if routine is None:
        abort(404)
    get_db().execute("""INSERT INTO workout_routines (user_id, name, description, items_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     (session["user_id"], routine["name"] + " Copy", routine["description"],
                      routine["items_json"], utcnow(), utcnow()))
    get_db().commit()
    flash("Routine duplicated.", "success")
    return redirect(url_for("workout_planner"))


@app.post("/planner/routine/<int:routine_id>/delete")
@login_required
def delete_routine(routine_id):
    if routine_owned(routine_id) is None:
        abort(404)
    get_db().execute("DELETE FROM workout_schedules WHERE user_id = ? AND routine_id = ?",
                     (session["user_id"], routine_id))
    get_db().execute("DELETE FROM workout_routines WHERE user_id = ? AND id = ?",
                     (session["user_id"], routine_id))
    get_db().commit()
    flash("Routine deleted.", "success")
    return redirect(url_for("workout_planner"))


@app.post("/planner/schedule")
@login_required
def schedule_routine():
    routine_id = as_int(request.form.get("routine_id"))
    scheduled = parse_date(request.form.get("scheduled_date"))
    notes = request.form.get("notes", "").strip()
    if scheduled is None or routine_owned(routine_id) is None:
        flash("Choose a valid routine and schedule date.", "error")
        return redirect(url_for("workout_planner"))
    get_db().execute("""INSERT INTO workout_schedules (user_id, routine_id, scheduled_date, notes, created_at)
                        VALUES (?, ?, ?, ?, ?)""",
                     (session["user_id"], routine_id, scheduled.isoformat(), notes, utcnow()))
    get_db().commit()
    flash("Workout scheduled.", "success")
    return redirect(url_for("workout_planner"))


@app.post("/planner/schedule/<int:schedule_id>/<action>")
@login_required
def update_schedule(schedule_id, action):
    row = get_db().execute("SELECT * FROM workout_schedules WHERE id = ? AND user_id = ?",
                           (schedule_id, session["user_id"])).fetchone()
    if row is None:
        abort(404)
    if action == "complete":
        get_db().execute("""UPDATE workout_schedules SET status = 'completed', completed_at = ?
                            WHERE id = ? AND user_id = ?""", (utcnow(), schedule_id, session["user_id"]))
        flash("Scheduled workout marked complete.", "success")
    elif action == "delete":
        get_db().execute("DELETE FROM workout_schedules WHERE id = ? AND user_id = ?",
                         (schedule_id, session["user_id"]))
        flash("Schedule removed.", "success")
    else:
        abort(404)
    get_db().commit()
    refresh_achievements()
    return redirect(url_for("workout_planner"))


@app.route("/goals")
@login_required
def fitness_goals():
    goals = get_db().execute("SELECT * FROM fitness_goals WHERE user_id = ? ORDER BY status, deadline",
                             (session["user_id"],)).fetchall()
    enriched = [(goal, goal_progress(goal)) for goal in goals]
    return render_template("fitness_goals.html", goals=enriched, exercises=exercise_library_items())


@app.post("/goals")
@login_required
def save_goal():
    goal_id = as_int(request.form.get("goal_id"))
    goal_type = request.form.get("goal_type", "").strip()
    title = request.form.get("title", "").strip()
    target = as_float(request.form.get("target_value"))
    exercise_key = request.form.get("exercise_key", "").strip() or None
    deadline = request.form.get("deadline", "").strip() or None
    valid_types = {"frequency", "consistency", "rep_target", "duration", "performance", "weight"}
    if goal_type not in valid_types or not title or target is None or target <= 0:
        flash("Goal type, title, and a positive target are required.", "error")
        return redirect(url_for("fitness_goals"))
    if exercise_key and exercise_by_key(exercise_key) is None:
        flash("Choose a valid exercise for this goal.", "error")
        return redirect(url_for("fitness_goals"))
    if deadline and parse_date(deadline) is None:
        flash("Choose a valid deadline.", "error")
        return redirect(url_for("fitness_goals"))
    db = get_db()
    if goal_id:
        existing = db.execute("SELECT id FROM fitness_goals WHERE id = ? AND user_id = ?",
                              (goal_id, session["user_id"])).fetchone()
        if existing is None:
            abort(404)
        db.execute("""UPDATE fitness_goals SET goal_type = ?, title = ?, target_value = ?, exercise_key = ?,
                      deadline = ?, updated_at = ? WHERE id = ? AND user_id = ?""",
                   (goal_type, title, target, exercise_key, deadline, utcnow(), goal_id, session["user_id"]))
        flash("Goal updated.", "success")
    else:
        db.execute("""INSERT INTO fitness_goals
                      (user_id, goal_type, title, target_value, exercise_key, deadline, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                   (session["user_id"], goal_type, title, target, exercise_key, deadline, utcnow(), utcnow()))
        flash("Goal created.", "success")
    db.commit()
    refresh_achievements()
    return redirect(url_for("fitness_goals"))


@app.post("/goals/<int:goal_id>/delete")
@login_required
def delete_goal(goal_id):
    existing = get_db().execute(
        "SELECT id FROM fitness_goals WHERE id = ? AND user_id = ?", (goal_id, session["user_id"])
    ).fetchone()
    if existing is None:
        abort(404)
    get_db().execute("DELETE FROM fitness_goals WHERE id = ? AND user_id = ?", (goal_id, session["user_id"]))
    get_db().commit()
    flash("Goal deleted.", "success")
    return redirect(url_for("fitness_goals"))


@app.route("/analytics")
@login_required
def progress_analytics():
    start = parse_date(request.args.get("start"))
    end = parse_date(request.args.get("end"))
    summary = analytics_summary(start, end)
    weights = get_db().execute("SELECT * FROM weight_entries WHERE user_id = ? ORDER BY entry_date",
                               (session["user_id"],)).fetchall()
    return render_template("analytics.html", summary=summary, weights=weights,
                           start=start.isoformat() if start else "", end=end.isoformat() if end else "")


@app.post("/analytics/weight")
@login_required
def save_weight_entry():
    entry_date = parse_date(request.form.get("entry_date"), date.today())
    weight = as_float(request.form.get("weight_kg"))
    notes = request.form.get("notes", "").strip()
    if weight is None or weight <= 0 or weight > 300:
        flash("Enter a valid weight.", "error")
        return redirect(url_for("progress_analytics"))
    get_db().execute("""INSERT INTO weight_entries (user_id, entry_date, weight_kg, notes, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, entry_date) DO UPDATE SET
                        weight_kg=excluded.weight_kg, notes=excluded.notes""",
                     (session["user_id"], entry_date.isoformat(), weight, notes, utcnow()))
    get_db().commit()
    flash("Weight entry saved.", "success")
    return redirect(url_for("progress_analytics"))


@app.route("/records")
@login_required
def records():
    return render_template("records.html", records=personal_records(), workouts=workout_rows())


@app.route("/achievements")
@login_required
def achievements():
    refresh_achievements()
    earned = get_db().execute("SELECT * FROM achievements WHERE user_id = ? ORDER BY earned_at DESC",
                              (session["user_id"],)).fetchall()
    summary = analytics_summary()
    upcoming = [
        {"title": "10 Workouts", "progress": min(summary["total_workouts"], 10), "target": 10},
        {"title": "3 Day Streak", "progress": min(summary["streak"], 3), "target": 3},
        {"title": "50 Rep Session", "progress": min(personal_records()["highest_reps"]["value"], 50), "target": 50},
    ]
    return render_template("achievements.html", earned=earned, upcoming=upcoming, streak=summary["streak"])


@app.route("/workout")
@login_required
def workout():
    selected = request.args.get("exercise", "")
    return render_template("workout.html", selected_exercise=selected if selected in ALL_EXERCISES else "")


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
        refresh_achievements()
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
        if request.path.startswith("/api/food/"):
            return jsonify(error="Food photo is too large. Maximum upload size is 8 MB."), 413
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
    refresh_achievements()
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
        result = engine.process(frame, draw_landmarks=True)
        data, annotated = result if isinstance(result, tuple) else (result, None)
        payload = dict(data)
        if annotated is not None:
            ok, output = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                payload["overlay"] = "data:image/jpeg;base64," + base64.b64encode(output.tobytes()).decode("ascii")
        return jsonify(payload)
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
    if action not in {"manual", "auto", "reset", "clear", "pause", "resume"}:
        return jsonify(error="Invalid workout command."), 400
    if action == "manual" and exercise not in ALL_EXERCISES:
        return jsonify(error="Invalid exercise."), 400
    return jsonify(engine.control(action, exercise))


@app.post("/api/workout/select")
@login_required
def select_exercise():
    """Allow users to select a specific supported exercise before starting a workout."""
    payload = request.get_json(silent=True) or {}
    exercise = payload.get("exercise")
    if exercise not in ALL_EXERCISES:
        return jsonify(error="Invalid exercise."), 400
    end_engine(save=False)
    workout_id = str(uuid.uuid4())
    engine = WorkoutEngine()
    engine.control("manual", exercise)
    engines[workout_id] = engine
    session["workout_id"] = workout_id
    session["workout_started_at"] = utcnow()
    return jsonify(engine.snapshot())


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
        refresh_achievements()
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


init_db()

# The scheduler runs while the Flask process is running, so delivery does not
# depend on a browser session or an open dashboard. Database claiming makes it
# safe if more than one application process is started.
if os.environ.get("REMINDER_SCHEDULER_ENABLED", "1") == "1":
    reminder_scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
    reminder_scheduler.add_job(
        dispatch_due_reminder_emails,
        trigger="interval",
        minutes=1,
        id="email_due_workout_reminders",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    reminder_scheduler.start()


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    ssl_context = "adhoc" if os.environ.get("FLASK_USE_HTTPS") == "1" else None
    app.run(host="0.0.0.0", port=port, debug=True, ssl_context=ssl_context)
