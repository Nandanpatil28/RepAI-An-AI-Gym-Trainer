"""Regression tests for per-account scheduled reminder emails."""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Keep the module-level scheduler out of this deterministic unit test.
os.environ["REMINDER_SCHEDULER_ENABLED"] = "0"
os.environ.update({
    "SMTP_HOST": "smtp.test.invalid",
    "SMTP_USERNAME": "sender@example.com",
    "SMTP_PASSWORD": "not-a-real-password",
    "SMTP_FROM": "sender@example.com",
})

import app as fitness_app


class ReminderEmailDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = str(Path(self.tempdir.name) / "fitness-test.db")
        self.original_database = fitness_app.app.config["DATABASE"]
        fitness_app.app.config["DATABASE"] = self.database
        fitness_app.init_db()
        self.sent_to = []
        self.original_sender = fitness_app.send_reminder_email
        fitness_app.send_reminder_email = self.fake_sender

        db = sqlite3.connect(self.database)
        db.executemany(
            "INSERT INTO users (name, email, password_hash, created_at, active) VALUES (?, ?, ?, ?, 1)",
            [
                ("Asha", "asha@gmail.com", "hash", "2035-01-01T00:00:00+00:00"),
                ("Ravi", "ravi@outlook.com", "hash", "2035-01-01T00:00:00+00:00"),
            ],
        )
        db.executemany(
            "INSERT INTO user_profiles (user_id, notify_email, updated_at) VALUES (?, 1, ?)",
            [(1, "2035-01-01T00:00:00+00:00"), (2, "2035-01-01T00:00:00+00:00")],
        )
        db.executemany("""
            INSERT INTO workout_reminders
            (user_id, title, reminder_date, reminder_time, timezone, enabled, created_at, updated_at)
            VALUES (?, ?, '2035-01-01', '09:00', 'UTC', 1, ?, ?)
        """, [
            (1, "Morning mobility", "2035-01-01T00:00:00+00:00", "2035-01-01T00:00:00+00:00"),
            (2, "Strength session", "2035-01-01T00:00:00+00:00", "2035-01-01T00:00:00+00:00"),
        ])
        db.commit()
        db.close()

    def tearDown(self):
        fitness_app.send_reminder_email = self.original_sender
        fitness_app.app.config["DATABASE"] = self.original_database
        self.tempdir.cleanup()

    def fake_sender(self, row):
        self.sent_to.append((row["user_id"], row["email"], row["title"]))
        return True, None

    def test_each_user_receives_only_their_own_due_reminder_once(self):
        now = datetime(2035, 1, 1, 9, 1, tzinfo=timezone.utc)
        self.assertEqual(fitness_app.dispatch_due_reminder_emails(now), 2)
        self.assertEqual(self.sent_to, [
            (1, "asha@gmail.com", "Morning mobility"),
            (2, "ravi@outlook.com", "Strength session"),
        ])
        # A subsequent scheduler poll cannot send either reminder again.
        self.assertEqual(fitness_app.dispatch_due_reminder_emails(now), 0)

        db = sqlite3.connect(self.database)
        statuses = db.execute(
            "SELECT user_id, status FROM reminder_email_deliveries ORDER BY user_id"
        ).fetchall()
        db.close()
        self.assertEqual(statuses, [(1, "sent"), (2, "sent")])

    def test_invalid_account_email_is_tracked_without_attempting_delivery(self):
        db = sqlite3.connect(self.database)
        db.execute("UPDATE workout_reminders SET enabled = 0")
        db.execute(
            "INSERT INTO users (name, email, password_hash, created_at, active) VALUES (?, ?, ?, ?, 1)",
            ("Legacy", "not-an-email", "hash", "2035-01-01T00:00:00+00:00"),
        )
        db.execute("INSERT INTO user_profiles (user_id, notify_email, updated_at) VALUES (3, 1, ?)",
                   ("2035-01-01T00:00:00+00:00",))
        db.execute("""INSERT INTO workout_reminders
            (user_id, title, reminder_date, reminder_time, timezone, enabled, created_at, updated_at)
            VALUES (3, 'Legacy reminder', '2035-01-01', '09:00', 'UTC', 1, ?, ?)""",
                   ("2035-01-01T00:00:00+00:00", "2035-01-01T00:00:00+00:00"))
        db.commit()
        db.close()
        fitness_app.send_reminder_email = self.original_sender

        self.assertEqual(fitness_app.dispatch_due_reminder_emails(datetime(2035, 1, 1, 9, 1, tzinfo=timezone.utc)), 0)
        db = sqlite3.connect(self.database)
        status = db.execute("SELECT status, failure_reason FROM reminder_email_deliveries").fetchone()
        db.close()
        self.assertEqual(status, ("skipped_invalid_recipient", "invalid_recipient"))


if __name__ == "__main__":
    unittest.main()
