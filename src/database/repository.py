"""Repository layer for SafeGate AI SQLite persistence."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "safegate.db"


class GateRepository:
    """Encapsulates SQLite persistence for resident plates and gate access logs."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Connect to the SQLite database, initialize the cursor, and ensure tables exist."""
        if str(db_path) == ":memory:":
            self.db_path: str | Path = ":memory:"
            self.conn = sqlite3.connect(":memory:")
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)

        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self) -> None:
        """Create the Residents and Access_Logs tables if they do not already exist."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Residents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                plate_number TEXT UNIQUE NOT NULL,
                apartment_num TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Access_Logs (
                id INTEGER PRIMARY KEY,
                plate_number TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL
            )
            """
        )

        self.conn.commit()

    def log_access(self, plate_number: str, status: str) -> None:
        """Insert an access attempt record into Access_Logs and commit."""
        self.cursor.execute(
            "INSERT INTO Access_Logs (plate_number, status) VALUES (?, ?)",
            (plate_number, status),
        )
        self.conn.commit()

    def check_and_log_vehicle(self, plate_number: str) -> bool:
        """Check if a plate number belongs to a resident, log the attempt, and return access status."""
        self.cursor.execute(
            "SELECT 1 FROM Residents WHERE plate_number = ?",
            (plate_number,),
        )
        match = self.cursor.fetchone()

        if match:
            status = "Authorized Resident"
            is_allowed = True
        else:
            status = "Unknown/Visitor"
            is_allowed = False

        # Log every access attempt regardless of authorization outcome
        self.log_access(plate_number, status)

        return is_allowed

    def get_recent_logs(self, limit: int = 10) -> list[tuple[int, str, str, str]]:
        """Return the most recent access log entries ordered newest-first."""
        self.cursor.execute(
            "SELECT id, plate_number, timestamp, status FROM Access_Logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return self.cursor.fetchall()

    def seed_dummy_data(self) -> None:
        """Insert sample Egyptian resident records into the Residents table."""
        dummy_residents = [
            ("Mohamed Zain", "ا ب ج ١٢٣", "A-10"),
            ("Samar", "س م ر ٤٥٦٧", "B-12"),
            ("Habiba", "ح ب ي ٧٨٩", "C-15"),
        ]

        # INSERT OR IGNORE prevents errors on repeated runs due to the UNIQUE plate_number constraint
        self.cursor.executemany(
            """
            INSERT OR IGNORE INTO Residents (name, plate_number, apartment_num)
            VALUES (?, ?, ?)
            """,
            dummy_residents,
        )
        self.conn.commit()
        print("Dummy residents added.")

    def verify_setup(self) -> list[str]:
        """Query Residents, print count and seeded plates, and return the plate list."""
        self.cursor.execute(
            "SELECT id, name, plate_number, apartment_num FROM Residents ORDER BY id"
        )
        residents = self.cursor.fetchall()
        plates = [row[2] for row in residents]

        print(f"Database path: {self.db_path}")
        print(f"Total seeded residents: {len(residents)}")
        print("Seeded Egyptian plates:")
        for _, name, plate_number, apartment_num in residents:
            print(f"- {plate_number} ({name}, Apt {apartment_num})")

        if residents:
            print("Phase 1 database setup is 100% operational.")
        else:
            print("Warning: No residents found in the Residents table.")

        return plates

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> GateRepository:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    with GateRepository() as repo:
        repo.seed_dummy_data()
        repo.verify_setup()

