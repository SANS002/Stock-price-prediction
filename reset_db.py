from app import app, db
import os
from pathlib import Path

# Get the database file path
DB_DIR = Path(__file__).resolve().parent / 'database'
DB_FILE = DB_DIR / 'users.db'

def reset_database():
    # Remove the existing database file if it exists
    if DB_FILE.exists():
        os.remove(DB_FILE)
        print(f"Removed existing database: {DB_FILE}")

    # Create the database directory if it doesn't exist
    DB_DIR.mkdir(exist_ok=True)
    
    # Create new database with tables
    with app.app_context():
        db.create_all()
        print("Database created successfully with all tables!")

if __name__ == "__main__":
    reset_database() 