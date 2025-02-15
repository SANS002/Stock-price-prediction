from app import app, db, DB_FILE
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Database path: {DB_FILE}")
print(f"Database directory exists: {DB_FILE.parent.exists()}")
print(f"Database file exists: {DB_FILE.exists()}")

with app.app_context():
    try:
        db.create_all()
        print("Database created successfully!")
    except Exception as e:
        print(f"Error creating database: {e}") 