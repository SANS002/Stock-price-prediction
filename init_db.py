from app import app, db, User
with app.app_context():
    # Query all users
    users = User.query.all()
    for user in users:
        print(f"User: {user.name}, Email: {user.email}") 