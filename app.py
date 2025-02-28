from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_bcrypt import Bcrypt
import pandas as pd
import os

app = Flask(__name__)

# Configurations
app.config["JWT_SECRET_KEY"] = "your_secret_key"  # Change this to a secure key
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
CSV_FILE = "data.csv"

# Ensure CSV file exists with correct columns
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["id", "username", "password", "email"])
    df.to_csv(CSV_FILE, index=False)

# Read CSV file
def load_csv():
    return pd.read_csv(CSV_FILE, dtype={"id": int, "username": str, "password": str, "email": str})

# Write to CSV file
def save_csv(df):
    df.to_csv(CSV_FILE, index=False)

# Ensure passwords are hashed in CSV
def hash_passwords():
    df = load_csv()
    if not df.empty and not df["password"].str.startswith("$2b$").all():  # Check if passwords are already hashed
        df["password"] = df["password"].apply(lambda pwd: bcrypt.generate_password_hash(pwd).decode("utf-8"))
        save_csv(df)

hash_passwords()  # Ensure password security at startup

# Login route
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    df = load_csv()

    user = df[df["username"] == data["username"]]
    if user.empty or not bcrypt.check_password_hash(user.iloc[0]["password"], data["password"]):
        return jsonify({"message": "Invalid username or password"}), 401

    token = create_access_token(identity=data["username"])
    return jsonify(access_token=token)

# Get all admins (Protected)
@app.route("/admins", methods=["GET"])
@jwt_required()
def get_admins():
    df = load_csv()
    return jsonify(df.drop(columns=["password"]).to_dict(orient="records"))

# Add a new admin (Protected)
@app.route("/admins", methods=["POST"])
@jwt_required()
def add_admin():
    data = request.json
    df = load_csv()

    if int(data["id"]) in df["id"].values:
        return jsonify({"message": "ID already exists"}), 400

    data["password"] = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    new_admin = pd.DataFrame([data])

    df = pd.concat([df, new_admin], ignore_index=True)
    save_csv(df)

    return jsonify({"message": "Admin added successfully"})

# Update an existing admin (Protected)
@app.route("/admins/<int:id>", methods=["PUT"])
@jwt_required()
def update_admin(id):
    data = request.json
    df = load_csv()

    if id not in df["id"].values:
        return jsonify({"message": "Admin not found"}), 404

    index = df[df["id"] == id].index[0]
    if "username" in data:
        df.at[index, "username"] = data["username"]
    if "email" in data:
        df.at[index, "email"] = data["email"]

    save_csv(df)
    return jsonify({"message": "Admin updated successfully"})

# Delete an admin (Protected)
@app.route("/admins/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_admin(id):
    df = load_csv()

    if id not in df["id"].values:
        return jsonify({"message": "Admin not found"}), 404

    df = df[df["id"] != id]
    save_csv(df)

    return jsonify({"message": "Admin deleted successfully"})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

