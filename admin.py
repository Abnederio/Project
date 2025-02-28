pip install fastapi uvicorn pandas
from fastapi import FastAPI, HTTPException
import pandas as pd
import os

app = FastAPI()
CSV_FILE = "data.csv"

# Ensure the CSV file exists
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["id", "username", "email"])
    df.to_csv(CSV_FILE, index=False)


# Read CSV Data
@app.get("/admins/")
def read_admins():
    df = pd.read_csv(CSV_FILE)
    return df.to_dict(orient="records")


# Create Admin
@app.post("/admins/")
def create_admin(id: int, username: str, email: str):
    df = pd.read_csv(CSV_FILE)
    
    if id in df["id"].values:
        raise HTTPException(status_code=400, detail="ID already exists")
    
    new_row = {"id": id, "username": username, "email": email}
    df = df.append(new_row, ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    
    return {"message": "Admin added successfully"}


# Update Admin
@app.put("/admins/{id}")
def update_admin(id: int, username: str = None, email: str = None):
    df = pd.read_csv(CSV_FILE)

    if id not in df["id"].values:
        raise HTTPException(status_code=404, detail="Admin not found")

    index = df[df["id"] == id].index[0]
    if username:
        df.at[index, "username"] = username
    if email:
        df.at[index, "email"] = email

    df.to_csv(CSV_FILE, index=False)
    return {"message": "Admin updated successfully"}


# Delete Admin
@app.delete("/admins/{id}")
def delete_admin(id: int):
    df = pd.read_csv(CSV_FILE)

    if id not in df["id"].values:
        raise HTTPException(status_code=404, detail="Admin not found")

    df = df[df["id"] != id]  # Remove the row
    df.to_csv(CSV_FILE, index=False)

    return {"message": "Admin deleted successfully"}
