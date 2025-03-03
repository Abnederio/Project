#!/bin/bash

echo "Starting Flask API..."
gunicorn app:app --bind 0.0.0.0:5000 &  # Run Flask in the background

sleep 5  # Give Flask time to start

echo "Starting Streamlit..."
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
