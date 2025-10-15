#!/usr/bin/env python3
"""
Startup script for the Flask prediction API
"""
import os
import sys
from app import app, load_models

if __name__ == "__main__":
    # Load models on startup
    print("Loading ML models...")
    load_models()
    
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Flask API on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)
