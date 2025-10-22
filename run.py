#!/usr/bin/env python3
"""
Startup script for the Flask prediction API
"""
import os
import sys
from app import app, load_models

if __name__ == "__main__":
    # Optionally eager-load models only if env requests it
    if os.getenv('EAGER_LOAD_MODELS', 'false').lower() == 'true':
        print("Loading ML models eagerly as requested by env...")
        load_models()
    else:
        print("Skipping eager model load (EAGER_LOAD_MODELS != true)")
    
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Flask API on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)
