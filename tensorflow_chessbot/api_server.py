#!/usr/bin/env python3
"""
Flask server for TensorFlow Chessbot
Provides an endpoint compatible with the chess-fen-frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import numpy as np
from PIL import Image
import io

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow_chessbot
from helper_functions import shortenFEN
import chessboard_finder
import helper_image_loading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global predictor instance (loaded once at startup)
predictor = None

def initialize_model():
    """Initialize the TensorFlow model"""
    global predictor
    print("Loading TensorFlow Chessbot model...")
    frozen_graph_path = os.path.join(
        os.path.dirname(__file__), 
        'saved_models/frozen_graph.pb'
    )
    predictor = tensorflow_chessbot.ChessboardPredictor(frozen_graph_path)
    print("Model loaded successfully!")

@app.route("/")
def root():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "TensorFlow Chessbot API",
        "version": "1.0.0"
    })

@app.route("/analyze", methods=["POST"])
def analyze_board():
    """
    Analyze a chess board image and return FEN notation
    
    Form data:
        file: The uploaded image file
        white_position: Position of white pieces (bottom, top, left, right)
    
    Returns:
        JSON with success status, FEN string, and pieces detected
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        white_position = request.form.get('white_position', 'bottom')
        
        # Read uploaded file
        contents = file.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize if needed
        img_array = helper_image_loading.resizeAsNeeded(img_array)
        
        if img_array is None:
            return jsonify({
                "success": False,
                "error": "Image too large to process"
            }), 400
        
        # Find chessboard and extract tiles
        tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img_array)
        
        if tiles is None:
            return jsonify({
                "success": False,
                "error": "Could not find a chessboard in the image"
            }), 400
        
        # Make prediction
        fen, tile_certainties = predictor.getPrediction(tiles)
        
        if fen is None:
            return jsonify({
                "success": False,
                "error": "Could not predict FEN from board"
            }), 400
        
        # Shorten FEN (convert 111 to 3, etc.)
        short_fen = shortenFEN(fen)
        
        # Handle white position rotation
        if white_position == "top":
            short_fen = flip_fen_vertical(short_fen)
        elif white_position == "left":
            short_fen = rotate_fen_90_cw(short_fen)
        elif white_position == "right":
            short_fen = rotate_fen_90_ccw(short_fen)
        
        # Add standard FEN suffix
        full_fen = f"{short_fen} w KQkq - 0 1"
        
        # Calculate certainty and pieces detected
        certainty = float(tile_certainties.min())
        pieces_detected = count_pieces_in_fen(short_fen)
        
        return jsonify({
            "success": True,
            "fen": full_fen,
            "pieces_detected": pieces_detected,
            "certainty": round(certainty * 100, 2),
            "message": f"Board analyzed successfully with {certainty*100:.1f}% certainty"
        })
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

def flip_fen_vertical(fen):
    """Flip FEN vertically (mirror top to bottom)"""
    ranks = fen.split('/')
    return '/'.join(reversed(ranks))

def rotate_fen_90_cw(fen):
    """Rotate FEN 90 degrees clockwise"""
    ranks = fen.split('/')
    board = []
    for rank in ranks:
        board.append(expand_fen_rank(rank))
    
    # Transpose and reverse for 90° CW rotation
    rotated = []
    for col in range(8):
        new_rank = []
        for row in range(7, -1, -1):
            new_rank.append(board[row][col])
        rotated.append(''.join(new_rank))
    
    return '/'.join([compress_fen_rank(rank) for rank in rotated])

def rotate_fen_90_ccw(fen):
    """Rotate FEN 90 degrees counter-clockwise"""
    ranks = fen.split('/')
    board = []
    for rank in ranks:
        board.append(expand_fen_rank(rank))
    
    # Transpose and reverse for 90° CCW rotation
    rotated = []
    for col in range(7, -1, -1):
        new_rank = []
        for row in range(8):
            new_rank.append(board[row][col])
        rotated.append(''.join(new_rank))
    
    return '/'.join([compress_fen_rank(rank) for rank in rotated])

def expand_fen_rank(rank):
    """Expand compressed FEN rank (3 -> 111)"""
    expanded = []
    for char in rank:
        if char.isdigit():
            expanded.extend(['1'] * int(char))
        else:
            expanded.append(char)
    return expanded

def compress_fen_rank(rank):
    """Compress FEN rank (111 -> 3)"""
    compressed = []
    empty_count = 0
    for char in rank:
        if char == '1':
            empty_count += 1
        else:
            if empty_count > 0:
                compressed.append(str(empty_count))
                empty_count = 0
            compressed.append(char)
    if empty_count > 0:
        compressed.append(str(empty_count))
    return ''.join(compressed)

def count_pieces_in_fen(fen):
    """Count number of pieces in FEN string"""
    pieces = 0
    for char in fen:
        if char.isalpha():
            pieces += 1
    return pieces

if __name__ == "__main__":
    # Initialize model before starting server
    initialize_model()
    
    # Run server on port 8002 (to avoid conflicts with other backends)
    print("Starting Flask server on http://0.0.0.0:8003")
    app.run(host="0.0.0.0", port=8002, debug=False)
