#!/usr/bin/env python3
"""
Quick test script for TensorFlow Chessbot API
Tests that the API server can start and process a simple request
"""

import requests
import sys
import time
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8002"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server. Is it running?")
        print(f"  Start it with: cd tensorflow_chessbot && python3 api_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_analyze_with_sample():
    """Test the analyze endpoint with a sample image"""
    print("\nTesting analyze endpoint...")
    
    # Look for a test image
    test_image_paths = [
        "/Users/rocha/Documents/IA/images/test-images",
        "/Users/rocha/Documents/IA/images/lichess_test_images",
        "/Users/rocha/Documents/IA/a/LiveChess2FEN/test",
        "/Users/rocha/Documents/IA/a/LiveChess2FEN/realimages"
    ]
    
    test_image = None
    for path_str in test_image_paths:
        path = Path(path_str)
        if path.exists():
            # Look for jpg or png files
            images = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        print("⚠ No test image found. Skipping analyze test.")
        print(f"  Searched in: {test_image_paths}")
        return False
    
    print(f"  Using test image: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            data = {'white_position': 'bottom'}
            
            print("  Sending request (this may take 10-20 seconds for first request)...")
            response = requests.post(
                f"{API_URL}/analyze",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Analysis successful!")
            print(f"  FEN: {result.get('fen', 'N/A')}")
            print(f"  Pieces detected: {result.get('pieces_detected', 'N/A')}")
            print(f"  Certainty: {result.get('certainty', 'N/A')}%")
            return True
        else:
            print(f"✗ Analysis failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (>60s)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("TensorFlow Chessbot API Test")
    print("=" * 60)
    print()
    
    # Test health
    if not test_health():
        print("\n❌ API server is not running or not responding")
        sys.exit(1)
    
    # Wait a moment
    time.sleep(1)
    
    # Test analyze (optional, can take time)
    print()
    response = input("Do you want to test image analysis? (y/n): ")
    if response.lower() == 'y':
        test_analyze_with_sample()
    
    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
