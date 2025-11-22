#!/usr/bin/env python3
"""
Simple test script for the Naija Food Classification API
"""

import os
import requests
import base64
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_classes():
    """Test getting the list of food classes"""
    print("\nTesting get classes...")
    try:
        response = requests.get(f"{BASE_URL}/classes")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total classes: {data.get('total_classes', 0)}")
        print(f"First 5 classes: {data.get('classes', [])[:5]}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_with_file(image_path):
    """Test prediction with image file"""
    print(f"\nTesting prediction with file: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success', False)}")
            if 'top_prediction' in data and data['top_prediction']:
                pred = data['top_prediction']
                print(f"Top prediction: {pred['class']} ({pred['percentage']})")
            
            if 'predictions' in data:
                print("All predictions:")
                for i, pred in enumerate(data['predictions'], 1):
                    print(f"  {i}. {pred['class']}: {pred['percentage']}")
        else:
            print(f"Error: {response.json()}")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_with_base64(image_path):
    """Test prediction with base64 encoded image"""
    print(f"\nTesting prediction with base64: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {'image': image_data}
        response = requests.post(f"{BASE_URL}/predict_base64", json=payload)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success', False)}")
            if 'top_prediction' in data and data['top_prediction']:
                pred = data['top_prediction']
                print(f"Top prediction: {pred['class']} ({pred['percentage']})")
        else:
            print(f"Error: {response.json()}")
        return response.status_code == 200
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Naija Food Classification API Test Script")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_check()
    classes_ok = test_get_classes()
    
    # Test image prediction (you'll need to provide an actual image file)
    test_image = os.path.join(os.path.dirname(__file__), 'goat_meat_peppersoup.webp')
     # Replace with actual image path
    
    # print(f"\nTo test image prediction, place an image at: {test_image}")
    # print("Then run this script again.")
    
    # Uncomment the lines below when you have a test image
    predict_file_ok = test_predict_with_file(test_image)
    # predict_b64_ok = test_predict_with_base64(test_image)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Health Check: {'✅' if health_ok else '❌'}")
    print(f"Get Classes: {'✅' if classes_ok else '❌'}")
    print("Image Prediction: 🔄 (Add test image to run)")