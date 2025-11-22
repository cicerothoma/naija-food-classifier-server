import os
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import joblib
from typing import Dict, Any

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ModelWrapper:
    """
    Wrapper class for the exported model that includes inference functionality.
    """
    
    def __init__(self, model: torch.nn.Module, label_map: Dict[str, int], 
                 model_name: str, metadata: Dict[str, Any]):
        self.model = model
        self.label_map = label_map
        self.model_name = model_name
        self.metadata = metadata
        
        # Create reverse mapping for predictions
        self.idx_to_label = {v: k for k, v in label_map.items()}
        self.class_names = list(label_map.keys())
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions on input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
            
            return {
                'predicted_class': self.idx_to_label[predicted_idx],
                'predicted_idx': predicted_idx,
                'confidence': confidence,
                'all_probabilities': {
                    self.idx_to_label[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def get_class_names(self):
        """Get list of all class names."""
        return self.class_names
    
    def get_metadata(self):
        """Get model metadata."""
        return self.metadata

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
DISH_LIST_PATH = os.path.join(os.path.dirname(__file__), 'dish_list.txt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
model = None
class_names = []
transform = None

def initialize_app():
    """Initialize the application components"""
    print("Initializing Naija Food Classification Server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    
    # Initialize transforms
    setup_transforms()
    
    # Try to load class names (don't exit if it fails in production)
    classes_loaded = load_class_names()
    if not classes_loaded:
        print("WARNING: Failed to load class names. API will have limited functionality.")
    
    # Try to load model (don't exit if it fails in production)
    model_loaded = load_model()
    if not model_loaded:
        print("WARNING: Failed to load model. Prediction endpoints will not work.")
    
    if classes_loaded and model_loaded:
        print(f"✅ Server ready with {len(class_names)} food classes!")
    else:
        print("⚠️  Server started with limited functionality:")
        print(f"   - Classes loaded: {classes_loaded}")
        print(f"   - Model loaded: {model_loaded}")
    
    print("Available endpoints:")
    print("  GET  / - Health check")
    print("  GET  /classes - List all food classes")
    print("  POST /predict - Predict from image file")
    print("  POST /predict_base64 - Predict from base64 image")
    
    return classes_loaded, model_loaded

# Initialize when module is imported (works with both direct run and Gunicorn)
initialize_app()

def load_class_names():
    """Load class names from dish_list.txt"""
    global class_names
    try:
        print(f"Looking for dish list at: {DISH_LIST_PATH}")
        print(f"File exists: {os.path.exists(DISH_LIST_PATH)}")
        
        if not os.path.exists(DISH_LIST_PATH):
            print(f"Dish list file not found at {DISH_LIST_PATH}")
            # List files in current directory for debugging
            current_dir = os.path.dirname(__file__) or '.'
            files = os.listdir(current_dir)
            print(f"Files in {current_dir}: {files}")
            return False
        
        with open(DISH_LIST_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(class_names)} food classes")
        return True
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return False

def load_model():
    """Load the PyTorch model from joblib file"""
    global model, class_names
    try:
        print(f"Looking for model at: {MODEL_PATH}")
        print(f"File exists: {os.path.exists(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}")
            # List files in current directory for debugging
            current_dir = os.path.dirname(__file__) or '.'
            files = os.listdir(current_dir)
            print(f"Files in {current_dir}: {files}")
            return False
        
        # Load the model wrapper using joblib
        print("Loading PyTorch model from joblib file...")
        model_wrapper = joblib.load(MODEL_PATH)
        
        # Extract the actual model from the wrapper
        model = model_wrapper.model
        
        # Use class names from the model wrapper if available
        if hasattr(model_wrapper, 'class_names'):
            class_names = model_wrapper.class_names
            print(f"Using class names from model wrapper: {len(class_names)} classes")
        
        # Move to device and set to evaluation mode
        model.to(DEVICE)
        model.eval()
        
        print("PyTorch model loaded successfully from joblib file")
        print(f"Model type: {type(model)}")
        print(f"Model wrapper type: {type(model_wrapper)}")
        return True
        
    except Exception as e:
        print(f"Error loading model from joblib: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def setup_transforms():
    """Setup image transformations for preprocessing"""
    global transform
    # Standard transformations for image classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 (common for many models)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def preprocess_image(image):
    """Preprocess image for PyTorch model prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # PyTorch model - use tensor preprocessing
        image_tensor = transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(DEVICE)
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(input_data):
    """Make prediction on preprocessed image using PyTorch model"""
    try:
        # PyTorch model prediction
        with torch.no_grad():
            outputs = model(input_data)
            
            # Handle different output formats
            if len(outputs.shape) == 2:  # Batch dimension exists
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            else:
                probabilities = torch.nn.functional.softmax(outputs, dim=0)
            
            # Get top 6 predictions
            top6_prob, top6_indices = torch.topk(probabilities, 6)
            
            predictions = []
            for i in range(6):
                class_idx = top6_indices[i].item()
                confidence = top6_prob[i].item()
                
                if class_idx < len(class_names):
                    predictions.append({
                        'class': class_names[class_idx],
                        'confidence': float(confidence),
                        'percentage': f"{confidence * 100:.2f}%"
                    })
            
            return predictions
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def health_check():
  """Health check endpoint"""
  return jsonify({
    'status': 'healthy',
    'message': 'Naija Food Classification API is running!',
    'model_loaded': model is not None,
    'classes_loaded': len(class_names) > 0,
    'total_classes': len(class_names)
  })

@app.route('/classes', methods=['GET'])
def get_classes():
  """Get list of all food classes"""
  return jsonify({
    'classes': class_names,
    'total_classes': len(class_names)
  })

@app.route('/predict', methods=['POST'])
def predict():
  """Predict food class from uploaded image"""
  try:
    # Check if model is loaded
    if model is None:
      return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if image is provided
    if 'image' not in request.files:
      return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
      return jsonify({'error': 'No image file selected'}), 400
    
    # Read and process image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess image
    processed_data = preprocess_image(image)
    if processed_data is None:
      return jsonify({'error': 'Failed to preprocess image'}), 500
    
    # Make prediction
    predictions = predict_image(processed_data)
    if predictions is None:
      return jsonify({'error': 'Failed to make prediction'}), 500
    
    return jsonify({
      'success': True,
      'predictions': predictions,
      'top_prediction': predictions[0] if predictions else None
    })
      
  except Exception as e:
    return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
  """Predict food class from base64 encoded image"""
  try:
    # Check if model is loaded
    if model is None:
      return jsonify({'error': 'Model not loaded'}), 500
    
    # Get JSON data
    data = request.get_json()
    if not data or 'image' not in data:
      return jsonify({'error': 'No base64 image data provided'}), 400
    
    # Decode base64 image
    image_data = data['image']
    # Remove data URL prefix if present
    if ',' in image_data:
      image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess image
    processed_data = preprocess_image(image)
    if processed_data is None:
      return jsonify({'error': 'Failed to preprocess image'}), 500
    
    # Make prediction
    predictions = predict_image(processed_data)
    if predictions is None:
      return jsonify({'error': 'Failed to make prediction'}), 500
    
    return jsonify({
      'success': True,
      'predictions': predictions,
      'top_prediction': predictions[0] if predictions else None
    })
      
  except Exception as e:
    return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
  """Handle file too large error"""
  return jsonify({'error': 'File too large'}), 413

@app.errorhandler(415)
def unsupported_media_type(e):
  """Handle unsupported media type error"""
  return jsonify({'error': 'Unsupported media type'}), 415

if __name__ == '__main__':
    # This only runs when called directly with 'python app.py'
    # Gunicorn will import the app without executing this block
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        print("Running in DEBUG mode - not for production!")
    
    print(f"Starting Flask development server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)