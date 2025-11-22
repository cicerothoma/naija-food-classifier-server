# Naija Food Classification Server

A Flask-based REST API for classifying Nigerian food dishes using PyTorch image classification.

## Features

- **Image Classification**: Identifies 18 different Nigerian food dishes
- **Multiple Input Formats**: Supports both file upload and base64 encoded images
- **Top-3 Predictions**: Returns the top 3 most likely food classes with confidence scores
- **CORS Enabled**: Ready for frontend integration
- **Health Check**: Monitor server status and model loading

## Supported Food Classes

The model can identify the following Nigerian dishes:
- Jollof Rice
- Egusi Soup
- Moi Moi
- Akara
- Suya
- Efo Riro
- Okra Soup
- Ofada Rice
- Pounded Yam
- Banga Soup
- Pepper Soup
- Nkwobi
- Amala
- Ewedu Soup
- Ogbono Soup
- Yam Porridge
- Puff Puff
- Chin Chin

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model File**: Make sure your trained PyTorch model file `best.pt` is in the project root directory.

3. **Run the Server**:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
**GET** `/`

Returns server status and model information.

```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Naija Food Classification API is running!",
  "model_loaded": true,
  "classes_loaded": true,
  "total_classes": 18
}
```

### 2. Get Food Classes
**GET** `/classes`

Returns the list of all supported food classes.

```bash
curl http://localhost:5000/classes
```

**Response:**
```json
{
  "classes": ["Jollof Rice", "Egusi Soup", "Moi Moi", ...],
  "total_classes": 18
}
```

### 3. Predict from Image File
**POST** `/predict`

Upload an image file for classification.

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "class": "Jollof Rice",
      "confidence": 0.89,
      "percentage": "89.00%"
    },
    {
      "class": "Ofada Rice",
      "confidence": 0.08,
      "percentage": "8.00%"
    },
    {
      "class": "Yam Porridge",
      "confidence": 0.03,
      "percentage": "3.00%"
    }
  ],
  "top_prediction": {
    "class": "Jollof Rice",
    "confidence": 0.89,
    "percentage": "89.00%"
  }
}
```

### 4. Predict from Base64 Image
**POST** `/predict_base64`

Send a base64 encoded image for classification.

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "iVBORw0KGgoAAAANSUhEUgAA..."}' \
  http://localhost:5000/predict_base64
```

**Response:** Same format as file upload endpoint.

## Testing

Use the included test script to verify the API:

```bash
python test_api.py
```

For testing with actual images:
1. Place a test image in the project directory
2. Update the `test_image` variable in `test_api.py`
3. Run the test script

## Model Configuration

The Flask app expects your PyTorch model to:
- Be saved as `best.pt` in the project root
- Accept input tensors of shape `(batch_size, 3, 224, 224)`
- Use standard ImageNet normalization
- Output class logits for 18 food classes

If your model was trained with different parameters, you may need to adjust:
- Image resize dimensions in `setup_transforms()`
- Normalization values
- Model loading logic

## Error Handling

The API handles common errors:
- **413**: File too large
- **415**: Unsupported media type
- **400**: Missing or invalid input
- **500**: Server errors (model loading, prediction failures)

## Development

To run in development mode with auto-reload:
```bash
python app.py
```

For production deployment, consider using:
- Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- Docker containerization
- Load balancing for multiple instances

## Notes

- The server uses CPU by default but will automatically use GPU if available
- Image preprocessing includes resizing to 224x224 and ImageNet normalization
- The model returns the top 6 predictions for each image
- CORS is enabled for all origins (adjust for production security)

## Nigerian Food Classification Model 🍲🇳🇬

An EfficientNetB4 feature extractor computer vision model to classify 18 classes of Nigerian food.

## About This Model
This model was trained to classify images of Nigerian food into 18 different categories using transfer learning with the EfficientNetB0 architecture. The model leverages pre-trained weights and fine-tunes the classifier layer to adapt to the specific task of Nigerian food classification.
The model achieved a high accuracy on the validation set, demonstrating its effectiveness in recognizing various Nigerian dishes.
## Model Architecture
The model uses the EfficientNetB4 architecture as a feature extractor. The classifier layer was replaced to accommodate the 18 classes of Nigerian food. The model was trained using cross-entropy loss and the Adam optimizer. Click [here](https://huggingface.co/cicerothoma/nigerian_food_classification) to download the trained model.
Key Details:
- **Model Architecture:** EfficientNet-B4
- **Pre-trained Weights:** ImageNet
- **Fine-tuning:** Classifier layer replaced for 18 Nigerian food classes
- **Training Framework:** PyTorch
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Validation Accuracy:** ~72.2%

## Dataset
The dataset used for training and validation consists of images of 18 different Nigerian food classes. The images were collected from various sources publicly available on the internet. Click [here](https://huggingface.co/datasets/cicerothoma/nigeria_food) to access the dataset.

## Performance
On the validation set, the EfficientNet-B4 variant achieved an overall accuracy of approximately 72.2%. Performance is strong on classes such as Moi Moi, Akara, Jollof Rice, Puff Puff, and Suya, with balanced precision and recall. However, recall remains lower for visually similar viscous soups like Okra, Ogbono, and Banga, indicating class-specific confusion. Macro F1 is around 0.715, suggesting reasonably balanced performance across classes, with room for targeted improvements.

## Food Classes
The model can classify the following Nigerian food classes:
- Jollof Rice
- Egusi Soup
- Moi Moi
- Akara
- Suya
- Efo Riro
- Okra Soup
- Ofada Rice
- Pounded Yam
- Banga Soup
- Pepper Soup
- Nkwobi
- Amala
- Ewedu Soup
- Ogbono Soup
- Yam Porridge
- Puff Puff
- Chin Chin

## Limitations
While the model performs well on the validation set, it may still face challenges with images that have poor lighting, occlusions, or unusual angles. Additionally, the model's performance may vary when tested on images from different sources or with different quality.

### EfficientNet-B4-specific limitations:
- Computationally Intensive: EfficientNet-B4 is more computationally intensive compared to smaller models, which may lead to longer inference times on devices with limited resources.
- Memory Usage: The model requires more memory, which could be a limitation for deployment on edge devices with constrained memory capacity.
- Class-specific recall: Without calibration, B4 may under-predict visually similar viscous soups (e.g., Okra, Ogbono, Banga), leading to lower recall unless class-wise thresholds, focal loss, or targeted augmentation are applied.
- Data Bias: The model's performance is highly dependent on the quality and diversity of the training dataset. If the dataset lacks representation of certain food classes or variations, the model may struggle to generalize well to unseen images.
- Domain Shift: The model may not perform well on images that differ significantly from the training data in terms of style, background, or context.

## Future Work
Future improvements could include expanding the dataset with more diverse images, experimenting with different architectures, and fine-tuning hyperparameters to further enhance model performance.

## Acknowledgements
This model was developed as part of a project to promote Nigerian cuisine and culture through technology. Special thanks to the open-source community for providing the tools and resources necessary for building this model.
