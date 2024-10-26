import os
import ssl
import certifi
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from pathlib import Path

# MacOS specific SSL fix
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

class BottleClassifier:
    def __init__(self):
        print("Initializing model...")
        
        # Set SSL certificate path
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # Custom weights download
        weights_path = os.path.expanduser('~/.keras/models/resnet50_weights.h5')
        if not os.path.exists(weights_path):
            print("Downloading model weights (this may take a few minutes)...")
            self._download_weights(weights_path)
        
        # Load model with local weights
        self.base_model = ResNet50(weights=weights_path)
        self.bottle_categories = {
            'water_bottle': 898,
            'wine_bottle': 907,
            'beer_bottle': 737,
            'plastic_bottle': 899,
        }
        self.bottle_threshold = 0.85
        self.pet_threshold = 0.85

    def _download_weights(self, weights_path):
        """Download weights with custom SSL handling"""
        url = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        # Download with requests
        response = requests.get(url, verify=certifi.where(), stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def preprocess_image(self, img_path):
        """Preprocess image for model input"""
        try:
            if not any(img_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                raise ValueError("Unsupported image format. Please use JPG, PNG, or BMP.")

            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def is_bottle(self, predictions):
        """Check if image contains a bottle"""
        bottle_prob = 0
        for category in self.bottle_categories.values():
            bottle_prob += predictions[0][category]
        return bottle_prob > self.bottle_threshold, bottle_prob

    def is_pet_bottle(self, predictions):
        """Check if image contains a PET bottle"""
        pet_prob = (predictions[0][self.bottle_categories['water_bottle']] + 
                   predictions[0][self.bottle_categories['plastic_bottle']])
        return pet_prob > self.pet_threshold, pet_prob

    def classify_image(self, img_path):
        """Classify image as bottle/non-bottle and PET/non-PET"""
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = self.preprocess_image(img_path)
        
        with tf.device('/CPU:0'):
            predictions = self.base_model.predict(img, verbose=0)
        
        is_bottle_result, bottle_confidence = self.is_bottle(predictions)
        is_pet_result, pet_confidence = self.is_pet_bottle(predictions) if is_bottle_result else (False, 0)
        
        return {
            'is_bottle': is_bottle_result,
            'is_pet_bottle': is_pet_result,
            'confidence_scores': {
                'bottle': float(bottle_confidence),
                'pet': float(pet_confidence)
            }
        }

def main():
    print("Starting Bottle Classification System...")
    
    try:
        # Ensure required packages are installed
        import requests
        import certifi
    except ImportError:
        print("Installing required packages...")
        os.system('pip3 install requests certifi --upgrade')
        import requests
        import certifi
    
    try:
        classifier = BottleClassifier()
        
        img_path = input("\nPlease enter the image path: ").strip().strip('"\'')
        
        print("\nAnalyzing image...")
        result = classifier.classify_image(img_path)
        
        print("\nClassification Results:")
        print(f"Is it a bottle? {'Yes' if result['is_bottle'] else 'No'}")
        print(f"Is it a PET bottle? {'Yes' if result['is_pet_bottle'] else 'No'}")
        print("\nConfidence Scores:")
        print(f"Bottle confidence: {result['confidence_scores']['bottle']:.2f}")
        print(f"PET bottle confidence: {result['confidence_scores']['pet']:.2f}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Run: /Applications/Python\\ 3.12/Install\\ Certificates.command")
        print("2. Make sure you have the latest pip: python3 -m pip install --upgrade pip")
        print("3. Install required packages: pip3 install tensorflow keras numpy certifi requests --upgrade")
        print("4. Check your internet connection")

if __name__ == "__main__":
    main()
