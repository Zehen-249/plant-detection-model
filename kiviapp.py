import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
import torch
import numpy as np
from PIL import Image as PILImage
import io

kivy.require('2.1.0')  # Replace with your Kivy version

class PlantLeafDetectionApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Load the model (quantized or regular)
        self.model = torch.jit.load('plant_leaf_detection_mobile_quantized.pt')  # Use the quantized model or regular
        self.model.eval()  # Set the model to evaluation mode
        
        # Button to trigger prediction
        self.predict_button = Button(text="Detect Plant Leaf", on_press=self.predict)
        self.layout.add_widget(self.predict_button)
        
        # Image widget to display the result (just for example)
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)
        
        return self.layout
    
    def preprocess_image(self, img_path):
        # Open the image using PIL and convert to RGB
        image = PILImage.open(img_path).convert('RGB')
        # Resize to the model's expected input size (640x640 for YOLO)
        image = image.resize((640, 640))
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32)
        image_np /= 255.0  # Normalize to [0, 1]
        # Add batch dimension
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 640, 640]
        return image_tensor
    
    def predict(self, instance):
        # Example image path (replace with the actual image path)
        img_path = 'path_to_leaf_image.jpg'
        
        # Preprocess the image
        image_tensor = self.preprocess_image(img_path)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Process the output (this depends on the model)
        # For example, if it's a detection task, you might extract the bounding boxes and classes
        # Here, just printing the raw output for now
        print("Prediction Output:", output)

        # You can also visualize the result if needed
        # For simplicity, using the image as an example
        self.image_widget.source = img_path  # Update the image widget with the predicted image
        self.image_widget.reload()

if __name__ == '__main__':
    PlantLeafDetectionApp().run()
