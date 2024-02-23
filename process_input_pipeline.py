from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import cv2

# Define custom transformers for image processing steps
class ImageResizer(BaseEstimator, TransformerMixin):
    def __init__(self, target_size=(299, 299)):
        self.target_size = target_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        resized_images = [cv2.resize(img, self.target_size) for img in X]
        return resized_images

class ContrastEnhancer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        enhanced_images = [self._enhance_contrast(img) for img in X]
        return enhanced_images

    def _enhance_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Define the pipeline
    
# image_pipeline = make_pipeline(
#     ImageResizer(),
#     ContrastEnhancer()
# )

# Example usage
# uploaded_image = cv2.imread()  # Load uploaded image
# enhanced_image = image_pipeline.transform([uploaded_image])[0]  # Apply pipeline

