# Test script
from services.embedding_service import EmbeddingService

service = EmbeddingService()

print(f"Text dim: {service.get_embedding_dim()}")
print(f"Image dim: {service.get_image_embedding_dim()}")

# Test actual encoding
text_emb = service.encode_text("test")
print(f"Actual text shape: {text_emb.shape}")

from PIL import Image
img = Image.new('RGB', (224, 224))
img_emb = service.encode_images(img)
print(f"Actual image shape: {img_emb.shape}")