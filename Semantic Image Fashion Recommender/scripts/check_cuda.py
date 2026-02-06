from services.embedding_service import EmbeddingService
from PIL import Image
import torch

print("=" * 60)
print("CUDA Test")
print("=" * 60)

# Check PyTorch CUDA
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

# Load service (will print device info)
service = EmbeddingService()

# Test encoding
test_img = Image.new('RGB', (256, 256), color='red')
embedding = service.encode_images(test_img)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding dtype: {embedding.dtype}")

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")

print("=" * 60)
print("✅ Test passed!")
