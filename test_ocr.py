
from PIL import Image
import pytesseract

# Windows Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = "test_notice.jpeg"

image = Image.open(image_path)

extracted_text = pytesseract.image_to_string(image)

print("----- EXTRACTED TEXT -----")
print(extracted_text)