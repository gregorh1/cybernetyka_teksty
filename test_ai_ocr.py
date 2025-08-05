#!/usr/bin/env python3
"""
Test script for AI OCR processor
"""

from ai_ocr_processor import AIORProcessor, find_image_pdfs
from pathlib import Path

def test_ai_ocr():
    """Test the AI OCR processor"""
    print("🧪 Testing AI OCR Processor")
    print("=" * 40)
    
    # Initialize processor
    processor = AIORProcessor()
    
    # Check if model is available
    if not processor.check_model_availability():
        print("❌ qwen2.5vl:7b model not available")
        print("💡 Install with: ollama pull qwen2.5vl:7b")
        return False
    
    # Look for image PDFs in TEXTS folder
    texts_dir = Path("TEXTS/autonom")
    if texts_dir.exists():
        for author_dir in texts_dir.iterdir():
            if author_dir.is_dir():
                print(f"\n📁 Checking {author_dir.name}")
                image_pdfs = find_image_pdfs(author_dir)
                
                if image_pdfs:
                    print(f"Found {len(image_pdfs)} image-only PDFs")
                    # You can process them here if needed
                    # for pdf_file in image_pdfs[:1]:  # Process just first one for testing
                    #     processor.process_pdf(pdf_file)
    else:
        print("❌ TEXTS/autonom directory not found")
    
    return True

if __name__ == "__main__":
    test_ai_ocr()