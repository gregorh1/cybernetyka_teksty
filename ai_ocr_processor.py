#!/usr/bin/env python3
"""
AI-powered OCR processor using qwen2.5vl:7b model
Processes image-only PDFs by extracting pages and using AI for text recognition
"""

import os
import sys
import base64
import io
from pathlib import Path
import argparse
import requests
import json
from typing import List, Optional
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


class AIORProcessor:
    def __init__(self, model_name: str = "qwen2.5vl:7b", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
    def check_model_availability(self) -> bool:
        """Check if the AI model is available in Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if self.model_name in available_models:
                    print(f"✅ Model {self.model_name} is available")
                    return True
                else:
                    print(f"❌ Model {self.model_name} not found")
                    print(f"Available models: {', '.join(available_models)}")
                    return False
            else:
                print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            print("Make sure Ollama is running with: ollama serve")
            return False
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        print(f"📄 Converting PDF to images: {pdf_path.name}")
        
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            print(f"  📑 Processing page {page_num + 1}/{len(doc)}")
            
            page = doc.load_page(page_num)
            
            # Convert to image with specified DPI
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        doc.close()
        print(f"✅ Converted {len(images)} pages to images")
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def process_image_with_ai(self, image: Image.Image, page_num: int) -> str:
        """Process a single image through the AI model"""
        print(f"  🤖 AI processing page {page_num}...")
        
        try:
            # Convert image to base64
            img_base64 = self.image_to_base64(image)
            
            # Prepare the prompt for OCR
            prompt = """Please extract all text from this image. This is a page from an academic document, likely in Polish about cybernetics or social sciences. 

Requirements:
- Extract ALL visible text accurately
- Maintain original formatting where possible
- Include headers, footnotes, and any marginal text
- If text is in Polish, preserve Polish characters (ą, ć, ę, ł, ń, ó, ś, ź, ż)
- Do not add any explanations or comments
- Output only the extracted text

Text:"""

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for accurate OCR
                    "top_p": 0.9
                }
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 2 minute timeout for vision processing
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result.get('response', '').strip()
                
                if extracted_text:
                    print(f"  ✅ Page {page_num}: Extracted {len(extracted_text)} characters")
                    return extracted_text
                else:
                    print(f"  ⚠️  Page {page_num}: No text extracted")
                    return ""
            else:
                print(f"  ❌ Page {page_num}: API error {response.status_code}")
                print(f"      Response: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            print(f"  ⏰ Page {page_num}: Request timeout (2 minutes)")
            return ""
        except Exception as e:
            print(f"  ❌ Page {page_num}: Error - {e}")
            return ""
    
    def process_pdf(self, pdf_path: Path, output_path: Optional[Path] = None, dpi: int = 200) -> bool:
        """Process entire PDF through AI OCR"""
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        if output_path is None:
            output_path = pdf_path.with_suffix('.txt')
        
        print(f"🚀 Starting AI OCR processing:")
        print(f"  📄 Input:  {pdf_path}")
        print(f"  📝 Output: {output_path}")
        print(f"  🤖 Model:  {self.model_name}")
        
        start_time = time.time()
        
        # Convert PDF to images
        try:
            images = self.pdf_to_images(pdf_path, dpi)
        except Exception as e:
            print(f"❌ Failed to convert PDF to images: {e}")
            return False
        
        if not images:
            print("❌ No images extracted from PDF")
            return False
        
        # Process each page with AI
        all_text = []
        failed_pages = []
        
        for i, image in enumerate(images, 1):
            page_text = self.process_image_with_ai(image, i)
            
            if page_text:
                all_text.append(f"--- Page {i} ---\n{page_text}\n")
            else:
                failed_pages.append(i)
                all_text.append(f"--- Page {i} ---\n[OCR FAILED]\n")
        
        # Join all text and save
        full_text = "\n".join(all_text)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add header with metadata
                f.write(f"=== {pdf_path.stem} ===\n")
                f.write(f"Original PDF: {pdf_path.name}\n")
                f.write(f"Processed with: AI OCR ({self.model_name})\n")
                f.write(f"Total pages: {len(images)}\n")
                f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                if failed_pages:
                    f.write(f"Failed pages: {', '.join(map(str, failed_pages))}\n")
                f.write("=" * 80 + "\n\n")
                f.write(full_text)
            
            elapsed = time.time() - start_time
            success_rate = ((len(images) - len(failed_pages)) / len(images)) * 100
            
            print(f"\n🎉 Processing complete!")
            print(f"  ⏱️  Time: {elapsed:.1f} seconds")
            print(f"  📊 Success rate: {success_rate:.1f}% ({len(images) - len(failed_pages)}/{len(images)} pages)")
            print(f"  📝 Output saved: {output_path}")
            
            if failed_pages:
                print(f"  ⚠️  Failed pages: {', '.join(map(str, failed_pages))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save output file: {e}")
            return False


def find_image_pdfs(directory: Path) -> List[Path]:
    """Find PDF files that are likely image-only (no text layer)"""
    print(f"🔍 Scanning for image-only PDFs in {directory}")
    
    image_pdfs = []
    
    for pdf_file in directory.glob("*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            has_text = False
            
            # Check first few pages for text content
            pages_to_check = min(3, len(doc))
            for page_num in range(pages_to_check):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if len(text) > 50:  # More than 50 characters suggests real text
                    has_text = True
                    break
            
            doc.close()
            
            if not has_text:
                image_pdfs.append(pdf_file)
                print(f"  📑 Found image-only PDF: {pdf_file.name}")
        
        except Exception as e:
            print(f"  ⚠️  Could not analyze {pdf_file.name}: {e}")
    
    print(f"✅ Found {len(image_pdfs)} image-only PDFs")
    return image_pdfs


def show_menu():
    """Display interactive menu"""
    print("\n" + "=" * 60)
    print("🤖 AI OCR PROCESSOR - qwen2.5vl:7b")
    print("=" * 60)
    print("1. 📄 Process specific PDF file")
    print("2. 🔍 Scan TEXTS/ folder for image-only PDFs")
    print("3. 🚀 Process all image-only PDFs in TEXTS/ folder")
    print("4. ❌ Exit")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Select option (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return '4'


def scan_texts_folder() -> List[Path]:
    """Scan entire TEXTS/ folder for image-only PDFs"""
    texts_path = Path("TEXTS")
    
    if not texts_path.exists():
        print("❌ TEXTS/ folder not found in current directory")
        return []
    
    print(f"🔍 Scanning entire TEXTS/ folder recursively...")
    
    all_image_pdfs = []
    
    # Recursively scan all subdirectories
    for pdf_file in texts_path.rglob("*.pdf"):
        try:
            doc = fitz.open(pdf_file)
            has_text = False
            
            # Check first few pages for text content
            pages_to_check = min(3, len(doc))
            for page_num in range(pages_to_check):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if len(text) > 50:  # More than 50 characters suggests real text
                    has_text = True
                    break
            
            doc.close()
            
            if not has_text:
                all_image_pdfs.append(pdf_file)
                relative_path = pdf_file.relative_to(texts_path)
                print(f"  📑 Found: TEXTS/{relative_path}")
        
        except Exception as e:
            print(f"  ⚠️  Could not analyze {pdf_file.name}: {e}")
    
    print(f"✅ Found {len(all_image_pdfs)} image-only PDFs in TEXTS/ folder")
    return all_image_pdfs


def process_specific_file(processor: AIORProcessor) -> bool:
    """Process a specific PDF file provided by user"""
    while True:
        file_path = input("\n📄 Enter PDF file path (or 'back' to return to menu): ").strip()
        
        if file_path.lower() == 'back':
            return False
        
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        if pdf_path.suffix.lower() != '.pdf':
            print(f"❌ Not a PDF file: {file_path}")
            continue
        
        print(f"🚀 Processing: {pdf_path}")
        success = processor.process_pdf(pdf_path)
        
        if success:
            print(f"✅ Successfully processed: {pdf_path}")
        else:
            print(f"❌ Failed to process: {pdf_path}")
        
        return True


def interactive_mode(processor: AIORProcessor):
    """Run in interactive menu mode"""
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Process specific file
            process_specific_file(processor)
            
        elif choice == '2':
            # Scan TEXTS folder
            scan_texts_folder()
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            # Process all image PDFs in TEXTS folder
            pdf_files = scan_texts_folder()
            
            if not pdf_files:
                print("❌ No image-only PDFs found in TEXTS/ folder")
                input("Press Enter to continue...")
                continue
            
            print(f"\n🚀 Ready to process {len(pdf_files)} image-only PDFs")
            confirm = input("Continue? (y/N): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                success_count = 0
                
                for i, pdf_file in enumerate(pdf_files, 1):
                    print(f"\n📄 Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
                    
                    if processor.process_pdf(pdf_file):
                        success_count += 1
                    
                    # Add delay between files
                    if i < len(pdf_files):
                        print("⏳ Waiting 2 seconds before next file...")
                        time.sleep(2)
                
                print(f"\n📊 PROCESSING COMPLETE:")
                print(f"  ✅ Successful: {success_count}/{len(pdf_files)}")
                print(f"  ❌ Failed: {len(pdf_files) - success_count}/{len(pdf_files)}")
            
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            # Exit
            print("👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="AI-powered OCR processor for image-only PDFs")
    parser.add_argument("input", nargs="?", help="PDF file to process (if provided, skips menu)")
    parser.add_argument("-m", "--model", default="qwen2.5vl:7b", help="AI model to use (default: qwen2.5vl:7b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF to image conversion (default: 200)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = AIORProcessor(args.model, args.ollama_url)
    
    # Check if model is available
    if not processor.check_model_availability():
        print(f"\n💡 To install the model, run: ollama pull {args.model}")
        return 1
    
    # If file argument provided, process it directly
    if args.input:
        pdf_path = Path(args.input)
        
        if not pdf_path.exists():
            print(f"❌ File not found: {args.input}")
            return 1
        
        if pdf_path.suffix.lower() != '.pdf':
            print(f"❌ Not a PDF file: {args.input}")
            return 1
        
        print(f"🚀 Processing: {pdf_path}")
        success = processor.process_pdf(pdf_path)
        
        return 0 if success else 1
    
    # Otherwise, run interactive menu
    interactive_mode(processor)
    return 0


if __name__ == "__main__":
    sys.exit(main())