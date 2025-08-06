#!/bin/bash

# Script to OCR multiple PDF books containing 'tiff' in filename to text
# Usage: ./tesseract_pdf_ocr.sh [specific_pdf_file]
#   Without parameter: processes all PDF files containing 'tiff' in subfolders
#   With parameter: processes only the specified PDF file

# Show help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "📖 Tesseract PDF OCR Script"
    echo ""
    echo "Usage:"
    echo "  ./tesseract_pdf_ocr.sh                    - Process all PDF files containing 'tiff'"
    echo "  ./tesseract_pdf_ocr.sh path/to/file.pdf   - Process specific PDF file"
    echo "  ./tesseract_pdf_ocr.sh -h                 - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./tesseract_pdf_ocr.sh                                        # Process all files"
    echo "  ./tesseract_pdf_ocr.sh Mazur/cybernetyka-tiff.pdf            # Test with one file"
    echo "  ./tesseract_pdf_ocr.sh /full/path/to/document.pdf            # Process specific file"
    echo ""
    exit 0
fi

# Check dependencies
echo "🔧 Checking dependencies..."
if ! command -v pdftoppm &> /dev/null; then
    echo "❌ pdftoppm not found. Please install poppler-utils (brew install poppler)"
    exit 1
fi

if ! command -v tesseract &> /dev/null; then
    echo "❌ tesseract not found. Please install tesseract (brew install tesseract)"
    exit 1
fi

# Check if Polish language pack is available
if ! tesseract --list-langs 2>/dev/null | grep -q "pol"; then
    echo "⚠️  Polish language pack not found. Installing with: brew install tesseract-lang"
    echo "   You can continue with English, but results may be less accurate for Polish text."
fi

echo "✅ Dependencies checked"
echo ""

# Check if specific file parameter is provided
if [ -n "$1" ]; then
    # Process specific file
    echo "🎯 Processing specific file: $1"
    
    if [ ! -f "$1" ]; then
        echo "❌ File not found: $1"
        exit 1
    fi
    
    if [[ "$1" != *.pdf ]]; then
        echo "❌ File is not a PDF: $1"
        exit 1
    fi
    
    # Use array for single file to preserve spaces
    pdf_files_array=("$1")
else
    # Process all files containing 'tiff'
    echo "🔍 Looking for PDF files containing 'tiff' in subfolders..."
    
    # Use array to store files found by find command
    mapfile -t pdf_files_array < <(find . -name "*tiff*.pdf" -type f)
    
    if [ ${#pdf_files_array[@]} -eq 0 ]; then
        echo "❌ No PDF files containing 'tiff' found in subfolders"
        exit 1
    fi
fi

# Count total files and set up processing
total_files=${#pdf_files_array[@]}
if [ -n "$1" ]; then
    echo "📚 Processing 1 specific PDF file"
else
    echo "📚 Found $total_files PDF files to process"
fi

# Process each PDF file
file_counter=1

for pdf_file in "${pdf_files_array[@]}"; do
    echo ""
    echo "📖 Processing file $file_counter/$total_files: $pdf_file"
    
    # Get directory, filename without extension and path
    pdf_dir=$(dirname "$pdf_file")
    base_name=$(basename "$pdf_file" .pdf)
    
    # Create output file path in same directory as source PDF
    if [[ "$pdf_file" = /* ]]; then
        # Absolute path
        output_file="${pdf_dir}/${base_name}.txt"
    else
        # Relative path - create absolute path for output
        if command -v realpath &> /dev/null; then
            abs_pdf_dir=$(realpath "$pdf_dir")
        else
            abs_pdf_dir="$(pwd)/$pdf_dir"
        fi
        output_file="${abs_pdf_dir}/${base_name}.txt"
    fi
    
    # Create temporary directory for this PDF
    temp_dir=$(mktemp -d)
    echo "📁 Created temp directory: $temp_dir"
    
    # Step 1: Convert PDF to images in temp directory
    echo "🖼️  Converting PDF to images..."
    
    # Get absolute path for pdftoppm (since we'll be in temp directory)
    if [[ "$pdf_file" = /* ]]; then
        # Already absolute path
        pdf_abs_path="$pdf_file"
    else
        # Convert relative path to absolute using realpath or fallback
        if command -v realpath &> /dev/null; then
            pdf_abs_path=$(realpath "$pdf_file")
        else
            pdf_abs_path="$(pwd)/$pdf_file"
        fi
    fi
    
    cd "$temp_dir"
    pdftoppm -png "$pdf_abs_path" page
    
    # Check if conversion worked (pdftoppm creates page-01.png, page-02.png, etc.)
    if [ ! -f page-01.png ] && [ ! -f page-001.png ]; then
        echo "❌ PDF conversion failed for $pdf_file"
        cd - > /dev/null
        rm -rf "$temp_dir"
        ((file_counter++))
        continue
    fi
    
    # Count pages
    page_count=$(ls page-*.png 2>/dev/null | wc -l)
    echo "📄 Found $page_count pages"
    
    # Step 2: OCR each page
    echo "🔤 Starting OCR process..."
    counter=1
    for img_file in $(ls page-*.png | sort -V); do
        echo "   Processing page $counter/$page_count..."
        basename_img=$(basename "$img_file" .png)
        
        # OCR with Polish language and book-optimized settings
        tesseract "$img_file" "${basename_img}-ocr" -l pol \
            --psm 1 \
            --oem 3 \
            -c preserve_interword_spaces=1 \
            2>/dev/null
        
        ((counter++))
    done
    
    # Step 3: Combine all text files into final output
    echo "📝 Combining all pages into final text file..."
    
    # Add header to output file
    echo "=== $base_name ===" > "$output_file"
    echo "Generated on: $(date)" >> "$output_file"
    echo "Original PDF: $pdf_file" >> "$output_file"
    echo "Total pages: $page_count" >> "$output_file"
    echo "" >> "$output_file"
    
    # Combine all OCR text files in order
    for txt_file in $(ls page-*-ocr.txt 2>/dev/null | sort -V); do
        page_num=$(echo "$txt_file" | grep -o '[0-9]\+')
        echo "--- Page $page_num ---" >> "$output_file"
        cat "$txt_file" >> "$output_file"
        echo "" >> "$output_file"
    done
    
    # Step 4: Cleanup temp directory
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    # Report for this file
    if [ -f "$output_file" ]; then
        file_size=$(wc -c < "$output_file")
        line_count=$(wc -l < "$output_file")
        echo "✅ OCR completed for: $base_name"
        echo "📁 Output file: $output_file"
        echo "📊 File size: $file_size bytes"
        echo "📄 Lines: $line_count"
    else
        echo "❌ Failed to create output file for: $base_name"
    fi
    
    ((file_counter++))
done

echo ""
if [ -n "$1" ]; then
    echo "🎉 PDF file processed successfully!"
    echo "💡 You can find the .txt file in the same directory as the original PDF"
else
    echo "🎉 All PDF files processed!"
    echo "💡 You can find the .txt files in their respective directories alongside the original PDFs"
fi
