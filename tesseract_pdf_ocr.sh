#!/bin/bash

# Script to OCR multiple PDF books containing 'tiff' in filename to text
# Usage: ./tesseract_pdf_ocr.sh [specific_pdf_file] [-t topic]
#   Without parameter: processes all PDF files containing 'tiff' in all topics
#   With specific file: processes only the specified PDF file
#   With -t topic: processes files only from specific topic folder

# Show help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "📖 Tesseract PDF OCR Script"
    echo ""
    echo "Usage:"
    echo "  ./tesseract_pdf_ocr.sh                        - Process all PDF files containing 'tiff' in all topics"
    echo "  ./tesseract_pdf_ocr.sh -t <topic>             - Process files only from specific topic"
    echo "  ./tesseract_pdf_ocr.sh path/to/file.pdf       - Process specific PDF file"
    echo "  ./tesseract_pdf_ocr.sh -h                     - Show this help"
    echo "  ./tesseract_pdf_ocr.sh --list-topics          - Show available topics"
    echo ""
    echo "Examples:"
    echo "  ./tesseract_pdf_ocr.sh                                    # Process all files"
    echo "  ./tesseract_pdf_ocr.sh -t cybernertics                    # Process only cybernertics topic"
    echo "  ./tesseract_pdf_ocr.sh TOPICS/cybernertics/file.pdf       # Process specific file"
    echo ""
    exit 0
fi

# List available topics
if [[ "$1" == "--list-topics" ]]; then
    echo "📂 Available topics:"
    if [ -d "TOPICS" ]; then
        for topic_dir in TOPICS/*/; do
            if [ -d "$topic_dir" ]; then
                topic_name=$(basename "$topic_dir")
                echo "  • $topic_name"
            fi
        done
    else
        echo "❌ TOPICS/ directory not found"
    fi
    exit 0
fi

# Parse arguments
TOPIC=""
SPECIFIC_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--topic)
            TOPIC="$2"
            shift 2
            ;;
        *.pdf)
            SPECIFIC_FILE="$1"
            shift
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

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
if [ -n "$SPECIFIC_FILE" ]; then
    # Process specific file
    echo "🎯 Processing specific file: $SPECIFIC_FILE"
    
    if [ ! -f "$SPECIFIC_FILE" ]; then
        echo "❌ File not found: $SPECIFIC_FILE"
        exit 1
    fi
    
    if [[ "$SPECIFIC_FILE" != *.pdf ]]; then
        echo "❌ File is not a PDF: $SPECIFIC_FILE"
        exit 1
    fi
    
    # Use array for single file to preserve spaces
    pdf_files_array=("$SPECIFIC_FILE")
else
    # Process files based on topic or all topics
    if [ -n "$TOPIC" ]; then
        # Process specific topic
        echo "🔍 Looking for PDF files containing 'tiff' in topic: $TOPIC"
        
        if [ ! -d "TOPICS/$TOPIC" ]; then
            echo "❌ Topic directory not found: TOPICS/$TOPIC"
            echo "Available topics:"
            for topic_dir in TOPICS/*/; do
                if [ -d "$topic_dir" ]; then
                    echo "  • $(basename "$topic_dir")"
                fi
            done
            exit 1
        fi
        
        # Use array to store files found by find command in specific topic
        mapfile -t pdf_files_array < <(find "TOPICS/$TOPIC" -name "*tiff*.pdf" -type f)
    else
        # Process all topics
        echo "🔍 Looking for PDF files containing 'tiff' in all topics..."
        
        if [ ! -d "TOPICS" ]; then
            echo "❌ TOPICS directory not found"
            exit 1
        fi
        
        # Use array to store files found by find command in all topics
        mapfile -t pdf_files_array < <(find "TOPICS" -name "*tiff*.pdf" -type f)
    fi
    
    if [ ${#pdf_files_array[@]} -eq 0 ]; then
        if [ -n "$TOPIC" ]; then
            echo "❌ No PDF files containing 'tiff' found in topic: $TOPIC"
        else
            echo "❌ No PDF files containing 'tiff' found in TOPICS"
        fi
        exit 1
    fi
fi

# Count total files and set up processing
total_files=${#pdf_files_array[@]}
if [ -n "$SPECIFIC_FILE" ]; then
    echo "📚 Processing 1 specific PDF file"
elif [ -n "$TOPIC" ]; then
    echo "📚 Found $total_files PDF files to process in topic: $TOPIC"
else
    echo "📚 Found $total_files PDF files to process across all topics"
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
if [ -n "$SPECIFIC_FILE" ]; then
    echo "🎉 PDF file processed successfully!"
    echo "💡 You can find the .txt file in the same directory as the original PDF"
elif [ -n "$TOPIC" ]; then
    echo "🎉 All PDF files in topic '$TOPIC' processed!"
    echo "💡 You can find the .txt files in their respective directories alongside the original PDFs"
else
    echo "🎉 All PDF files across all topics processed!"
    echo "💡 You can find the .txt files in their respective directories alongside the original PDFs"
fi
