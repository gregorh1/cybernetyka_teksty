#!/usr/bin/env python3
"""
Inventory Script for Cybernetyka Teksty Collection
Analyzes all files in Kossecki and Mazur folders to prepare data for AI training
"""

import os
import glob
from pathlib import Path
from collections import defaultdict
import json

def get_file_info(filepath):
    """Get detailed information about a file"""
    path = Path(filepath)
    try:
        size = path.stat().st_size
        return {
            'name': path.name,
            'size_bytes': size,
            'size_mb': round(size / (1024 * 1024), 2),
            'extension': path.suffix.lower(),
            'exists': True
        }
    except FileNotFoundError:
        return {
            'name': path.name,
            'size_bytes': 0,
            'size_mb': 0,
            'extension': path.suffix.lower(),
            'exists': False
        }

def analyze_folder(folder_path):
    """Analyze all files in a folder"""
    folder_stats = {
        'total_files': 0,
        'by_extension': defaultdict(list),
        'by_type': defaultdict(list),
        'total_size_mb': 0,
        'text_ready': [],  # Files already in text format
        'needs_processing': [],  # Files that need conversion
        'processed_pairs': [],  # PDF->TXT pairs from OCR
        'orphaned_files': []  # Files without corresponding text
    }
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return folder_stats
    
    all_files = list(Path(folder_path).glob('*'))
    folder_stats['total_files'] = len(all_files)
    
    # Group files by base name (without extension)
    file_groups = defaultdict(list)
    
    for file_path in all_files:
        if file_path.is_file():
            file_info = get_file_info(file_path)
            folder_stats['total_size_mb'] += file_info['size_mb']
            folder_stats['by_extension'][file_info['extension']].append(file_info)
            
            # Categorize by type
            ext = file_info['extension']
            if ext == '.txt':
                folder_stats['by_type']['text'].append(file_info)
                folder_stats['text_ready'].append(file_info)
            elif ext == '.pdf':
                folder_stats['by_type']['pdf'].append(file_info)
            elif ext == '.doc':
                folder_stats['by_type']['doc'].append(file_info)
                folder_stats['needs_processing'].append(file_info)
            elif ext == '.sh':
                folder_stats['by_type']['script'].append(file_info)
            else:
                folder_stats['by_type']['other'].append(file_info)
            
            # Group by base name for pairing analysis
            base_name = file_path.stem
            # Handle special cases like -tiff, -ocr suffixes
            for suffix in ['-tiff', '-ocr']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            file_groups[base_name].append((file_path, file_info))
    
    # Analyze file pairs and relationships
    for base_name, files in file_groups.items():
        pdf_files = []
        txt_files = []
        doc_files = []
        
        for file_path, file_info in files:
            ext = file_info['extension']
            if ext == '.pdf':
                pdf_files.append((file_path, file_info))
            elif ext == '.txt':
                txt_files.append((file_path, file_info))
            elif ext == '.doc':
                doc_files.append((file_path, file_info))
        
        # Check for OCR-processed pairs
        tiff_pdfs = [f for f in pdf_files if '-tiff' in f[0].name]
        corresponding_txts = [f for f in txt_files if any(tp[0].stem + '.txt' == f[0].name for tp in tiff_pdfs)]
        
        for tiff_pdf in tiff_pdfs:
            corresponding_txt = next((txt for txt in txt_files if txt[0].name == tiff_pdf[0].stem + '.txt'), None)
            if corresponding_txt:
                folder_stats['processed_pairs'].append({
                    'pdf': tiff_pdf[1],
                    'txt': corresponding_txt[1],
                    'base_name': base_name
                })
        
        # Check for files that need processing
        ocr_pdfs = [f for f in pdf_files if '-ocr' in f[0].name and not any('-tiff' in tf[0].name for tf in pdf_files)]
        for ocr_pdf in ocr_pdfs:
            if not any(txt[0].name.replace('.txt', '') == ocr_pdf[0].name.replace('.pdf', '') for txt in txt_files):
                folder_stats['needs_processing'].append(ocr_pdf[1])
        
        # Check for orphaned DOC files
        for doc_file in doc_files:
            if not any(txt[0].name.replace('.txt', '') == doc_file[0].name.replace('.doc', '') for txt in txt_files):
                folder_stats['orphaned_files'].append(doc_file[1])
    
    return folder_stats

def create_transformation_plan(kossecki_stats, mazur_stats):
    """Create a plan for transforming all data to text format"""
    plan = {
        'summary': {},
        'ready_for_ai': [],
        'conversion_needed': [],
        'recommendations': []
    }
    
    # Calculate totals
    total_text_files = len(kossecki_stats['text_ready']) + len(mazur_stats['text_ready'])
    total_needs_conversion = len(kossecki_stats['needs_processing']) + len(mazur_stats['needs_processing']) + len(kossecki_stats['orphaned_files']) + len(mazur_stats['orphaned_files'])
    
    plan['summary'] = {
        'total_text_ready': total_text_files,
        'total_needs_conversion': total_needs_conversion,
        'kossecki_processed_pairs': len(kossecki_stats['processed_pairs']),
        'mazur_processed_pairs': len(mazur_stats['processed_pairs'])
    }
    
    # Files ready for AI training
    for stats, author in [(kossecki_stats, 'Kossecki'), (mazur_stats, 'Mazur')]:
        for txt_file in stats['text_ready']:
            plan['ready_for_ai'].append({
                'author': author,
                'file': txt_file['name'],
                'size_mb': txt_file['size_mb'],
                'status': 'ready'
            })
    
    # Files needing conversion
    for stats, author in [(kossecki_stats, 'Kossecki'), (mazur_stats, 'Mazur')]:
        for file_info in stats['needs_processing'] + stats['orphaned_files']:
            conversion_type = 'OCR' if file_info['extension'] == '.pdf' else 'DOC_to_TXT'
            plan['conversion_needed'].append({
                'author': author,
                'file': file_info['name'],
                'size_mb': file_info['size_mb'],
                'conversion_type': conversion_type
            })
    
    # Generate recommendations
    plan['recommendations'] = [
        "Convert all .doc files to .txt using LibreOffice or similar tool",
        "Process remaining -ocr.pdf files with OCR if they contain scanned text",
        "Clean and normalize existing .txt files for consistent formatting",
        "Create metadata file with author, title, year, and topic information",
        "Consider organizing by topic/theme rather than just by author",
        "Implement quality control checks for OCR accuracy"
    ]
    
    return plan

def main():
    """Main function to create inventory and transformation plan"""
    print("🔍 Creating inventory for Cybernetyka Teksty collection...")
    print("=" * 60)
    
    # Analyze both folders
    kossecki_stats = analyze_folder('Kossecki')
    mazur_stats = analyze_folder('Mazur')
    
    # Print detailed inventory
    print("\n📁 KOSSECKI FOLDER ANALYSIS")
    print("-" * 30)
    print(f"Total files: {kossecki_stats['total_files']}")
    print(f"Total size: {kossecki_stats['total_size_mb']:.1f} MB")
    print(f"Text files ready: {len(kossecki_stats['text_ready'])}")
    print(f"OCR-processed pairs: {len(kossecki_stats['processed_pairs'])}")
    print(f"Files needing conversion: {len(kossecki_stats['needs_processing']) + len(kossecki_stats['orphaned_files'])}")
    
    print("\nFile types breakdown:")
    for ext, files in kossecki_stats['by_extension'].items():
        print(f"  {ext}: {len(files)} files ({sum(f['size_mb'] for f in files):.1f} MB)")
    
    print("\n📁 MAZUR FOLDER ANALYSIS")
    print("-" * 30)
    print(f"Total files: {mazur_stats['total_files']}")
    print(f"Total size: {mazur_stats['total_size_mb']:.1f} MB")
    print(f"Text files ready: {len(mazur_stats['text_ready'])}")
    print(f"OCR-processed pairs: {len(mazur_stats['processed_pairs'])}")
    print(f"Files needing conversion: {len(mazur_stats['needs_processing']) + len(mazur_stats['orphaned_files'])}")
    
    print("\nFile types breakdown:")
    for ext, files in mazur_stats['by_extension'].items():
        print(f"  {ext}: {len(files)} files ({sum(f['size_mb'] for f in files):.1f} MB)")
    
    # Create transformation plan
    plan = create_transformation_plan(kossecki_stats, mazur_stats)
    
    print("\n🎯 TRANSFORMATION PLAN")
    print("-" * 30)
    print(f"Files ready for AI training: {plan['summary']['total_text_ready']}")
    print(f"Files needing conversion: {plan['summary']['total_needs_conversion']}")
    print(f"Successfully OCR-processed: {plan['summary']['kossecki_processed_pairs'] + plan['summary']['mazur_processed_pairs']}")
    
    print("\n📋 CONVERSION TASKS NEEDED:")
    conversion_tasks = defaultdict(list)
    for item in plan['conversion_needed']:
        conversion_tasks[item['conversion_type']].append(item)
    
    for conv_type, items in conversion_tasks.items():
        print(f"\n{conv_type}:")
        for item in items:
            print(f"  - {item['author']}/{item['file']} ({item['size_mb']} MB)")
    
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(plan['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save detailed report
    report = {
        'kossecki_analysis': kossecki_stats,
        'mazur_analysis': mazur_stats,
        'transformation_plan': plan,
        'generated_at': str(Path.cwd()),
        'total_authors': 2
    }
    
    # Convert defaultdict to regular dict for JSON serialization
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        for k, v in d.items():
            if isinstance(v, defaultdict):
                d[k] = dict(v)
            elif isinstance(v, list):
                d[k] = [convert_defaultdict(item) if isinstance(item, (dict, defaultdict)) else item for item in v]
            elif isinstance(v, dict):
                d[k] = convert_defaultdict(v)
        return d
    
    report = convert_defaultdict(report)
    
    with open('inventory_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed report saved to: inventory_report.json")
    print("\n🎉 Inventory complete!")

if __name__ == "__main__":
    main() 