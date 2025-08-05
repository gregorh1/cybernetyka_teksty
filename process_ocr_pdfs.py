#!/usr/bin/env python3
"""
Przetwarzanie pozostałych plików PDF-OCR z wykrywaniem duplikatów
"""

import os
import subprocess
from pathlib import Path
import hashlib

def check_for_duplicates():
    """Sprawdź czy są duplikaty plików tekstowych"""
    print("🔍 Sprawdzam duplikaty...")
    
    duplicates = []
    txt_files = []
    
    for folder in ['TEXTS/autonom/Kossecki', 'TEXTS/autonom/Mazur']:
        if os.path.exists(folder):
            txt_files.extend(list(Path(folder).glob('*.txt')))
    
    # Szukaj plików o podobnych nazwach
    base_names = {}
    for txt_file in txt_files:
        # Usuń sufiksy -ocr, -tiff itp.
        base_name = txt_file.stem
        for suffix in ['-tiff', '-ocr', '-wyd1', '-wyd2']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        if base_name in base_names:
            duplicates.append((base_names[base_name], txt_file))
            print(f"⚠️  Potencjalny duplikat: {base_names[base_name].name} ↔ {txt_file.name}")
        else:
            base_names[base_name] = txt_file
    
    return duplicates

def find_ocr_pdfs_to_process():
    """Znajdź pliki PDF-OCR które nie mają odpowiadających plików TXT"""
    
    print("📋 Sprawdzam pliki PDF-OCR do przetworzenia...")
    
    to_process = []
    skipped = []
    
    for folder in ['TEXTS/autonom/Kossecki', 'TEXTS/autonom/Mazur']:
        if not os.path.exists(folder):
            continue
            
        folder_path = Path(folder)
        
        # Znajdź wszystkie pliki PDF z -ocr
        ocr_pdfs = list(folder_path.glob('*-ocr*.pdf'))
        
        for pdf_file in ocr_pdfs:
            # Sprawdź czy odpowiadający plik .txt już istnieje
            potential_txt_names = [
                pdf_file.stem + '.txt',
                pdf_file.stem.replace('-ocr', '') + '.txt',
                pdf_file.stem.replace('-ocr', '-tiff') + '.txt'
            ]
            
            existing_txt = None
            for txt_name in potential_txt_names:
                txt_path = folder_path / txt_name
                if txt_path.exists():
                    existing_txt = txt_path
                    break
            
            if existing_txt:
                skipped.append({
                    'pdf': pdf_file,
                    'existing_txt': existing_txt,
                    'reason': f'Istnieje {existing_txt.name}'
                })
            else:
                to_process.append(pdf_file)
    
    # Podsumowanie
    print(f"\n📊 ANALIZA PDF-OCR:")
    print(f"🔄 Do przetworzenia: {len(to_process)}")
    print(f"⏭️  Pominiętych: {len(skipped)}")
    
    if skipped:
        print(f"\n⏭️  POMIJAM (już mają TXT):")
        for item in skipped[:10]:  # Pokaż tylko pierwsze 10
            print(f"   {item['pdf'].name} → {item['reason']}")
        if len(skipped) > 10:
            print(f"   ... i {len(skipped)-10} więcej")
    
    if to_process:
        print(f"\n🔄 DO PRZETWORZENIA:")
        for pdf_file in to_process[:10]:
            print(f"   {pdf_file}")
        if len(to_process) > 10:
            print(f"   ... i {len(to_process)-10} więcej")
    
    return to_process, skipped

def estimate_pdf_text_content(pdf_path):
    """Szybka ocena czy PDF ma tekst czy tylko obrazy"""
    try:
        # Próba wyciągnięcia tekstu z PDF
        result = subprocess.run([
            'pdftotext', str(pdf_path), '-'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            text = result.stdout.strip()
            if len(text) > 100:  # Jeśli mamy dużo tekstu
                return 'text', len(text)
            elif len(text) > 0:
                return 'some_text', len(text)
        
        return 'images_only', 0
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 'unknown', 0

def process_ocr_pdf_with_tesseract(pdf_path):
    """Przetwórz PDF używając naszego skryptu OCR"""
    
    print(f"\n🔄 OCR dla: {pdf_path.name}")
    
    try:
        # Wywołaj nasz istniejący skrypt OCR
        result = subprocess.run([
            'python3', 'ocr_book.sh', str(pdf_path)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Sprawdź czy powstał plik TXT
            expected_txt = pdf_path.parent / f"{pdf_path.stem}.txt"
            if expected_txt.exists():
                size = expected_txt.stat().st_size
                print(f"✅ Sukces OCR: {expected_txt.name} ({size} bajtów)")
                return expected_txt
        
        print(f"❌ OCR failed: {result.stderr}")
        return None
        
    except subprocess.TimeoutExpired:
        print(f"❌ OCR timeout dla {pdf_path.name}")
        return None

def process_with_direct_text_extraction(pdf_path):
    """Wyciągnij tekst bezpośrednio z PDF (jeśli ma wbudowany tekst)"""
    
    try:
        result = subprocess.run([
            'pdftotext', str(pdf_path), '-'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            txt_path = pdf_path.parent / f"{pdf_path.stem}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== {pdf_path.stem} ===\n")
                f.write(f"Extracted from: {pdf_path.name}\n")
                f.write("Generated by: pdftotext\n\n")
                f.write(result.stdout)
            
            print(f"✅ Tekst wyciągnięty: {txt_path.name}")
            return txt_path
        
        return None
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def main():
    """Główna funkcja przetwarzania PDF-OCR"""
    
    print("📄 Przetwarzanie pozostałych plików PDF-OCR")
    print("=" * 50)
    
    # Sprawdź duplikaty
    duplicates = check_for_duplicates()
    
    # Znajdź pliki do przetworzenia  
    to_process, skipped = find_ocr_pdfs_to_process()
    
    if not to_process:
        print("\n✅ Wszystkie pliki PDF-OCR już zostały przetworzone!")
        return
    
    print(f"\n🎯 Rozpoczynam przetwarzanie {len(to_process)} plików...")
    
    processed = 0
    failed = 0
    
    for pdf_file in to_process:
        # Sprawdź typ zawartości PDF
        content_type, text_length = estimate_pdf_text_content(pdf_file)
        
        print(f"\n📄 {pdf_file.name} ({content_type})")
        
        result = None
        
        if content_type == 'text' and text_length > 500:
            # PDF ma już tekst - wyciągnij bezpośrednio
            result = process_with_direct_text_extraction(pdf_file)
        
        if not result:
            # Fallback: OCR z tesseract
            print("🔄 Próba OCR...")
            # Tu użyjemy tesseract bezpośrednio, bo nasz skrypt jest dla plików -tiff
            # TODO: Zmodyfikuj ocr_book.sh żeby obsługiwał różne pliki
            
            # Na razie proste podejście z pdftotext
            try:
                result = subprocess.run([
                    'pdftotext', str(pdf_file), str(pdf_file.parent / f"{pdf_file.stem}.txt")
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    txt_path = pdf_file.parent / f"{pdf_file.stem}.txt"
                    if txt_path.exists() and txt_path.stat().st_size > 100:
                        print(f"✅ Wyciągnięto tekst: {txt_path.name}")
                        processed += 1
                        continue
                
            except subprocess.TimeoutExpired:
                pass
        
        if result:
            processed += 1
        else:
            print(f"❌ Nie udało się przetworzyć: {pdf_file.name}")
            failed += 1
    
    print(f"\n📊 PODSUMOWANIE PRZETWARZANIA PDF-OCR:")
    print(f"✅ Przetworzone: {processed}")
    print(f"❌ Nieudane: {failed}")
    print(f"⏭️  Pominięte: {len(skipped)}")
    
    if processed > 0:
        print(f"\n🔄 Uruchom ponownie create_corpus.py żeby zaktualizować korpus!")

if __name__ == "__main__":
    main() 