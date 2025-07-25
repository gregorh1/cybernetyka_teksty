#!/usr/bin/env python3
"""
Setup fine-tuned Bielik model for Ollama integration
"""

import os
import subprocess
import shutil
from pathlib import Path
import argparse

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        print(f"✅ Ollama zainstalowane: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Ollama nie jest zainstalowane")
        print("   Zainstaluj z: https://ollama.ai/download")
        return False

def merge_lora_adapter(model_path, output_path):
    """Merge LoRA adapter with base model"""
    print(f"🔄 Łączę LoRA adapter z modelem bazowym...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # Load base model
        base_model_name = "speakleash/Bielik-11B-v2.2-Instruct"
        print(f"   Ładuję model bazowy: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="cpu"  # Use CPU for merging
        )
        
        # Load LoRA adapter
        print(f"   Ładuję adapter LoRA z: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Merge adapter
        print("   Łączę adapter z modelem...")
        merged_model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save merged model
        print(f"   Zapisuję połączony model do: {output_path}")
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print("✅ Model połączony pomyślnie")
        return True
        
    except Exception as e:
        print(f"❌ Błąd łączenia modelu: {e}")
        return False

def convert_to_gguf(model_path, output_path):
    """Convert model to GGUF format for Ollama"""
    print(f"🔄 Konwertuję model do formatu GGUF...")
    
    # Check if llama.cpp convert script exists
    convert_script = "convert-hf-to-gguf.py"
    
    if not shutil.which("python3"):
        print("❌ Python3 nie znaleziony")
        return False
    
    # Try to find llama.cpp converter
    common_paths = [
        "./llama.cpp/convert-hf-to-gguf.py",
        "../llama.cpp/convert-hf-to-gguf.py",
        f"{os.path.expanduser('~')}/llama.cpp/convert-hf-to-gguf.py"
    ]
    
    converter_path = None
    for path in common_paths:
        if os.path.exists(path):
            converter_path = path
            break
    
    if not converter_path:
        print("❌ Nie znaleziono konwertera llama.cpp")
        print("💡 Instrukcje instalacji:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp")
        print("   pip install -r requirements.txt")
        return False
    
    try:
        cmd = [
            "python3", converter_path,
            model_path,
            "--outtype", "f16",
            "--outfile", output_path
        ]
        
        print(f"   Wykonuję: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Konwersja do GGUF zakończona")
            return True
        else:
            print(f"❌ Błąd konwersji: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Błąd konwersji: {e}")
        return False

def create_modelfile(model_name, gguf_path, output_path):
    """Create Ollama Modelfile"""
    print(f"📝 Tworzę Modelfile dla Ollama...")
    
    modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"Użytkownik: {{ .Prompt }}

Asystent cybernetyki: \"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER top_k 40

SYSTEM \"\"\"Jesteś ekspertem w dziedzinie cybernetyki, szczególnie w teorii opracowanej przez Mariana Mazura i Józefa Kosseckiego. Odpowiadasz szczegółowo na pytania dotyczące cybernetyki społecznej, teorii systemów, sprzężeń zwrotnych, homeostazy i innych zagadnień cybernetycznych. Udzielasz wyczerpujących, akademickich odpowiedzi z przykładami i objaśnieniami.\"\"\"
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile zapisany: {output_path}")
    return True

def import_to_ollama(modelfile_path, model_name):
    """Import model to Ollama"""
    print(f"🦙 Importuję model do Ollama jako '{model_name}'...")
    
    try:
        cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        print(f"   Wykonuję: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Model '{model_name}' zaimportowany do Ollama")
            print(f"💡 Użyj: ollama run {model_name}")
            return True
        else:
            print(f"❌ Błąd importu: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Błąd importu: {e}")
        return False

def test_ollama_model(model_name):
    """Test the imported model"""
    print(f"🧪 Testuję model w Ollama...")
    
    try:
        test_prompt = "Co to jest sprzężenie zwrotne w cybernetyce?"
        cmd = ["ollama", "run", model_name, test_prompt]
        
        print(f"   Test prompt: {test_prompt}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ Test zakończony sukcesem:")
            print(f"   Odpowiedź: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ Test nieudany: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test przekroczył limit czasu")
        return False
    except Exception as e:
        print(f"❌ Błąd testu: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup fine-tuned model for Ollama")
    parser.add_argument("--model-path", default="./bielik-cybernetyka-lora",
                       help="Ścieżka do fine-tuned modelu")
    parser.add_argument("--model-name", default="bielik-cybernetyka",
                       help="Nazwa modelu w Ollama")
    parser.add_argument("--skip-merge", action="store_true",
                       help="Pomiń łączenie LoRA (jeśli już wykonane)")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Pomiń konwersję GGUF (jeśli już wykonana)")
    
    args = parser.parse_args()
    
    print("🦙 SETUP MODELU DLA OLLAMA")
    print("=" * 40)
    
    # Check prerequisites
    if not check_ollama_installed():
        return
    
    # Define paths
    merged_path = f"{args.model_path}-merged"
    gguf_path = f"{args.model_name}.gguf"
    modelfile_path = f"{args.model_name}.Modelfile"
    
    # Step 1: Merge LoRA adapter
    if not args.skip_merge:
        if not merge_lora_adapter(args.model_path, merged_path):
            print("❌ Nie udało się połączyć modelu")
            return
    else:
        print("⏭️  Pomijam łączenie LoRA")
    
    # Step 2: Convert to GGUF
    if not args.skip_convert:
        if not convert_to_gguf(merged_path, gguf_path):
            print("❌ Nie udało się skonwertować do GGUF")
            print("💡 Alternatywa: użyj online converters lub llama.cpp")
            return
    else:
        print("⏭️  Pomijam konwersję GGUF")
    
    # Step 3: Create Modelfile
    if not create_modelfile(args.model_name, gguf_path, modelfile_path):
        return
    
    # Step 4: Import to Ollama
    if not import_to_ollama(modelfile_path, args.model_name):
        return
    
    # Step 5: Test model
    test_ollama_model(args.model_name)
    
    print(f"\n🎉 SETUP ZAKOŃCZONY!")
    print(f"   Model dostępny jako: {args.model_name}")
    print(f"   Użyj: ollama run {args.model_name}")
    print(f"   W Open WebUI: dodaj model '{args.model_name}'")

if __name__ == "__main__":
    main() 