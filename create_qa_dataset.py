#!/usr/bin/env python3
"""
Automatyczne tworzenie datasetu Q&A z korpusu cybernetyki
Obsługuje: OpenAI API, local models, oraz Claude API
"""

import json
import re
from pathlib import Path
import time
from typing import List, Dict, Optional
import argparse
import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Załadowano zmienne z .env")
except ImportError:
    print("📝 Zainstaluj python-dotenv dla obsługi .env: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Błąd ładowania .env: {e}")

def load_corpus(corpus_file: str = "cybernetyka_corpus.txt") -> List[str]:
    """Załaduj i podziel korpus na fragmenty"""
    print("📚 Ładuję korpus...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Podziel na dokumenty
    documents = re.split(r'={80}', full_text)
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 200]
    
    # Podziel długie dokumenty na fragmenty ~500-1000 słów
    chunks = []
    for doc in documents:
        paragraphs = doc.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(test_chunk.split()) <= 800:  # ~800 words
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    print(f"✅ Przygotowano {len(chunks)} fragmentów do przetworzenia")
    return chunks

def generate_qa_openai(text_chunk: str, api_key: str) -> List[Dict]:
    """Generuj Q&A używając OpenAI API"""
    try:
        import openai
        openai.api_key = api_key
        
        prompt = f"""Przeczytaj poniższy tekst z polskiej cybernetyki i wygeneruj 3-5 par pytanie-odpowiedź.

WYMAGANIA:
- Pytania różnego typu: faktyczne, koncepcyjne, analityczne
- Odpowiedzi oparte na tekście, precyzyjne
- Używaj polskiej terminologii cybernetycznej
- Zachowaj akademicki styl

TEKST:
{text_chunk}

FORMAT (zwróć TYLKO JSON):
[
  {{"question": "pytanie", "answer": "odpowiedź"}},
  {{"question": "pytanie", "answer": "odpowiedź"}}
]"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper option
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Błąd OpenAI API: {e}")
        return []

def generate_qa_local(text_chunk: str, model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> List[Dict]:
    """Generuj Q&A używając lokalnego modelu"""
    try:
        from transformers import pipeline
        
        generator = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        prompt = f"""<|user|>
Wygeneruj 3-4 pytania i odpowiedzi z tego tekstu o cybernetyce polskiej. Zwróć w formacie JSON:

{text_chunk[:1500]}

<|assistant|>
Oto pytania i odpowiedzi w formacie JSON:
["""

        result = generator(
            prompt, 
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        generated = result[0]['generated_text'][len(prompt):]
        
        # Try to extract JSON
        try:
            # Find JSON in generated text
            json_start = generated.find('[')
            json_end = generated.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = generated[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
            
        # Fallback: manual parsing
        qa_pairs = []
        lines = generated.split('\n')
        current_q = None
        
        for line in lines:
            if line.strip().startswith('Q:') or 'question' in line.lower():
                current_q = line.strip()
            elif line.strip().startswith('A:') or 'answer' in line.lower():
                if current_q:
                    qa_pairs.append({
                        "question": current_q.replace('Q:', '').strip(),
                        "answer": line.strip().replace('A:', '').strip()
                    })
                    current_q = None
                    
        return qa_pairs[:4]  # Max 4
        
    except Exception as e:
        print(f"❌ Błąd lokalnego modelu: {e}")
        return []

def generate_qa_claude(text_chunk: str, api_key: str) -> List[Dict]:
    """Generuj Q&A używając Claude API"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""Przeczytaj ten tekst z polskiej cybernetyki i wygeneruj 3-5 par pytanie-odpowiedź.

TEKST:
{text_chunk}

Zwróć odpowiedź w formacie JSON:
[
  {{"question": "pytanie", "answer": "odpowiedź"}},
  ...
]"""

        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Cheapest option
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = message.content[0].text.strip()
        
        # Extract JSON
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        return json.loads(content)
        
    except Exception as e:
        print(f"❌ Błąd Claude API: {e}")
        return []

def generate_qa_ollama(text_chunk: str, model_name: str = "phi3:mini") -> List[Dict]:
    """Generuj Q&A używając Ollama"""
    try:
        import requests
        
        prompt = f"""Na podstawie tego tekstu z cybernetyki polskiej wygeneruj 3-4 pytania i odpowiedzi w formacie JSON:

{text_chunk[:1500]}

Format JSON:
[
  {{"question": "pytanie", "answer": "odpowiedź"}},
  ...
]"""

        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            
            # Try to extract JSON
            try:
                json_start = generated_text.find('[')
                json_end = generated_text.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = generated_text[json_start:json_end]
                    qa_pairs = json.loads(json_str)
                    return qa_pairs[:4]
            except:
                pass
                
        print(f"⚠️  Ollama response parsing failed")
        return []
        
    except Exception as e:
        print(f"❌ Błąd Ollama: {e}")
        print(f"💡 Sprawdź czy Ollama działa: ollama list")
        return []

def generate_qa_finetuned(text_chunk: str, model_path: str) -> List[Dict]:
    """Generuj Q&A używając wytrenowanego modelu"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        
        print(f"🔄 Ładuję model z: {model_path}")
        
        # Load your fine-tuned model
        base_model_name = "speakleash/Bielik-11B-v2.2-Instruct"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set for inference
        model.eval()
        
        prompt = f"""Na podstawie tego tekstu z cybernetyki wygeneruj 3-4 pytania i odpowiedzi:

{text_chunk[:1200]}

Format:
Q: [pytanie o cybernetyce]
A: [odpowiedź na podstawie tekstu]

Q:"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_response = generated_text[len(prompt):]
        
        # Parse Q&A pairs
        qa_pairs = []
        lines = generated_response.split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_q and current_a:
                    qa_pairs.append({
                        "question": current_q,
                        "answer": current_a
                    })
                current_q = line[2:].strip()
                current_a = None
            elif line.startswith('A:'):
                current_a = line[2:].strip()
            elif current_a and line and not line.startswith('Q:'):
                current_a += " " + line
        
        # Add last Q&A
        if current_q and current_a:
            qa_pairs.append({
                "question": current_q,
                "answer": current_a
            })
        
        print(f"🧠 Wygenerowano {len(qa_pairs)} par Q&A ekspertem cybernetyki")
        return qa_pairs[:4]
        
    except Exception as e:
        print(f"❌ Błąd fine-tuned model: {e}")
        return []

def create_instruction_format(qa_pairs: List[Dict]) -> List[Dict]:
    """Przekonwertuj Q&A na format instrukcyjny"""
    instructions = []
    
    for qa in qa_pairs:
        instructions.append({
            "instruction": qa["question"],
            "input": "",
            "output": qa["answer"]
        })
        
        # Dodaj warianty
        instructions.append({
            "instruction": "Odpowiedz na pytanie dotyczące cybernetyki:",
            "input": qa["question"],
            "output": qa["answer"]
        })
    
    return instructions

def save_progress(qa_pairs: List[Dict], output_file: str, chunk_index: int):
    """Zapisz postęp po każdym fragmencie"""
    progress_file = output_file.replace('.json', f'_progress.json')
    backup_file = output_file.replace('.json', f'_backup_{chunk_index}.json')
    
    progress_data = {
        "last_processed_chunk": chunk_index,
        "total_qa_pairs": len(qa_pairs),
        "qa_pairs": qa_pairs,
        "timestamp": time.time()
    }
    
    # Zapisz główny plik postępu
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    # Zapisz backup co 10 fragmentów
    if chunk_index % 10 == 0:
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Zapisano postęp: {len(qa_pairs)} par Q&A (fragment {chunk_index})")

def load_progress(output_file: str) -> tuple:
    """Załaduj postęp z poprzedniej sesji"""
    progress_file = output_file.replace('.json', f'_progress.json')
    
    if not Path(progress_file).exists():
        return [], 0
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        qa_pairs = progress_data.get("qa_pairs", [])
        last_chunk = progress_data.get("last_processed_chunk", 0)
        
        print(f"📂 Znaleziono poprzedni postęp:")
        print(f"   Ostatni fragment: {last_chunk}")
        print(f"   Zebrano par Q&A: {len(qa_pairs)}")
        
        resume = input("   Kontynuować od tego miejsca? [T/n]: ").strip().lower()
        if resume in ['', 't', 'tak', 'y', 'yes']:
            return qa_pairs, last_chunk + 1
        else:
            return [], 0
            
    except Exception as e:
        print(f"❌ Błąd wczytywania postępu: {e}")
        return [], 0

def generate_qa_with_retry(chunk: str, method: str, api_key: str = None, model: str = None, max_retries: int = 3) -> List[Dict]:
    """Generuj Q&A z ponowną próbą w przypadku błędu"""
    for attempt in range(max_retries):
        try:
            if method == "openai":
                return generate_qa_openai(chunk, api_key)
            elif method == "claude":
                return generate_qa_claude(chunk, api_key)
            elif method == "local":
                return generate_qa_local(chunk, model)
            elif method == "ollama":
                return generate_qa_ollama(chunk, model)
            elif method == "finetuned":
                return generate_qa_finetuned(chunk, model)
                
        except Exception as e:
            print(f"   ⚠️  Próba {attempt + 1}/{max_retries} nieudana: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"   ⏳ Oczekiwanie {wait_time}s przed kolejną próbą...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Wszystkie próby wyczerpane dla fragmentu")
                return []
    
    return []

def get_api_key(method: str, provided_key: str = None) -> str:
    """Pobierz API key z argumentów lub zmiennych środowiskowych"""
    if provided_key:
        return provided_key
    
    if method == "openai":
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            print("🔑 Używam OPENAI_API_KEY z .env")
            return env_key
    elif method == "claude":
        env_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if env_key:
            print("🔑 Używam ANTHROPIC_API_KEY z .env")
            return env_key
    
    return None

def show_examples():
    """Pokaż przykłady użycia"""
    print("""
🚀 PRZYKŁADY UŻYCIA create_qa_dataset.py
================================================

🤖 OPENAI (Zalecane - najwyższa jakość):
   python3 create_qa_dataset.py --method openai --max-chunks 10
   python3 create_qa_dataset.py --method openai --max-chunks 100 --output full_qa.json
   
   💡 Wymaga OPENAI_API_KEY w .env lub --api-key
   💰 Koszt: ~$0.01 za chunk (~$1 za 100 fragmentów)

🦾 CLAUDE (Alternatywa dla OpenAI):
   python3 create_qa_dataset.py --method claude --api-key sk-ant-your-key --max-chunks 10
   
   💡 Wymaga ANTHROPIC_API_KEY w .env lub --api-key
   💰 Koszt: ~$0.005 za chunk (tańsze od OpenAI)

🧠 FINE-TUNED MODEL (Najlepsze dla cybernetyki - po Phase 1):
   python3 create_qa_dataset.py --method finetuned --model-path ./bielik-cybernetyka-lora --max-chunks 50
   
   💡 Wymaga ukończonego fine-tuningu (Phase 1)
   💰 Koszt: Darmowe!
   🎯 Jakość: Doskonała dla cybernetyki

💻 LOCAL (HuggingFace models - wolne, ale darmowe):
   python3 create_qa_dataset.py --method local --model microsoft/Phi-3-mini-4k-instruct --max-chunks 5
   python3 create_qa_dataset.py --method local --model mistralai/Mistral-7B-Instruct-v0.1 --max-chunks 10
   
   💡 Pobiera model z HuggingFace (~3-7GB)
   💰 Koszt: Darmowe (po pobraniu)

🦙 OLLAMA (Jeśli masz Ollama setup):
   ollama pull phi3:mini                    # Najpierw zainstaluj model
   python3 create_qa_dataset.py --method ollama --model phi3:mini --max-chunks 10
   
   💡 Wymaga działającego Ollama
   💰 Koszt: Darmowe

🔄 KONTROLA POSTĘPU:
   python3 create_qa_dataset.py --method openai --resume                    # Wznów przerwane
   python3 create_qa_dataset.py --method openai --save-every 5 --max-chunks 100  # Zapisuj co 5

📊 PRZYKŁADOWE SKALE:
   --max-chunks 3      # Test (3 fragmenty, ~15 Q&A, ~$0.03)
   --max-chunks 50     # Mały dataset (~250 Q&A, ~$0.50)
   --max-chunks 200    # Średni dataset (~1000 Q&A, ~$2)
   --max-chunks 2180   # Pełny korpus (~10,000 Q&A, ~$20)

💡 PRO TIPS:
   - Zacznij od małej liczby chunks dla testów
   - OpenAI ma najwyższą jakość
   - Fine-tuned model będzie najlepszy dla cybernetyki
   - Użyj --resume jeśli proces się przerwie
""")

def main():
    parser = argparse.ArgumentParser(
        description="Generuj dataset Q&A z korpusu cybernetyki",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRZYKŁADY:
  %(prog)s --method openai --max-chunks 10
  %(prog)s --method finetuned --model-path ./bielik-cybernetyka-lora --max-chunks 50
  %(prog)s --method local --model microsoft/Phi-3-mini-4k-instruct --max-chunks 5
  %(prog)s --examples  # Pokaż szczegółowe przykłady
        """)
    
    parser.add_argument("--method", choices=["openai", "claude", "local", "ollama", "finetuned"],
                       help="Metoda generowania Q&A")
    parser.add_argument("--api-key", 
                       help="API key dla OpenAI/Claude (opcjonalnie z .env)")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct",
                       help="Model: HF model dla 'local', Ollama model dla 'ollama'")
    parser.add_argument("--model-path", 
                       help="Ścieżka do fine-tuned modelu (wymagana dla --method finetuned)")
    parser.add_argument("--max-chunks", type=int, default=50,
                       help="Maksymalna liczba fragmentów do przetworzenia (default: 50)")
    parser.add_argument("--output", default="cybernetics_qa_dataset.json",
                       help="Plik wyjściowy (default: cybernetics_qa_dataset.json)")
    parser.add_argument("--resume", action="store_true",
                       help="Automatycznie wznów z ostatniego postępu")
    parser.add_argument("--save-every", type=int, default=1,
                       help="Zapisuj postęp co N fragmentów (default: 1)")
    parser.add_argument("--examples", action="store_true",
                       help="Pokaż szczegółowe przykłady użycia")
    
    args = parser.parse_args()
    
    # Show examples if requested
    if args.examples:
        show_examples()
        return
    
    # Validate required arguments
    if not args.method:
        parser.print_help()
        print("\n❌ Wymagany argument --method")
        print("💡 Użyj --examples aby zobaczyć przykłady")
        return
    
    # Get API key from args or environment
    api_key = get_api_key(args.method, args.api_key)
    
    # Załaduj korpus
    chunks = load_corpus()
    
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
        print(f"🔢 Ograniczenie do {len(chunks)} fragmentów")
    
    # Załaduj poprzedni postęp
    if args.resume or Path(args.output.replace('.json', '_progress.json')).exists():
        all_qa_pairs, start_chunk = load_progress(args.output)
    else:
        all_qa_pairs = []
        start_chunk = 0
    
    # Sprawdź wymagania dla różnych metod
    if args.method in ["openai", "claude"] and not api_key:
        print(f"❌ Wymagany API key dla {args.method}")
        print(f"   Dodaj do .env: {'OPENAI_API_KEY' if args.method == 'openai' else 'ANTHROPIC_API_KEY'}=sk-...")
        print(f"   Lub użyj: --api-key YOUR_KEY")
        return
    
    if args.method == "finetuned" and not args.model_path:
        print(f"❌ Wymagana ścieżka do modelu dla --method finetuned")
        print(f"   Użyj: --model-path ./bielik-cybernetyka-lora")
        return
    
    if args.method == "ollama":
        print(f"💡 Sprawdź czy Ollama działa: ollama list")
        print(f"   Model: {args.model}")
    
    if args.method == "finetuned":
        print(f"🧠 Używam fine-tuned modelu z: {args.model_path}")
    
    print(f"🤖 Rozpoczynam generowanie Q&A metodą: {args.method}")
    print(f"📊 Zakres: fragment {start_chunk + 1} - {len(chunks)}")
    
    try:
        for i in range(start_chunk, len(chunks)):
            chunk = chunks[i]
            print(f"📝 Przetwarzam fragment {i+1}/{len(chunks)}...")
            
            # Generuj Q&A z retry logic
            model_param = args.model_path if args.method == "finetuned" else args.model
            qa_pairs = generate_qa_with_retry(
                chunk, 
                args.method, 
                api_key, 
                model_param
            )
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                print(f"   ✅ Wygenerowano {len(qa_pairs)} par Q&A")
            else:
                print(f"   ❌ Brak wyników dla fragmentu {i+1}")
            
            # Zapisz postęp
            if i % args.save_every == 0 or qa_pairs:
                save_progress(all_qa_pairs, args.output, i)
            
            # Small delay to respect rate limits
            if args.method in ["openai", "claude"]:
                time.sleep(2)  # Increased for safety
                
    except KeyboardInterrupt:
        print(f"\n⚠️  Przerwano przez użytkownika")
        print(f"💾 Zapisuję ostatni postęp...")
        save_progress(all_qa_pairs, args.output, i)
        print(f"📊 Zebrano {len(all_qa_pairs)} par Q&A przed przerwaniem")
        return
        
    except Exception as e:
        print(f"\n❌ Nieoczekiwany błąd: {e}")
        print(f"💾 Zapisuję ostatni postęp...")
        save_progress(all_qa_pairs, args.output, i)
        return
    
    # Final save and summary
    print(f"\n📊 PODSUMOWANIE:")
    processed_chunks = len(chunks) - start_chunk if 'start_chunk' in locals() else len(chunks)
    print(f"   Przetworzono fragmentów: {processed_chunks}/{len(chunks)}")
    print(f"   Wygenerowano par Q&A: {len(all_qa_pairs)}")
    
    if all_qa_pairs:
        # Convert to instruction format
        instruction_data = create_instruction_format(all_qa_pairs)
        
        # Save final results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Zapisano finalny dataset do: {args.output}")
        print(f"📈 Format instrukcyjny: {len(instruction_data)} przykładów")
        
        # Show sample
        if instruction_data:
            print(f"\n🔍 PRZYKŁAD:")
            sample = instruction_data[0]
            print(f"Instrukcja: {sample['instruction']}")
            print(f"Odpowiedź: {sample['output'][:200]}...")
        
        # Clean up progress files
        progress_file = args.output.replace('.json', '_progress.json')
        if Path(progress_file).exists():
            print(f"🧹 Usuwam pliki postępu...")
            Path(progress_file).unlink()
            
        print(f"\n🎉 ZAKOŃCZONO POMYŚLNIE!")
        print(f"📁 Pliki do usunięcia: *_backup_*.json (opcjonalnie)")
        
    else:
        print(f"❌ Brak danych Q&A do zapisania")
        
    # Final progress save if needed
    if all_qa_pairs and 'i' in locals():
        save_progress(all_qa_pairs, args.output, i)

if __name__ == "__main__":
    main() 