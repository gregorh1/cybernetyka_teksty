#!/usr/bin/env python3
"""
Testowanie konwersacji z wytrenowanym modelem cybernetyki
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch
import threading
import time

class CustomStreamer:
    """Custom streamer for real-time token generation"""
    def __init__(self, tokenizer, skip_prompt=True):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.generated_text = ""
        self.printed_length = 0
        
    def put(self, value):
        """Called for each generated token"""
        if len(value.shape) > 1:
            value = value[0]  # Take first batch
            
        # Decode full sequence
        try:
            full_text = self.tokenizer.decode(value, skip_special_tokens=True)
            
            # Extract only new text
            if len(full_text) > self.printed_length:
                new_text = full_text[self.printed_length:]
                print(new_text, end='', flush=True)
                self.printed_length = len(full_text)
                
        except Exception as e:
            # Fallback: try single token decode
            try:
                if len(value) > 0:
                    last_token = value[-1].item()
                    if last_token not in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                        text = self.tokenizer.decode([last_token], skip_special_tokens=True)
                        if text and text not in ['\n', '\r', ' ']:
                            print(text, end='', flush=True)
            except:
                pass
    
    def end(self):
        """Called when generation ends"""
        print()  # New line at the end

def load_model(model_path="./bielik-cybernetyka-lora"):
    """Załaduj model do konwersacji"""
    print(f"🔄 Ładuję model z: {model_path}")
    
    base_model_name = "speakleash/Bielik-11B-v2.2-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    print("✅ Model załadowany do konwersacji")
    return model, tokenizer

def chat_with_model(model, tokenizer, use_streaming=True):
    """Interaktywna konwersacja z modelem"""
    print("\n🤖 KONWERSACJA Z MODELEM CYBERNETYKI")
    print("=" * 50)
    print("Wpisz pytania o cybernetykę (lub 'quit' aby zakończyć)")
    print("Przykłady:")
    print("- Wyjaśnij sprzężenie zwrotne")
    print("- Co to jest homeostaza?")
    print("- Jak cybernetyka odnosi się do społeczeństwa?")
    print()
    
    while True:
        user_input = input("\n🔵 Ty: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'koniec']:
            print("👋 Do widzenia!")
            break
        
        if not user_input:
            continue
        
        # Create prompt
        prompt = f"Pytanie: {user_input}\n\nOdpowiedź:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generation with optional streaming
        print("🔄 Generuję odpowiedź...")
        
        # Check tokenizer tokens
        if tokenizer.eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id
        else:
            eos_token_id = tokenizer.eos_token_id
        
        if use_streaming:
            try:
                # Method 1: Try with TextStreamer (built-in)
                from transformers import TextStreamer
                print("\n🤖 Model: ", end='', flush=True)
                
                # Use built-in TextStreamer
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1500,  # ~1000 words for educational content
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=eos_token_id,
                        repetition_penalty=1.1,
                        top_p=0.9,
                        no_repeat_ngram_size=3,
                        early_stopping=False,
                        streamer=streamer
                    )
                    
            except Exception as e:
                print(f"\n⚠️  Streaming error: {e}")
                use_streaming = False  # Fall back to non-streaming
        
        if not use_streaming:
            # Non-streaming generation
            print("\n🤖 Model: ", end='', flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    early_stopping=False
                )
                
            # Show result all at once
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            print(response)
        
        # Get final response for word count
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        print(f"\n📏 Długość odpowiedzi: {len(response.split())} słów")

def test_specific_scenarios(model, tokenizer):
    """Test konkretnych scenariuszy"""
    scenarios = [
        "Wyjaśnij szczegółowo koncepcję sprzężenia zwrotnego w cybernetyce",
        "Opisz różnice między systemami otwartymi a zamkniętymi", 
        "Jak cybernetyka społeczna odnosi się do zarządzania organizacjami?",
        "Co oznacza homeostaza w kontekście systemów społecznych?",
        "Jakie są główne zasady teorii informacji w cybernetyce?"
    ]
    
    print("\n🧪 TEST SCENARIUSZY KONWERSACYJNYCH")
    print("=" * 50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario}")
        
        prompt = f"Pytanie: {scenario}\n\nOdpowiedź:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Check tokenizer tokens
            if tokenizer.eos_token_id is None:
                eos_token_id = tokenizer.pad_token_id
            else:
                eos_token_id = tokenizer.eos_token_id
                
            outputs = model.generate(
                **inputs,
                max_new_tokens=1200,  # Educational content
                temperature=0.7,
                do_sample=True,
                pad_token_id=eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                no_repeat_ngram_size=3,
                early_stopping=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        print(f"Odpowiedź: {response}")
        print(f"Długość: {len(response.split())} słów")
        print("-" * 30)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test konwersacji z modelem")
    parser.add_argument("--model-path", default="./bielik-cybernetyka-lora",
                       help="Ścieżka do modelu")
    parser.add_argument("--interactive", action="store_true",
                       help="Tryb interaktywny")
    parser.add_argument("--test-scenarios", action="store_true",
                       help="Test predefiniowanych scenariuszy")
    parser.add_argument("--no-streaming", action="store_true",
                       help="Wyłącz streaming (pokazuj całą odpowiedź na raz)")
    parser.add_argument("--debug", action="store_true",
                       help="Tryb debug z dodatkowymi informacjami")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Determine streaming mode
    use_streaming = not args.no_streaming
    
    if args.debug:
        print(f"🔧 Debug mode: streaming={use_streaming}")
        print(f"   Tokenizer EOS: {tokenizer.eos_token_id}")
        print(f"   Tokenizer PAD: {tokenizer.pad_token_id}")
    
    if args.test_scenarios:
        test_specific_scenarios(model, tokenizer)
    
    if args.interactive or not args.test_scenarios:
        chat_with_model(model, tokenizer, use_streaming)

if __name__ == "__main__":
    main() 