#!/usr/bin/env python3
"""
Test wytrenowanego modelu Bielik-11B + Cybernetyka
"""

def test_model():
    """Testuj wytrenowany model"""
    
    # Sprawdź środowisko
    try:
        import torch
        from unsloth import FastLanguageModel
        
        if not torch.cuda.is_available():
            print("❌ CUDA nie jest dostępne!")
            return
            
    except ImportError as e:
        print(f"❌ Brak wymaganej biblioteki: {e}")
        print("   Uruchom: python3 setup_environment.py")
        return
    
    print("🧪 Testowanie modelu Bielik-Cybernetyka")
    print("=" * 50)
    
    # Załaduj model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./bielik-cybernetyka-lora",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Przygotuj do inferencji
    FastLanguageModel.for_inference(model)
    
    # Pytania testowe
    test_questions = [
        "Co to jest cybernetyka społeczna?",
        "Jak Mazur definiuje system autonomiczny?",
        "Jakie są podstawowe zasady cybernetyki?",
        "Czym różni się cybernetyka od informatyki?",
        "Co to jest sprzężenie zwrotne w cybernetyce?",
        "Jak działają systemy sterowania społecznego?",
        "Co to jest teoria układów samodzielnych?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔬 Test {i}: {question}")
        print("-" * 40)
        
        # Format dla Bielik
        prompt = f"""<|start_header_id|>system<|end_header_id|>

Jesteś ekspertem z zakresu cybernetyki społecznej i ogólnej. Odpowiadasz na podstawie polskiej szkoły cybernetycznej, w szczególności prac Józefa Kosseckiego i Mariana Mazura.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenizuj
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generuj odpowiedź
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Dekoduj odpowiedź
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Wyciągnij tylko odpowiedź asystenta
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            answer = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        
        print(f"🤖 Odpowiedź: {answer}")
        print()

if __name__ == "__main__":
    test_model() 