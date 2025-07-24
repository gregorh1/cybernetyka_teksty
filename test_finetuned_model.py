#!/usr/bin/env python3
"""
Test wytrenowanego modelu Bielik-11B + Cybernetyka
"""

def test_model():
    """Testuj wytrenowany model"""
    
    # Sprawdź środowisko
    try:
        import torch
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        if not torch.cuda.is_available():
            print("❌ CUDA nie jest dostępne!")
            return
            
    except ImportError as e:
        print(f"❌ Brak wymaganej biblioteki: {e}")
        print("   Uruchom: python3 setup_environment.py")
        return
    
    print("🧪 Testowanie modelu Bielik-Cybernetyka")
    print("=" * 50)
    
    # METODA PEFT: Załaduj adapter config, potem bazowy model, potem zastosuj adapter
    print("📥 Ładowanie konfiguracji LoRA adaptera...")
    adapter_config = PeftConfig.from_pretrained("./bielik-cybernetyka-lora")
    
    print("📥 Ładowanie bazowego modelu Bielik-11B...")
    
    # Konfiguracja 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path,  # speakleash/Bielik-11B-v2.2-Instruct
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    
    print("📥 Ładowanie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained("./bielik-cybernetyka-lora")
    
    # Upewnij się, że tokenizer ma pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔌 Stosowanie LoRA adaptera...")
    model = PeftModel.from_pretrained(base_model, "./bielik-cybernetyka-lora")
    
    print("✅ Model gotowy do testów!")
    
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
        
        # Generuj odpowiedź z bardziej konserwatywymi parametrami
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.3,  # Niższa temperatura = bardziej deterministyczne
                top_p=0.8,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
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