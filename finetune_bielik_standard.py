#!/usr/bin/env python3
"""
Fine-tuning Bielik-11B z korpusem cybernetyki bez Unsloth
Używa standardowych bibliotek: transformers + peft + trl
Optimized dla RTX 3090 (24GB VRAM)
"""

import torch
from datasets import Dataset
import json
from pathlib import Path
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os

def check_environment():
    """Sprawdź czy środowisko jest gotowe"""
    print("🔧 Sprawdzam środowisko...")
    
    # Sprawdź CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA nie jest dostępne!")
        return False
    
    # Sprawdź wymagane biblioteki
    required_modules = ['transformers', 'datasets', 'peft', 'trl', 'bitsandbytes']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Brakujące biblioteki: {', '.join(missing)}")
        print("   Uruchom: python3 setup_environment.py")
        return False
    
    print("✅ Środowisko gotowe do fine-tuningu")
    return True

def prepare_corpus_for_training(corpus_file="cybernetyka_corpus.txt", max_length=2048):
    """Przygotuj korpus do trenowania"""
    
    print("📚 Przygotowuję korpus do trenowania...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Podziel na dokumenty
    documents = re.split(r'={80}', full_text)
    documents = [doc.strip() for doc in documents if len(doc.strip()) > 100]
    
    print(f"📄 Znaleziono {len(documents)} dokumentów")
    
    # Podziel długie dokumenty na fragmenty
    text_chunks = []
    
    for doc in documents:
        # Podziel na akapity
        paragraphs = doc.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Sprawdź czy dodanie akapitu nie przekroczy limitu
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                # Zapisz aktualny chunk i zacznij nowy
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                current_chunk = para
        
        # Dodaj ostatni chunk
        if current_chunk:
            text_chunks.append(current_chunk.strip())
    
    # Przygotuj dane w formacie dla SFTTrainer
    training_data = []
    for chunk in text_chunks:
        if len(chunk) > 50:  # Filtruj bardzo krótkie teksty
            training_data.append({
                "text": chunk
            })
    
    print(f"✅ Przygotowano {len(training_data)} przykładów treningowych")
    return training_data

def main():
    """Główna funkcja fine-tuningu"""
    
    print("🚀 Fine-tuning Bielik-11B z korpusem cybernetyki (Standard)")
    print("=" * 60)
    
    # Sprawdź środowisko
    if not check_environment():
        return {"status": "failed", "reason": "environment"}
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    
    # Konfiguracja modelu
    model_name = "speakleash/Bielik-11B-v2.2-Instruct"
    
    print(f"\n🤖 Ładuję model: {model_name}")
    
    # Konfiguracja BitsAndBytes dla 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Załaduj tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Załaduj model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Konfiguracja LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Zastosuj LoRA do modelu
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Przygotuj dane
    print("\n📚 Przygotowuję dane treningowe...")
    training_data = prepare_corpus_for_training()
    
    # Utwórz dataset
    dataset = Dataset.from_list(training_data)
    
    # Podziel na train/eval
    train_size = int(0.95 * len(dataset))
    dataset = dataset.train_test_split(train_size=train_size, seed=42)
    
    # Konfiguracja trenowania
    output_dir = "./bielik-cybernetyka-lora"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        save_strategy="epoch",
        logging_steps=10,
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True
    )
    
    # Utwórz trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=2048,
        packing=False
    )
    
    # Uruchom trenowanie
    print("\n🔥 Rozpoczynam trenowanie...")
    trainer.train()
    
    # Zapisz model
    print("\n💾 Zapisuję model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n🎉 Fine-tuning zakończony!")
    print(f"📂 Model zapisany w: {output_dir}")
    
    return {
        "status": "success",
        "output_dir": output_dir,
        "training_samples": len(training_data)
    }

if __name__ == "__main__":
    stats = main()
    print(f"\n📊 Status: {stats}") 