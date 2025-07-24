#!/usr/bin/env python3
"""
Fine-tuning Bielik-11B z korpusem cybernetyki przy użyciu Unsloth
Optimized dla RTX 3090 (24GB VRAM)
"""

import torch
from datasets import Dataset
import json
from pathlib import Path
import re

def check_environment():
    """Sprawdź czy środowisko jest gotowe"""
    print("🔧 Sprawdzam środowisko...")
    
    # Sprawdź CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA nie jest dostępne!")
        print("   Uruchom setup_environment.py aby skonfigurować")
        return False
    
    # Sprawdź wymagane biblioteki
    required_modules = ['unsloth', 'transformers', 'datasets', 'peft', 'trl']
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
    
    training_data = []
    
    for i, doc in enumerate(documents):
        if len(doc.strip()) < 100:  # Pomiń puste sekcje
            continue
            
        # Wyczyść dokument
        doc = doc.strip()
        
        # Usuń metadane nagłówka jeśli istnieją
        lines = doc.split('\n')
        content_start = 0
        for j, line in enumerate(lines):
            if 'DOKUMENT:' in line or 'AUTOR:' in line:
                content_start = j + 6  # Pomiń metadane
                break
        
        if content_start > 0:
            doc = '\n'.join(lines[content_start:])
        
        # Podziel na fragmenty o maksymalnej długości
        chunks = split_text_smart(doc, max_length)
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Pomiń za krótkie fragmenty
                training_data.append({
                    "text": chunk.strip(),
                    "input": "",  # Dla continued pre-training
                    "output": chunk.strip()
                })
    
    print(f"✅ Przygotowano {len(training_data)} fragmentów do trenowania")
    return training_data

def split_text_smart(text, max_length=2048):
    """Inteligentny podział tekstu na fragmenty"""
    
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_instruction_format(data):
    """Konwertuj do formatu instrukcyjnego dla Bielik"""
    
    formatted_data = []
    
    for item in data:
        text = item["text"]
        
        # Format dla Bielik (podobny do ChatML)
        formatted_text = f"""<|start_header_id|>system<|end_header_id|>

Jesteś ekspertem z zakresu cybernetyki społecznej i ogólnej. Odpowiadasz na podstawie polskiej szkoły cybernetycznej, w szczególności prac Józefa Kosseckiego i Mariana Mazura.<|eot_id|><|start_header_id|>user<|end_header_id|>

Kontynuuj lub wyjaśnij następujący tekst z zakresu cybernetyki: {text[:200]}...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{text}<|eot_id|>"""
        
        formatted_data.append({
            "text": formatted_text
        })
    
    return formatted_data

def main():
    """Główna funkcja fine-tuningu"""
    
    print("🚀 Fine-tuning Bielik-11B z korpusem cybernetyki")
    print("=" * 60)
    
    # Sprawdź środowisko
    if not check_environment():
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print("⚠️  Uwaga: Mało VRAM. Użyj QLoRA zamiast LoRA")
    
    # Import bibliotek
    from unsloth import FastLanguageModel
    
    # Przygotuj dane
    print("\n📚 Przygotowuję dane treningowe...")
    training_data = prepare_corpus_for_training()
    
    # Formatuj do instrukcji (opcjonalne)
    # training_data = create_instruction_format(training_data)
    
    # Utwórz dataset
    dataset = Dataset.from_list(training_data)
    
    print(f"📊 Dataset: {len(dataset)} przykładów")
    
    # Model config dla RTX 3090
    model_config = {
        "model_name": "speakleash/Bielik-11B-v2.3-Instruct",
        "max_seq_length": 2048,  # Zwiększ jeśli masz więcej VRAM
        "dtype": None,  # Auto-detect
        "load_in_4bit": True,  # QLoRA dla oszczędności VRAM
    }
    
    print(f"\n🤖 Ładuję model Bielik-11B...")
    model, tokenizer = FastLanguageModel.from_pretrained(**model_config)
    
    # Konfiguracja LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank - zwiększ dla lepszej jakości
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Training arguments
    training_args = {
        "per_device_train_batch_size": 2,  # Zwiększ jeśli możesz
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "num_train_epochs": 1,  # Zwiększ dla lepszych rezultatów
        "learning_rate": 2e-4,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "./outputs",
        "save_steps": 100,
        "save_total_limit": 3,
    }
    
    print(f"\n🏋️ Rozpoczynam trenowanie...")
    print(f"⚙️  Batch size: {training_args['per_device_train_batch_size']}")
    print(f"⚙️  Epochs: {training_args['num_train_epochs']}")
    print(f"⚙️  Learning rate: {training_args['learning_rate']}")
    
    # Oszacuj czas
    steps_per_epoch = len(dataset) // (training_args['per_device_train_batch_size'] * training_args['gradient_accumulation_steps'])
    total_steps = steps_per_epoch * training_args['num_train_epochs']
    estimated_hours = total_steps * 0.1 / 60  # ~0.1 min per step na RTX 3090
    
    print(f"⏱️  Szacowany czas: {estimated_hours:.1f} godzin")
    
    # Trenowanie
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(**training_args),
    )
    
    # Start training
    trainer_stats = trainer.train()
    
    # Zapisz model
    print(f"\n💾 Zapisuję model...")
    model.save_pretrained("bielik-cybernetyka-lora")
    tokenizer.save_pretrained("bielik-cybernetyka-lora")
    
    # Zapisz jako GGUF dla Ollama (opcjonalne)
    print(f"\n📦 Eksportuję do GGUF...")
    model.save_pretrained_gguf("bielik-cybernetyka", tokenizer, quantization_method="q4_k_m")
    
    print(f"\n🎉 Fine-tuning zakończony!")
    print(f"📂 Model zapisany w: ./bielik-cybernetyka-lora")
    print(f"📂 GGUF zapisany w: ./bielik-cybernetyka")
    
    return trainer_stats

if __name__ == "__main__":
    # Uruchom fine-tuning
    stats = main()
    
    print(f"\n📊 STATYSTYKI TRENOWANIA:")
    print(f"⏱️  Czas trenowania: {stats.metrics['train_runtime']:.1f} sekund")
    print(f"📈 Loss: {stats.metrics['train_loss']:.4f}")
    print(f"🚀 Samples/second: {stats.metrics['train_samples_per_second']:.2f}") 