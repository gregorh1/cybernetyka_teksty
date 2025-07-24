# 🚀 Fine-tuning Bielik-11B z Korpusem Cybernetyki

Przewodnik krok po kroku trenowania polskiego modelu AI na korpusie cybernetyki (Kossecki + Mazur).

## 📋 Wymagania

### Hardware:
- **RTX 3090** (24GB VRAM) ✅
- **32GB+ RAM** (zalecane)
- **50GB+ miejsca** na dysku

### Software:
- **Python 3.10+**
- **CUDA 12.1+**
- **PyTorch 2.1+**

## ⚡ SZYBKI START

### 1. Instalacja zależności
```bash
# Aktywuj environment
pyenv activate cybernetyka-corpus

# Zainstaluj pakiety
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install xformers trl peft accelerate bitsandbytes datasets
```

### 2. Uruchom fine-tuning
```bash
# Sprawdź czy korpus istnieje
ls -la cybernetyka_corpus.txt

# Uruchom trenowanie (2-6 godzin na RTX 3090)
python3 finetune_bielik.py
```

### 3. Testuj model
```bash
# Po zakończeniu trenowania
python3 test_finetuned_model.py
```

## 🔧 OPCJE KONFIGURACJI

### Dla większej jakości (więcej czasu):
Edytuj `finetune_bielik.py`:
```python
training_args = {
    "num_train_epochs": 3,         # Zwiększ z 1 do 3
    "learning_rate": 1e-4,         # Zmniejsz learning rate
    "per_device_train_batch_size": 1,  # Zmniejsz batch size
    "r": 32,                       # Zwiększ LoRA rank w get_peft_model()
}
```

### Dla szybszego trenowania:
```python
training_args = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,  # Zwiększ jeśli masz VRAM
    "max_seq_length": 1024,            # Zmniejsz długość sekwencji
}
```

## ⏱️ CZASY TRENOWANIA na RTX 3090

| Konfiguracja | Epochs | Batch Size | Czas | Jakość |
|--------------|--------|------------|------|--------|
| **Szybka**   | 1      | 4          | 2h   | Dobra  |
| **Standard** | 2      | 2          | 4h   | Bardzo dobra |
| **Najwyższa**| 3      | 1          | 8h   | Doskonała |

## 📊 MONITOROWANIE

### GPU Usage:
```bash
watch nvidia-smi
```

### Logi trenowania:
```bash
tail -f outputs/training.log
```

### TensorBoard (opcjonalne):
```bash
pip install tensorboard
tensorboard --logdir outputs/runs
```

## 🎯 TYPY FINE-TUNINGU

### 1. LoRA (Recommended) ⭐
- **Szybko:** 2-6h
- **VRAM:** 12-16GB
- **Pliki:** ~1GB
- **Jakość:** 95% full fine-tuning

### 2. QLoRA (Oszczędne)
- **Szybko:** 2-8h  
- **VRAM:** 8-12GB
- **Pliki:** ~500MB
- **Jakość:** 90% full fine-tuning

### 3. Full Fine-tuning (Nie dla RTX 3090)
- **VRAM:** 40GB+
- **Jakość:** 100%

## 🔍 TROUBLESHOOTING

### CUDA Out of Memory:
```python
# Zmniejsz batch size
"per_device_train_batch_size": 1

# Lub zmniejsz max_seq_length  
"max_seq_length": 1024
```

### Wolne trenowanie:
```python
# Zwiększ batch size (jeśli masz VRAM)
"per_device_train_batch_size": 4

# Lub zainstaluj Flash Attention
pip install flash-attn --no-build-isolation
```

### Błędy instalacji Unsloth:
```bash
# Alternatywne źródło
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Lub użyj conda
conda install unsloth -c conda-forge
```

## 📁 STRUKTURA PLIKÓW

```
cybernetyka_teksty/
├── cybernetyka_corpus.txt          # Korpus treningowy (11MB)
├── finetune_bielik.py              # Skrypt trenowania  
├── test_finetuned_model.py         # Testy modelu
├── bielik-cybernetyka-lora/        # Wytrenowany model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors   # LoRA weights (~1GB)
│   └── tokenizer.json
├── bielik-cybernetyka/             # GGUF dla Ollama
│   └── model-q4_k_m.gguf          # Kvantyzowany model
└── outputs/                        # Logi i checkpointy
    ├── checkpoint-100/
    └── training.log
```

## 🧪 TESTOWANIE MODELU

### Automatyczne testy:
```bash
python3 test_finetuned_model.py
```

### Ręczne testy:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("./bielik-cybernetyka-lora")
FastLanguageModel.for_inference(model)

prompt = "Co to jest cybernetyka społeczna?"
# ... generuj odpowiedź
```

## 🚀 UŻYWANIE W OLLAMA

```bash
# Importuj wytrenowany model
ollama create bielik-cybernetyka -f bielik-cybernetyka/model-q4_k_m.gguf

# Testuj
ollama run bielik-cybernetyka "Wyjaśnij teorię układów samodzielnych Mazura"
```

## 📈 METRYKI SUKCESU

### Dobre trenowanie:
- **Loss:** Spada z ~2.0 do ~0.8
- **Perplexity:** < 3.0 
- **GPU Utilization:** 90%+
- **No OOM errors**

### Jakość odpowiedzi:
- Używa terminologii cybernetycznej
- Odwołuje się do Kosseckiego/Mazura
- Spójne z polską szkołą cybernetyczną
- Po polsku, bez błędów

## 🎯 DALSZE KROKI

1. **Zwiększ epochs** dla lepszej jakości
2. **Dodaj instruction tuning** z Q&A
3. **Kvantyzuj** do Q4_K_M dla Ollama
4. **Testuj** na rzeczywistych zadaniach
5. **Iteruj** konfigurację

---

**💡 Pro Tip:** Zapisuj checkpointy co 100 kroków. Fine-tuning można wznowić w przypadku błędu!

**🎉 Po trenowaniu masz specjalistyczny polski model AI w cybernetyce!** 