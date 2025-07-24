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
- **PyTorch 2.4+**

## 🔧 METODY FINE-TUNINGU

Dostępne są **dwie metody** fine-tuningu:

### 🦥 **UNSLOTH** (Zalecane jeśli działa)
- ✅ **2x szybsze** trenowanie (~2-4h)
- ✅ **Mniejsze zużycie VRAM** (może 8-bit)
- ✅ **Wbudowane optymalizacje** (Flash Attention)
- ❌ **Problemy z kompatybilnością** PyTorch/CUDA
- ❌ **Niestabilne** z nowymi wersjami

### 🔧 **STANDARD** (Zawsze działa)
- ✅ **Wysoka kompatybilność** wszystkich wersji
- ✅ **Stabilne i niezawodne**
- ✅ **Łatwiejsze debugowanie**
- ❌ **Wolniejsze trenowanie** (~3-6h)
- ❌ **Wymaga 4-bit quantization**

## 🚀 **AUTOMATYCZNY START**

```bash
# Uruchom skrypt - wybierze najlepszą metodę automatycznie
./run_finetune.sh
```

Skrypt:
1. ✅ Sprawdzi środowisko
2. 🧪 Przetestuje dostępność Unsloth
3. 🤖 Pozwoli wybrać metodę (jeśli obie działają)
4. 🔥 Uruchomi trenowanie
5. 🧪 Przetestuje wytrenowany model

## ⚡ SZYBKI START - METODA UNSLOTH

### 1. Instalacja zależności
```bash
# Aktywuj environment
pyenv activate cybernetyka-corpus

# Zainstaluj pakiety dla Unsloth
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets accelerate peft trl bitsandbytes
pip install flash-attn --no-build-isolation
```

### 2. Uruchom fine-tuning
```bash
python3 finetune_bielik.py
```

## 🔧 SZYBKI START - METODA STANDARD

### 1. Instalacja zależności
```bash
# Aktywuj environment
pyenv activate cybernetyka-corpus

# Zainstaluj pakiety dla Standard
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl bitsandbytes
pip install "numpy<2"  # Kompatybilność
```

### 2. Uruchom fine-tuning
```bash
python3 finetune_bielik_standard.py
```

## 📊 PORÓWNANIE METOD

| Aspekt | 🦥 Unsloth | 🔧 Standard |
|--------|-----------|------------|
| **Czas trenowania** | 2-4h | 3-6h |
| **VRAM na RTX 3090** | 16-20GB | 22-24GB |
| **Kompatybilność** | Problematyczna | Wysoka |
| **Stabilność** | Niestabilna | Bardzo stabilna |
| **Jakość modelu** | Identyczna | Identyczna |
| **Quantization** | 8-bit możliwa | 4-bit wymagana |
| **Debugowanie** | Trudne | Łatwe |
| **Dla produkcji** | Ryzykowne | Zalecane |

## 📁 PLIKI SKRYPTÓW

```
cybernetyka_teksty/
├── run_finetune.sh                 # 🚀 Główny skrypt (AUTO-WYBÓR)
├── finetune_bielik.py              # 🦥 Unsloth (szybkie)
├── finetune_bielik_standard.py     # 🔧 Standard (stabilne)
├── test_finetuned_model.py         # 🧪 Testy modelu
└── setup_environment.py            # 📦 Setup pakietów
```

## ⏱️ CZASY TRENOWANIA na RTX 3090

### 🦥 Unsloth:
| Konfiguracja | Epochs | Batch Size | VRAM | Czas |
|--------------|--------|------------|------|------|
| **Szybka**   | 1      | 4          | 16GB | 1.5h |
| **Standard** | 2      | 2          | 18GB | 3h   |
| **Najwyższa**| 3      | 1          | 20GB | 4.5h |

### 🔧 Standard:
| Konfiguracja | Epochs | Batch Size | VRAM | Czas |
|--------------|--------|------------|------|------|
| **Szybka**   | 1      | 2          | 22GB | 3h   |
| **Standard** | 2      | 1          | 23GB | 5h   |
| **Najwyższa**| 3      | 1          | 24GB | 7.5h |

## 🔧 OPCJE KONFIGURACJI

### Dla większej jakości (więcej czasu):
```python
# W obu skryptach
training_args = {
    "num_train_epochs": 3,         # Zwiększ z 1 do 3
    "learning_rate": 1e-4,         # Zmniejsz learning rate
    "per_device_train_batch_size": 1,  # Zmniejsz batch size
}

# W LoRA config
lora_config = {
    "r": 32,                       # Zwiększ LoRA rank
    "lora_alpha": 64,              # Zwiększ alpha
}
```

### Dla szybszego trenowania:
```python
training_args = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,  # Zwiększ jeśli masz VRAM
    "max_seq_length": 1024,            # Zmniejsz długość
}
```

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

## 🔍 TROUBLESHOOTING

### ❌ Unsloth nie działa:
```bash
# Użyj Standard
python3 finetune_bielik_standard.py

# Lub napraw wersje PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### ❌ CUDA Out of Memory:
```python
# Zmniejsz batch size
"per_device_train_batch_size": 1

# Zmniejsz max_seq_length  
"max_seq_length": 1024

# Użyj gradient checkpointing
"gradient_checkpointing": True
```

### ❌ Flash Attention błędy:
```bash
# Reinstall dla odpowiedniej wersji PyTorch
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### ❌ NumPy incompatibility:
```bash
pip install "numpy<2"
```

## 📁 STRUKTURA PLIKÓW PO TRENOWANIU

```
cybernetyka_teksty/
├── cybernetyka_corpus.txt          # Korpus treningowy (11MB)
├── bielik-cybernetyka-lora/        # Wytrenowany model
│   ├── adapter_config.json         # Konfiguracja LoRA
│   ├── adapter_model.safetensors   # LoRA weights (~1GB)
│   ├── tokenizer.json              # Tokenizer
│   └── training_args.bin           # Argumenty trenowania
├── outputs/                        # Logi i checkpointy
│   ├── checkpoint-100/             # Checkpointy
│   ├── training.log                # Logi trenowania
│   └── runs/                       # TensorBoard logs
└── bielik-cybernetyka/             # GGUF dla Ollama (opcjonalne)
    └── model-q4_k_m.gguf          # Kvantyzowany model
```

## 🧪 TESTOWANIE MODELU

### Automatyczne testy:
```bash
python3 test_finetuned_model.py
```

### Ręczne testy - Unsloth:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("./bielik-cybernetyka-lora")
FastLanguageModel.for_inference(model)

prompt = "Co to jest cybernetyka społeczna?"
# ... generuj odpowiedź
```

### Ręczne testy - Standard:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("speakleash/Bielik-11B-v2.2-Instruct")
model = PeftModel.from_pretrained(base_model, "./bielik-cybernetyka-lora")
tokenizer = AutoTokenizer.from_pretrained("./bielik-cybernetyka-lora")

prompt = "Co to jest cybernetyka społeczna?"
# ... generuj odpowiedź
```

## 🚀 UŻYWANIE W OLLAMA

```bash
# Importuj wytrenowany model (wymaga konwersji do GGUF)
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

## 🎯 KTÓRA METODA WYBRAĆ?

### Wybierz **🦥 Unsloth** jeśli:
- ✅ Chcesz szybkie trenowanie
- ✅ Masz dużo czasu na debugowanie
- ✅ Experimentujesz z parametrami
- ✅ Nie jest to środowisko produkcyjne

### Wybierz **🔧 Standard** jeśli:
- ✅ Potrzebujesz stabilności
- ✅ To środowisko produkcyjne
- ✅ Masz ograniczony czas na troubleshooting
- ✅ Unsloth nie działa na twoim systemie

## 🎯 DALSZE KROKI

1. **Zwiększ epochs** dla lepszej jakości
2. **Dodaj instruction tuning** z Q&A
3. **Kvantyzuj** do Q4_K_M dla Ollama
4. **Testuj** na rzeczywistych zadaniach
5. **Iteruj** konfigurację

---

**💡 Pro Tip:** Skrypt `run_finetune.sh` automatycznie wybierze najlepszą metodę dla twojego systemu!

**🎉 Po trenowaniu masz specjalistyczny polski model AI w cybernetyce!** 