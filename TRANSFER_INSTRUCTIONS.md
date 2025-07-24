# 📦 TRANSFER NA MASZYNĘ Z GPU - INSTRUKCJE

Przewodnik przeniesienia korpusu cybernetyki i uruchomienia fine-tuningu na maszynie z RTX 3090.

## 📋 PLIKI DO PRZENIESIENIA

### ✅ WYMAGANE (MINIMUM):
```bash
cybernetyka_corpus.txt           # 11MB - korpus treningowy
setup_environment.py             # Setup środowiska
finetune_bielik.py              # Skrypt fine-tuningu  
test_finetuned_model.py         # Testy modelu
run_finetune.sh                 # Automatyczny launcher
README_FINETUNE.md              # Dokumentacja
```

### 📁 OPCJONALNE (dla pełnego archiwum):
```bash
Kossecki/                       # Wszystkie pliki źródłowe
Mazur/                          
cybernetyka_corpus_metadata.json
*.py                           # Wszystkie skrypty
```

## 🚀 SZYBKI START na nowej maszynie

### 1. Skopiuj pliki
```bash
# Utwórz folder projektu
mkdir cybernetyka-finetune
cd cybernetyka-finetune

# Skopiuj wymagane pliki (scp, rsync, USB, itp.)
# Minimum: korpus + skrypty Python + launcher
```

### 2. Setup środowiska
```bash
# Stwórz pyenv environment
pyenv install 3.11.7  # jeśli nie ma
pyenv virtualenv 3.11.7 cybernetyka-finetune
pyenv activate cybernetyka-finetune

# ALBO użyj virtualenv/conda
python3 -m venv cybernetyka-env
source cybernetyka-env/bin/activate
```

### 3. Automatyczny setup i fine-tuning
```bash
# OPCJA A: Wszystko automatycznie
chmod +x run_finetune.sh
./run_finetune.sh

# OPCJA B: Krok po kroku
python3 setup_environment.py        # Setup (5-15 min)
python3 finetune_bielik.py          # Fine-tuning (2-6h)
python3 test_finetuned_model.py     # Test modelu
```

## ⚙️ WYMAGANIA SPRZĘTOWE

### MINIMUM:
- **GPU:** RTX 3090 24GB ✅
- **RAM:** 16GB+ 
- **Dysk:** 50GB wolne
- **CUDA:** 11.8+ lub 12.1+

### ZALECANE:
- **RAM:** 32GB+
- **Dysk:** SSD + 100GB 
- **CPU:** 8+ cores

## 🔧 TROUBLESHOOTING

### Problem: CUDA not available
```bash
# Sprawdź sterowniki
nvidia-smi

# Zainstaluj CUDA (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Arch Linux
sudo pacman -S cuda
```

### Problem: PyTorch bez CUDA
```bash
# Odinstaluj i zainstaluj z CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problem: Out of Memory
```bash
# Edytuj finetune_bielik.py:
"per_device_train_batch_size": 1    # Zmniejsz z 2 do 1
"max_seq_length": 1024              # Zmniejsz z 2048
```

### Problem: Błędy instalacji Unsloth
```bash
# Alternatywa 1:
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"

# Alternatywa 2: 
conda install unsloth -c conda-forge

# Alternatywa 3: bez Unsloth (wolniej)
# Zakomentuj Unsloth imports, użyj standardowego HuggingFace
```

## 📊 MONITORING

### Podczas trenowania:
```bash
# Terminal 1: Trenowanie
python3 finetune_bielik.py

# Terminal 2: Monitor GPU
watch nvidia-smi

# Terminal 3: Logi
tail -f outputs/training.log
```

### Metryki sukcesu:
- **GPU Utilization:** 90%+
- **VRAM Usage:** ~16GB/24GB
- **Training Loss:** spada z ~2.0 do ~0.8
- **Temperature:** <85°C

## 📁 STRUKTURA WYNIKÓW

Po fine-tuningu otrzymasz:
```
cybernetyka-finetune/
├── bielik-cybernetyka-lora/        # LoRA model (~1GB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer_config.json
├── bielik-cybernetyka/             # GGUF model dla Ollama
│   └── model-q4_k_m.gguf
├── outputs/                        # Logi i checkpointy
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── training.log
└── requirements.txt                # Lista pakietów
```

## 🎯 SPRAWDZENIE JAKOŚCI

### Test podstawowy:
```python
python3 test_finetuned_model.py
```

### Test manualny:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("./bielik-cybernetyka-lora")
FastLanguageModel.for_inference(model)

# Zadaj pytanie o cybernetykę...
```

### Oczekiwane wyniki:
- ✅ Odpowiada po polsku
- ✅ Używa terminologii cybernetycznej  
- ✅ Odwołuje się do Kosseckiego/Mazura
- ✅ Spójne z polską szkołą cybernetyczną

## 🚀 UŻYCIE MODELU

### W kodzie Python:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("./bielik-cybernetyka-lora")
# ... generuj odpowiedzi
```

### W Ollama:
```bash
ollama create bielik-cybernetyka -f bielik-cybernetyka/model-q4_k_m.gguf
ollama run bielik-cybernetyka "Wyjaśnij teorię układów samodzielnych"
```

### W OpenWebUI + lokalny model:
1. Import modelu do Ollama
2. Wybierz w OpenWebUI
3. Dodaj RAG z korpusem

## 💡 OPTYMALIZACJE

### Dla lepszej jakości:
```python
# W finetune_bielik.py zmień:
"num_train_epochs": 3              # Więcej epochs
"r": 32                           # Większy LoRA rank  
"learning_rate": 1e-4             # Mniejszy LR
```

### Dla szybszego trenowania:
```python
"per_device_train_batch_size": 4   # Większy batch
"max_seq_length": 1024            # Krótsze sekwencje
"num_train_epochs": 1             # Mniej epochs
```

---

## 🎉 SUKCES!

Po zakończeniu będziesz miał:
- ✅ Specjalistyczny polski model cybernetyki
- ✅ Wiedzę z 66 dokumentów (1.55M słów)
- ✅ Ekspert w pracach Kosseckiego i Mazura
- ✅ Model gotowy do produkcji

**Czas trenowania na RTX 3090: 2-6 godzin**  
**Końcowy rozmiar: ~2GB (LoRA) + ~6GB (GGUF)**

💪 **Pierwszy polski specjalistyczny model AI w cybernetyce!** 🇵🇱🤖 