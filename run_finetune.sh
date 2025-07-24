#!/bin/bash

# 🚀 AUTOMATYCZNY FINE-TUNING BIELIK-11B + CYBERNETYKA
# Kompletny skrypt uruchamiający wszystko w odpowiedniej kolejności

echo "🚀 FINE-TUNING BIELIK-11B z KORPUSEM CYBERNETYKI"
echo "=================================================="

# Sprawdź czy jesteśmy w odpowiednim folderze
if [ ! -f "cybernetyka_corpus.txt" ]; then
    echo "❌ Brak pliku cybernetyka_corpus.txt"
    echo "   Uruchom skrypt w folderze z korpusem!"
    exit 1
fi

# Sprawdź Python
echo "🐍 Sprawdzam Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 nie znaleziony!"
    exit 1
fi

echo "✅ Python OK"

# Sprawdź NVIDIA
echo "🎮 Sprawdzam NVIDIA GPU..."
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ NVIDIA GPU lub sterowniki nie znalezione!"
    echo "   Zainstaluj sterowniki NVIDIA i CUDA"
    exit 1
fi

echo "✅ NVIDIA GPU wykryte"

# Krok 1: Setup środowiska
echo ""
echo "📦 KROK 1: Setup środowiska"
echo "----------------------------"
python3 setup_environment.py

if [ $? -ne 0 ]; then
    echo "❌ Setup środowiska nieudany!"
    echo "   Sprawdź błędy i uruchom ponownie"
    exit 1
fi

echo "✅ Środowisko skonfigurowane"

# Krok 2: Sprawdź korpus
echo ""
echo "📚 KROK 2: Sprawdzam korpus"
echo "---------------------------"
CORPUS_SIZE=$(wc -c < cybernetyka_corpus.txt)
CORPUS_MB=$(echo "scale=1; $CORPUS_SIZE / 1024 / 1024" | bc)

echo "📄 Korpus: $CORPUS_MB MB"
if (( $(echo "$CORPUS_MB < 5" | bc -l) )); then
    echo "❌ Korpus za mały! Oczekiwano ~11MB"
    exit 1
fi

echo "✅ Korpus gotowy do trenowania"

# Krok 3: Fine-tuning
echo ""
echo "🔥 KROK 3: Fine-tuning (może zająć 2-6 godzin)"
echo "----------------------------------------------"
echo "⏰ Rozpoczynam: $(date)"
echo ""

# Monitorowanie w tle
{
    sleep 60
    while pgrep -f "finetune_bielik.py" > /dev/null; do
        echo "⚡ $(date): Trenowanie w toku... (Monitor: nvidia-smi)"
        sleep 300  # Co 5 minut
    done
} &

# Uruchom fine-tuning
python3 finetune_bielik.py

FINETUNE_EXIT=$?

# Zatrzymaj monitorowanie
kill $! 2>/dev/null

echo ""
echo "⏰ Zakończono: $(date)"

if [ $FINETUNE_EXIT -ne 0 ]; then
    echo "❌ Fine-tuning nieudany!"
    echo "   Sprawdź logi w folderze outputs/"
    exit 1
fi

echo "✅ Fine-tuning zakończony pomyślnie!"

# Krok 4: Test modelu
echo ""
echo "🧪 KROK 4: Test wytrenowanego modelu"
echo "------------------------------------"

if [ -d "bielik-cybernetyka-lora" ]; then
    python3 test_finetuned_model.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Model działa poprawnie!"
    else
        echo "⚠️  Problemy z testowaniem modelu"
    fi
else
    echo "❌ Folder modelu nie znaleziony!"
fi

# Podsumowanie
echo ""
echo "🎉 FINE-TUNING ZAKOŃCZONY!"
echo "=========================="

if [ -d "bielik-cybernetyka-lora" ]; then
    MODEL_SIZE=$(du -sh bielik-cybernetyka-lora | cut -f1)
    echo "📂 Model LoRA: bielik-cybernetyka-lora/ ($MODEL_SIZE)"
fi

if [ -d "bielik-cybernetyka" ]; then
    GGUF_SIZE=$(du -sh bielik-cybernetyka | cut -f1)
    echo "📦 Model GGUF: bielik-cybernetyka/ ($GGUF_SIZE)"
fi

echo ""
echo "🚀 NASTĘPNE KROKI:"
echo "1. Testuj model: python3 test_finetuned_model.py"
echo "2. Użyj w kodzie: ./bielik-cybernetyka-lora/"
echo "3. Import do Ollama: ollama create bielik-cybernetyka -f bielik-cybernetyka/model.gguf"
echo ""
echo "💡 Model jest ekspertem w cybernetyce polskiej szkoły (Kossecki + Mazur)!"

echo "📊 LOGI:"
echo "- Training: outputs/training.log"
echo "- GPU usage: nvidia-smi" 