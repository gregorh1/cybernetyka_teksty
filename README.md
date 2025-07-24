# 🚀 Cybernetyka Teksty - Fine-tuning Bielik-11B

**Polski model AI wyspecjalizowany w cybernetyce** oparty na korpusie prac Józefa Kosseckiego i Mariana Mazura.

## 📚 Co to jest?

Kompletny zestaw narzędzi do wytrenowania polskiego modelu językowego **Bielik-11B** na korpusie cybernetyki polskiej szkoły:

- **📖 Korpus 11MB** - Wszystkie dostępne prace Kosseckiego + Mazura (OCR + ręczne korekty)
- **🔥 Fine-tuning** - Gotowe skrypty dla RTX 3090 (24GB VRAM)
- **🧪 Testy** - Automatyczne sprawdzanie jakości modelu
- **⚡ Dwie metody** - Unsloth (szybka) i Standard (stabilna)

## 🎯 Rezultat

Model który rozumie i generuje teksty o:
- Cybernetyce społecznej i kulturze
- Teorii układów samodzielnych  
- Sterowaniu społecznymi procesami
- Metacybernetyce i filozofii nauki
- Analizie systemów sterowania

## ⚡ Szybki Start

### 1. Klonuj repozytorium
```bash
git clone https://github.com/twój-user/cybernetyka_teksty.git
cd cybernetyka_teksty
```

### 2. Przygotuj środowisko
```bash
# Python 3.10+ + CUDA + RTX 3090
pyenv virtualenv 3.11.7 cybernetyka-finetune
pyenv activate cybernetyka-finetune
```

### 3. Uruchom automatyczny fine-tuning
```bash
./run_finetune.sh
```

Skrypt automatycznie:
- ✅ Sprawdzi środowisko  
- 🧪 Przetestuje dostępność bibliotek
- 🤖 Pozwoli wybrać metodę trenowania
- 🔥 Uruchomi fine-tuning (2-6h)
- 🧪 Przetestuje wytrenowany model

## 📊 Metody Fine-tuningu

| Metoda | Czas | VRAM | Stabilność | Kompatybilność |
|--------|------|------|------------|----------------|
| **🦥 Unsloth** | 2-4h | 16-20GB | Niestabilna | Problematyczna |
| **🔧 Standard** | 3-6h | 22-24GB | Bardzo stabilna | Wysoka |

## 📁 Zawartość

```
cybernetyka_teksty/
├── 📖 cybernetyka_corpus.txt           # Korpus 11MB
├── 🚀 run_finetune.sh                  # Główny skrypt
├── 🦥 finetune_bielik.py               # Unsloth (szybkie)
├── 🔧 finetune_bielik_standard.py      # Standard (stabilne)
├── 🧪 test_finetuned_model.py          # Testy modelu
├── 📦 setup_environment.py             # Setup pakietów
├── Kossecki/                           # Prace Kosseckiego
├── Mazur/                              # Prace Mazura
└── 📖 README_FINETUNE.md               # Szczegółowa dokumentacja
```

## 🔧 Wymagania

### Hardware
- **RTX 3090** (24GB VRAM) lub lepsze
- **32GB+ RAM** 
- **50GB+ miejsca** na dysku

### Software  
- **Python 3.10+**
- **CUDA 12.1+**
- **PyTorch 2.4+**
- **Linux** (testowane na Manjaro/Ubuntu)

## 🧪 Przykład użycia

Po wytrenowaniu modelu:

```python
# Załaduj model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("speakleash/Bielik-11B-v2.2-Instruct")
model = PeftModel.from_pretrained(base_model, "./bielik-cybernetyka-lora")
tokenizer = AutoTokenizer.from_pretrained("./bielik-cybernetyka-lora")

# Generuj tekst
prompt = "Co to jest cybernetyka społeczna według Kosseckiego?"
# ... model wygeneruje odpowiedź opartą na korpusie
```

## 📖 Dokumentacja

- **[📋 README_FINETUNE.md](README_FINETUNE.md)** - Kompletny przewodnik fine-tuningu
- **[🔄 TRANSFER_INSTRUCTIONS.md](TRANSFER_INSTRUCTIONS.md)** - Instrukcje przenoszenia między systemami

## 🎯 Przykłady odpowiedzi modelu

**Pytanie:** *"Wyjaśnij teorię układów samodzielnych Mazura"*

**Model:** *"Teoria układów samodzielnych Mariana Mazura to fundamentalna koncepcja w cybernetyce polskiej szkoły. Układ samodzielny to system zdolny do utrzymywania swojej struktury i funkcji w zmieniających się warunkach środowiska poprzez mechanizmy sprzężeń zwrotnych i autoregulacji..."*

## 📊 Statystyki korpusu

- **📚 Dokumenty:** ~150 prac naukowych
- **📄 Tekst:** 11MB czystego tekstu polskiego  
- **🏷️ Tematyka:** Cybernetyka, socjotechnika, metacybernetyka
- **👨‍🔬 Autorzy:** Józef Kossecki, Marian Mazur
- **📅 Okres:** 1960-2010
- **✅ Jakość:** OCR + ręczne korekty

## 🚀 Status projektu

- ✅ **Korpus gotowy** - 11MB wysoce jakościowych tekstów
- ✅ **Fine-tuning działa** - Przetestowane na RTX 3090
- ✅ **Dwie metody** - Unsloth i Standard
- ✅ **Automatyzacja** - Jeden skrypt uruchamia wszystko
- 🔄 **Testowanie** - Ciągłe ulepszanie jakości

## 🤝 Wkład

Mile widziane:
- 🐛 **Zgłoszenia błędów**
- 💡 **Sugestie ulepszeń**  
- 📝 **Korekty korpusu**
- 🔧 **Optymalizacje kodu**

## 📄 Licencja

- **Kod:** MIT License
- **Korpus:** Dozwolony użytek naukowy/edukacyjny
- **Model:** Według licencji Bielik-11B

## 🏆 Cel projektu

Stworzenie **pierwszego polskiego modelu AI wyspecjalizowanego w cybernetyce**, który:
- 🧠 Rozumie terminologię polskiej szkoły cybernetycznej
- 📚 Opiera odpowiedzi na pracach Kosseckiego i Mazura  
- 🔬 Może wspierać badania i edukację w cybernetyce
- 🇵🇱 Zachowuje polski kontekst naukowy

---

**💡 Zbuduj swojego eksperta AI w cybernetyce w 6 godzin!**

**⭐ Jeśli projekt Ci się podoba - zostaw gwiazdkę!** 