# 🦙 Simple Ollama Setup Guide

## Quick Setup (Recommended)

### Option 1: Direct HuggingFace Model
```bash
# 1. Upload your merged model to HuggingFace (or use local path)
# 2. Create simple Modelfile:
echo 'FROM speakleash/Bielik-11B-v2.2-Instruct
SYSTEM "Jesteś ekspertem cybernetyki polskiej."' > Modelfile

# 3. Import to Ollama
ollama create bielik-cybernetyka -f Modelfile
```

### Option 2: Use Online GGUF Converter
1. Go to: https://huggingface.co/spaces/ggml-org/gguf-my-repo
2. Upload your model directory
3. Download GGUF file
4. Create Modelfile and import

### Option 3: Manual Setup
```bash
# 1. First backup and merge LoRA
cp -r ./bielik-cybernetyka-lora ./bielik-cybernetyka-lora-backup
python3 setup_ollama.py --model-path ./bielik-cybernetyka-lora

# 2. If conversion fails, use this temporary setup:
echo 'FROM ./bielik-cybernetyka-lora-merged
TEMPLATE """Użytkownik: {{ .Prompt }}

Asystent: """
SYSTEM "Ekspert cybernetyki polskiej."' > Modelfile

ollama create bielik-cybernetyka -f Modelfile
```

## Testing in Open WebUI
1. Install: `docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui:main`
2. Open: http://localhost:3000
3. Add model: `bielik-cybernetyka`
4. Start chatting!

## Troubleshooting
- If GGUF conversion fails: Use online converters
- If memory issues: Add `device_map="cpu"` in merge step
- If slow: Consider quantized versions (Q4_K_M) 