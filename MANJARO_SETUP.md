# 🐧 AI OCR Setup for Manjaro Linux

Complete setup guide for running the AI OCR processor on Manjaro Linux with Ollama and qwen2.5vl:7b.

## 🔧 Prerequisites Installation

### 1. Install pyenv (if not already installed)
```bash
# Install pyenv dependencies
sudo pacman -S --needed base-devel openssl zlib xz tk

# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration (~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

### 2. Install Python 3.11.7
```bash
# Install Python 3.11.7
pyenv install 3.11.7

# Verify installation
pyenv versions
```

### 3. Install Ollama (if not already installed)
```bash
# Method 1: Official installer
curl -fsSL https://ollama.ai/install.sh | sh

# Method 2: From AUR
yay -S ollama

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Or run manually
ollama serve
```

### 4. Install qwen2.5vl:7b model
```bash
ollama pull qwen2.5vl:7b
ollama list  # Verify model is installed
```

## 📦 Project Setup

### 1. Clone and navigate to project
```bash
git clone <your-repo-url> cybernetyka_teksty
cd cybernetyka_teksty
```

### 2. Setup Python environment
```bash
# The .python-version file will automatically activate the environment
# But first create it:
pyenv virtualenv 3.11.7 cybernetyka-ai-ocr

# Verify environment is active (should show cybernetyka-ai-ocr)
python --version
which python
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements_ai_ocr.txt
```

### 4. Test installation
```bash
# Test script functionality
python ai_ocr_processor.py --help

# Test Ollama connection
python -c "
import requests
try:
    response = requests.get('http://localhost:11434/api/tags')
    print('✅ Ollama is running and accessible')
    models = response.json().get('models', [])
    model_names = [m['name'] for m in models]
    if 'qwen2.5vl:7b' in model_names:
        print('✅ qwen2.5vl:7b model is available')
    else:
        print('❌ qwen2.5vl:7b model not found')
        print(f'Available models: {model_names}')
except:
    print('❌ Cannot connect to Ollama')
"
```

## 🚀 Usage

### Start Ollama (if not running as service)
```bash
ollama serve
```

### Run AI OCR processor
```bash
# Interactive menu
python ai_ocr_processor.py

# Process specific file
python ai_ocr_processor.py "TEXTS/autonom/Kossecki/document.pdf"
```

## ⚡ Performance Notes for Manjaro

### 1. GPU Acceleration
If you have an NVIDIA GPU:
```bash
# Install CUDA (if needed)
sudo pacman -S cuda

# Verify GPU is available to Ollama
nvidia-smi
```

### 2. Memory Management
For large PDFs, you may need to adjust system settings:
```bash
# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. Ollama Configuration
Create/edit Ollama config for better performance:
```bash
# Create systemd override
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## 🐛 Troubleshooting

### pyenv not found
```bash
# Make sure pyenv is in PATH
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

### Python build issues
```bash
# Install additional dependencies
sudo pacman -S sqlite readline gdbm
pyenv install 3.11.7
```

### Ollama connection issues
```bash
# Check if Ollama is running
sudo systemctl status ollama
curl http://localhost:11434/api/tags

# Check firewall
sudo ufw allow 11434
```

### Missing libraries
```bash
# Install system dependencies for PDF processing
sudo pacman -S poppler-utils tesseract
```

## ✅ Verification Checklist

- [ ] pyenv installed and working
- [ ] Python 3.11.7 installed
- [ ] cybernetyka-ai-ocr environment created and active
- [ ] All Python dependencies installed
- [ ] Ollama service running
- [ ] qwen2.5vl:7b model downloaded
- [ ] AI OCR processor shows menu correctly
- [ ] Can access TEXTS/ folder and find PDFs

## 🔄 Cross-Platform Notes

The `.python-version` file will work on Manjaro exactly the same as on macOS:
- Automatically activates the environment when you `cd` into the directory
- No need for manual `pyenv activate`
- Consistent Python version across machines

## 📊 Expected Performance

On typical Manjaro Linux hardware:
- **CPU only**: ~10-15 seconds per page
- **With GPU**: ~2-4 seconds per page
- **Memory usage**: ~2-4GB during processing
- **Model load time**: ~30 seconds first time

---

**🎯 After setup, the AI OCR processor will work identically on Manjaro as on macOS!**