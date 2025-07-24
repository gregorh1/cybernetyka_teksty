#!/usr/bin/env python3
"""
Setup Environment dla Fine-tuningu Bielik-11B
Sprawdza i instaluje wszystkie wymagane biblioteki
"""

import sys
import subprocess
import importlib
import platform

def check_python_version():
    """Sprawdź wersję Python"""
    print("🐍 Sprawdzam wersję Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} jest za stary!")
        print("   Wymagany Python 3.10+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_cuda():
    """Sprawdź dostępność CUDA"""
    print("🎮 Sprawdzam CUDA...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU wykryte")
            return True
        else:
            print("❌ nvidia-smi nie działa")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi nie znalezione - brak NVIDIA GPU lub sterowników")
        return False

def check_pytorch_cuda():
    """Sprawdź czy PyTorch ma obsługę CUDA"""
    print("🔥 Sprawdzam PyTorch z CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ PyTorch z CUDA - OK")
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f} GB")
            print(f"   Count: {gpu_count}")
            
            if gpu_memory < 20:
                print("⚠️  VRAM < 20GB - użyj QLoRA zamiast LoRA")
            
            return True
        else:
            print("❌ PyTorch bez obsługi CUDA")
            return False
            
    except ImportError:
        print("⚠️  PyTorch nie zainstalowany - będzie zainstalowany")
        return False

def install_package(package, check_name=None):
    """Zainstaluj pakiet pip"""
    check_name = check_name or package.split('[')[0].split('@')[0].split('==')[0]
    
    try:
        importlib.import_module(check_name.replace('-', '_'))
        print(f"✅ {check_name} już zainstalowany")
        return True
    except ImportError:
        print(f"📦 Instaluję {package}...")
        
        try:
            if '@' in package or 'git+' in package:
                # Git packages
                cmd = f"pip install '{package}'"
            else:
                cmd = f"pip install {package}"
                
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} zainstalowany pomyślnie")
                return True
            else:
                print(f"❌ Błąd instalacji {package}:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Wyjątek podczas instalacji {package}: {e}")
            return False

def install_pytorch_cuda():
    """Zainstaluj PyTorch z obsługą CUDA"""
    print("🔥 Instaluję PyTorch z CUDA...")
    
    # Sprawdź aktualną wersję CUDA
    cuda_version = "cu121"  # Default dla nowych systemów
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "CUDA Version: 12.1" in result.stdout or "CUDA Version: 12.2" in result.stdout:
            cuda_version = "cu121"
        elif "CUDA Version: 11.8" in result.stdout:
            cuda_version = "cu118"
    except:
        pass
    
    pytorch_cmd = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}"
    
    print(f"📦 Instaluję PyTorch dla {cuda_version}...")
    result = subprocess.run(pytorch_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ PyTorch z CUDA zainstalowany")
        return True
    else:
        print("❌ Błąd instalacji PyTorch:")
        print(result.stderr)
        return False

def main():
    """Główna funkcja setup"""
    
    print("🚀 SETUP ENVIRONMENT - Fine-tuning Bielik-11B")
    print("=" * 60)
    
    # Lista sprawdzeń
    checks = []
    
    # Sprawdź Python
    checks.append(check_python_version())
    
    # Sprawdź CUDA
    checks.append(check_cuda())
    
    # Sprawdź PyTorch
    pytorch_ok = check_pytorch_cuda()
    if not pytorch_ok:
        pytorch_ok = install_pytorch_cuda()
        if pytorch_ok:
            pytorch_ok = check_pytorch_cuda()
    checks.append(pytorch_ok)
    
    if not all(checks[:3]):
        print("\n❌ Podstawowe wymagania nie są spełnione!")
        print("   Napraw błędy przed kontynuowaniem.")
        return False
    
    print("\n📦 Instaluję biblioteki dla fine-tuningu...")
    
    # Lista pakietów do zainstalowania
    packages = [
        # Core packages
        ("datasets", "datasets"),
        ("transformers>=4.36.0", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("bitsandbytes", "bitsandbytes"),
        
        # Unsloth (najważniejsze)
        ("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git", "unsloth"),
        
        # Optional but recommended
        ("xformers", "xformers"),
        ("flash-attn --no-build-isolation", "flash_attn"),
    ]
    
    success_count = 0
    for package, check_name in packages:
        if install_package(package, check_name):
            success_count += 1
        else:
            print(f"⚠️  {package} nie został zainstalowany - może działać bez tego")
    
    print(f"\n📊 PODSUMOWANIE INSTALACJI:")
    print(f"✅ Zainstalowane: {success_count}/{len(packages)} pakietów")
    
    # Test finalny
    print(f"\n🧪 FINAL TEST...")
    try:
        import torch
        from transformers import AutoTokenizer
        import datasets
        
        if torch.cuda.is_available():
            print("✅ Środowisko gotowe do fine-tuningu!")
            
            # Sprawdź Unsloth
            try:
                from unsloth import FastLanguageModel
                print("✅ Unsloth zainstalowany i gotowy")
            except ImportError:
                print("⚠️  Unsloth niedostępny - spróbuj ponownie")
            
            return True
        else:
            print("❌ CUDA niedostępne w PyTorch")
            return False
            
    except ImportError as e:
        print(f"❌ Brak wymaganych bibliotek: {e}")
        return False

def create_requirements_file():
    """Utwórz plik requirements.txt"""
    requirements = """# Fine-tuning Bielik-11B Requirements
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
xformers>=0.0.22
# Unsloth - install separately with git+
# unsloth @ git+https://github.com/unslothai/unsloth.git
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📄 Utworzono requirements.txt")

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 ENVIRONMENT GOTOWY!")
        print(f"   Możesz uruchomić: python3 finetune_bielik.py")
        create_requirements_file()
    else:
        print(f"\n❌ ENVIRONMENT NIE GOTOWY")
        print(f"   Napraw błędy i uruchom ponownie")
        
    print(f"\n💡 Pro Tips:")
    print(f"   - Monitor GPU: watch nvidia-smi")
    print(f"   - Logi trenowania: tail -f outputs/training.log") 
    print(f"   - Zmniejsz batch_size jeśli OOM error") 