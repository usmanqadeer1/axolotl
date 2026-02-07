# Axolotl with Gemma 3 Support

This fork of [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) adds comprehensive support for fine-tuning **Google Gemma 3** models with automated patches to resolve compatibility issues.

## 🎯 What This Fork Adds

This repository extends the original Axolotl framework with:

- **Gemma 3 Compatibility Patches** - Automatic `token_type_ids` generation and `transformers.modelcard` fixes
- **Complete Training Pipeline** - End-to-end setup script for Gemma 3 fine-tuning
- **Memory Optimization** - Configured for 4-bit quantization with QLoRA
- **Example Configuration** - Ready-to-use YAML config for Urdu dataset training

## ✨ Why This Fork?

The standard Axolotl installation doesn't work out-of-the-box with Gemma 3 models due to:
1. Missing `token_type_ids` parameter requirement in Gemma 3's forward pass
2. `transformers.modelcard` import compatibility issues

This fork automatically handles both issues through runtime patches.

---

## 📦 Installation

### Prerequisites

- NVIDIA GPU (Ampere or newer recommended)
- Python 3.11
- CUDA 12.8

### Setup Steps

```bash
# Update system and install git
apt-get update
apt-get install -y git

# Clone this repository
cd /workspace
git clone https://github.com/usmanqadeer1/axolotl.git
cd axolotl
git checkout main
git pull

# Install dependencies
pip install packaging ninja
pip install -e '.[flash-attn,deepspeed]'

# Install PyTorch 2.8.0 with CUDA 12.8
pip uninstall torch torchvision torchaudio -y
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Accelerate and finalize installation
pip install accelerate
pip install -e .

# Login to Hugging Face (required for Gemma 3 access)
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

> **Note**: Replace `YOUR_HF_TOKEN` with your actual Hugging Face token. You can get one from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Quick Start - Gemma 3 Fine-tuning

### Run Complete Training Pipeline

```bash
bash complete_gemma3_setup.sh
```

This single command will:
1. ✅ Clear GPU memory
2. ✅ Create optimized configuration file
3. ✅ Apply all necessary patches
4. ✅ Test patch compatibility
5. ✅ Preprocess your dataset
6. ✅ Start training

### What Gets Created

The setup script automatically creates these files:

1. **`gemma3-4b-urdu-train.yml`** - Training configuration with memory-optimized settings
2. **`patch_modelcard.py`** - Fixes `transformers.modelcard` compatibility
3. **`gemma3_token_ids_patch.py`** - Auto-generates `token_type_ids` for Gemma 3
4. **`train_gemma3.py`** - Training script with patches pre-applied

---

## 🔧 Added Files Explained

### 1. `complete_gemma3_setup.sh`

Master setup and training script that orchestrates the entire pipeline. It handles:
- GPU memory management
- Configuration file generation
- Patch creation and testing
- Dataset preprocessing
- Training execution

**Usage**: `bash complete_gemma3_setup.sh`

### 2. `patch_modelcard.py`

Resolves the `transformers.modelcard` module import issue by creating a dummy module if it doesn't exist.

**What it does**:
```python
# Checks if transformers.modelcard exists
# If not, creates a dummy class with required attributes
# Prevents import errors during training
```

### 3. `gemma3_token_ids_patch.py`

Patches Gemma 3 model's forward pass to automatically generate `token_type_ids` when missing.

**What it patches**:
- `Gemma3Model.forward()` - Generates zero-filled token_type_ids for all tokens
- `Gemma3ForCausalLM.forward()` - Creates token_type_ids during training, marking assistant tokens (where labels != -100) as type 1

**How it works**:
```python
# During training, if token_type_ids is None:
# 1. Create zero tensor matching input_ids shape
# 2. Set token_type_ids[labels != -100] = 1 (mark assistant tokens)
# 3. Pass to model's forward function
```

### 4. `train_gemma3.py`

Custom training script that applies all patches before importing Axolotl.

**Execution flow**:
1. Applies `patch_modelcard.py`
2. Applies `gemma3_token_ids_patch.py`
3. Imports and runs Axolotl's training pipeline

### 5. `gemma3-4b-urdu-train.yml`

Optimized training configuration for Gemma 3 with 4-bit quantization.

**Key configurations**:
```yaml
# Model settings
base_model: google/gemma-3-4b-it
load_in_4bit: true                    # Memory efficient 4-bit quantization
bnb_4bit_quant_type: nf4             # NormalFloat4 quantization
chat_template: gemma3                 # Gemma 3 specific template

# LoRA settings
adapter: qlora                        # Quantized LoRA
lora_r: 16                           # LoRA rank
lora_alpha: 16                       # LoRA scaling factor

# Training optimization
gradient_accumulation_steps: 8        # Effective batch size multiplier
micro_batch_size: 1                  # Fits in limited GPU memory
gradient_checkpointing: true         # Reduces memory usage
```

---

## 📝 Custom Dataset Training

To train on your own dataset, modify the `datasets` section in `gemma3-4b-urdu-train.yml`:

```yaml
datasets:
  - path: your-username/your-dataset
    name: subset_name  # Optional
    type: chat_template
    field_messages: messages
    message_property_mappings:
      role: role
      content: content
```

**Dataset format** should follow the chat template structure:
```json
{
  "messages": [
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

---

## 🔍 How the Patches Work

### Token Type IDs Generation

Gemma 3 requires `token_type_ids` to distinguish between user and assistant tokens. The patch automatically:

1. **During Forward Pass**: If `token_type_ids` is None, creates a zero tensor
2. **During Training**: Marks assistant tokens (where `labels != -100`) as type 1
3. **Seamless Integration**: No manual intervention required

### ModelCard Compatibility

The `patch_modelcard.py` creates a minimal dummy module that satisfies import requirements without breaking functionality.

---

## 🐛 Troubleshooting

### Out of Memory Errors

Reduce memory usage in `gemma3-4b-urdu-train.yml`:
```yaml
micro_batch_size: 1                  # Try reducing if still OOM
sequence_len: 1024                   # Reduce from 2048
gradient_accumulation_steps: 16      # Increase to maintain effective batch size
```

### Dataset Loading Errors

Verify your dataset format:
```bash
# Test dataset loading
from datasets import load_dataset
ds = load_dataset("your-username/your-dataset")
print(ds[0])  # Should show messages field
```

### Authentication Issues

Ensure you're logged into Hugging Face:
```bash
huggingface-cli login
# Or
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"
```

---

## 📚 Additional Resources

For general Axolotl documentation and features:
- [Official Axolotl Repository](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Documentation](https://docs.axolotl.ai/)
- [Configuration Reference](https://docs.axolotl.ai/docs/config-reference.html)
- [Dataset Formats](https://docs.axolotl.ai/docs/dataset-formats/)

---

## 🤝 Contributing

Found an issue or want to improve Gemma 3 support? Contributions are welcome!

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📜 License

This project maintains the Apache 2.0 License from the original Axolotl repository.

---

## 🙏 Acknowledgments

- [Axolotl Team](https://github.com/axolotl-ai-cloud/axolotl) for the excellent fine-tuning framework
- Google for the Gemma 3 models
- The open-source community for continuous improvements
