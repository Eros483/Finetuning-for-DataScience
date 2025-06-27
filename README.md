# 📟 FineTuning LLM for Data Science workloads
Finetunes Meta's Llama 2-7b sequentially over python code and basic use case data science code.

## 📦 Usage

Refer to [Hugging Face](https://huggingface.co/Eros483/llama-ds-q5.gguf/tree/main) for how to download and use the model.

## 💡 Project Structure
```
Finetuning-for-DataScience.
├───datasets
├───evaluation
├───llama.cpp
├───models
│   ├───base_model
│   │   └───.cache
│   │       └───huggingface
│   │           └───download
│   ├───finetuned-python
│   │   ├───checkpoint-6556
│   │   └───checkpoint-9834
│   ├───finetuned-python-datascience
│   │   ├───checkpoint-12336
│   │   └───checkpoint-8224
│   └───merged_model
├───notebooks
│   └───unsloth_compiled_cache
│       └───__pycache__
└───processed-data

```
## ⚙️ Environment Setup
1. We recommend using Anaconda for environment management.

2. Create the environment using the provided environment.yml, and activate it:
    - #### Make sure to remove llama-cpp-python from yml file
```
conda env create -f environment.yml
conda activate finetuning
```

## ⚙️ How The Model was created
### Finetuning Library
- Refer to [Unsloth](https://docs.unsloth.ai/), for installing unsloth.
    - Allows fine tuning on lower amounts of VRAM at a far more efficient rate.
### Data Acquisition
- We need a dataset consisting of reliable Python code, for initially teaching the model the basics of python syntax and logic.
- Then we proceeded to tune the model over data science code.
- Python Dataset was acquired from [Kaggle](https://www.kaggle.com/datasets/bhaveshmittal/python-programming-questions-dataset).
- DS code was acquired from [Hugging Face](https://huggingface.co/datasets/ed001/ds-coder-instruct-v2).
- We preprocessed the datasets for effective finetuning in `notebooks/training_data_processing.ipynb`.
    - Cleaned dataset of irrelevant values.
    - Formatted columns into strict Prompt/Response style data.
    - saved as jsonl files.

### Downloading base model to be finetuned
- Run `notebooks/model_hf_setup.ipynb`.
- Used older Llama 2 for direct HF file access, for greater control over training parameters as instructed in `Unsloth` Documentation.
- Download both model and tokenizers.

### Sequential Fine Tuning Strategy
- Employed PEFT training strategy with LORA.
    - PEFT is Parameter Efficient Fine Tuning, and involves fine tuning only a few necessary weight parameters.
    - LORA is Low Rank Adaptation, which allows us to achieve PEFT by training only self injected small matrices.
- Parameters followed are the optimized values provided by unsloth.
- Allows us to be efficient with RAM utilization.
- Initial finetuning was done in `notebooks/python_finetuning.ipynb`, and followed up in `notebooks/data_science_finetuning.ipynb`.

### Merging and exporting model
- Combined the fine-tuned weights with the base model, and saved it as hugging face model.
- Done in `notebooks/merge_export.ipynb`

### Conversion and quantization
- Clone `llama-cpp-python`
- Convert saved hf model using `llama.cpp/convert_hf_to_gguf_update.py` and `llama.cpp/convert_hf_to_gguf.py` into a GGUF format, for greater ease of use through platforms such as `llama-cpp-python` and `OLlama`.
- Quantize model to a state as per available VRAM byusing `llama.cpp/build/bin/Release/llama-quantize.exe`.

### Testing Inference
- Install `llama-cpp-python`.
    - Utilise provided wheel via `pip install llama_cpp_python-0.3.9+cuda-cp310-cp310-win_amd64.whl`.
        - Working build for 4060 graphics card, CUDA 12.9, and Python 3.10.
    - Else refer to relevant documentation.
- Utilise `notebooks/testing.ipynb` for checking on inference.
    - Can work very well for basic data science code.
    - Excels at creating structure for complicated code.

### Running HumanEval Benchmarks
- HumanEval is strictly linux only.
- Utilised [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) instead.