# Stuttering Detection Platform

This repository contains code for detecting and classifying stuttering events in speech audio using deep learning (PyTorch). It is designed for research and experimentation as part of a final year project.

## Features
- **Frame-level stutter detection** using Wav2Vec2-based models
- **Custom preprocessing pipeline** for converting raw Parquet datasets to training-ready format
- **Training scripts** for both CPU and GPU environments
- **Inference scripts** for analyzing new audio files
- **Class imbalance handling** via computed class weights
- **Evaluation and reporting** of model performance

## Directory Structure
```
├── dataset/
│   ├── raw/           # Raw Parquet files (input)
│   ├── processed/     # Processed samples (.pt files, class weights)
├── models/            # Saved model checkpoints (.pth)
├── docs/              # Documentation and plans
├── extras/            # Utility scripts and notebooks
├── preprocessing.py   # Main preprocessing script
├── train_gpu.py       # Training script (GPU)
├── train_cpu.py       # Training script (CPU)
├── inference.py       # Inference script
├── inference2.py      # Alternative inference script
├── test_dataset.py    # Dataset validation script
```

## Setup Instructions
1. **Clone the repository**
2. **Install Python 3.8+** and recommended packages:
   ```bash
   pip install torch torchaudio transformers pandas pyarrow tqdm librosa
   ```
3. (Optional) For Jupyter notebooks:
   ```bash
   pip install notebook
   ```
4. (Optional) For GPU support, ensure CUDA is installed and available.

## Data Preparation
- Place your raw Parquet files in `dataset/raw/`.
- Run the preprocessing script to generate processed samples:
  ```bash
  python preprocessing.py
  ```
- Processed files will be saved in `dataset/processed/`.

## Training
- **GPU:**
  ```bash
  python train_gpu.py
  ```
- **CPU:**
  ```bash
  python train_cpu.py
  ```
- Model checkpoints are saved in `models/`.

## Inference
- Use `inference.py` or `inference2.py` to analyze new audio files:
  ```bash
  python inference.py --audio_path path/to/audio.wav --model_path models/stutter_model_epoch_6.pth
  ```
  (Edit the script to set paths as needed)

## Classes
- `fluent` (0)
- `block` (1)
- `word_rep` (2)
- `syllab_rep` (3)
- `prolongation` (4)

## Dataset Reference
- Synthetic stuttering dataset: [miosipov/Stutter_EN_Synthetic](https://huggingface.co/datasets/miosipov/Stutter_EN_Synthetic)
- Parquet schema: audio samples, metadata (start_time, end_time, stutter_type), transcription

## Example Workflow
1. Download dataset and place in `dataset/raw/`
2. Run `preprocessing.py` to create processed samples
3. Train model with `train_gpu.py` or `train_cpu.py`
4. Run inference on new audio with `inference.py`

## Contributors
- Aditya (Final Year Project)

## License
This project is for academic/research use. Please cite appropriately if used.
