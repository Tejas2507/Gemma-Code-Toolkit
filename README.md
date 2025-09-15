# Gemma-Lab 🧪

*A cutting-edge exploration of Google's Gemma models, featuring educational implementations and practical fine-tuning applications.*

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🌟 Overview

Gemma-Lab is an innovative research initiative that bridges theoretical understanding with practical implementation of state-of-the-art language models. This project offers a dual-path approach:

-   **🧠 Educational Foundation**: A from-scratch implementation of the Gemma architecture to deeply understand transformer mechanics.
-   **⚡ Practical Application**: A complete, production-ready pipeline for fine-tuning powerful models on custom tasks.

Dive in to transition from fundamental concepts to deploying a specialized code documentation assistant.

---

## 🏗️ Project Architecture

### Core Components

| Component | Purpose | Technologies |
| :--- | :--- | :--- |
| **🧪 Gemma From Scratch** | Educational implementation | PyTorch, Transformers |
| **📥 Data Scraper** | Custom dataset collection | Python, Web scraping |
| **⚙️ Fine-Tuning Pipeline** | Model specialization | QLoRA, Flash Attention 2 |
| **🚀 Inference Module** | Production deployment | Hugging Face, Transformers |

---

## 📁 Component Deep Dive

### 1. 🧪 Gemma From Scratch
**Location: `gemma_from_scratch/`**
A clean-room implementation of a Gemma-like transformer (270M parameters), perfect for educational exploration. This module demystifies:
-   Attention mechanisms
-   Feed-forward blocks
-   Embedding layers
-   Core transformer architecture

### 2. 📥 Intelligent Data Collection
**Location: `data_scrapper/`**
Customizable Python scripts for building high-quality, specialized datasets. Easily adapt these scripts to gather data for your own unique ML tasks.

### 3. ⚙️ Advanced Fine-Tuning Pipeline
**File: `CodeGemma_Fine-Tuning.ipynb`**
This notebook contains a state-of-the-art training pipeline featuring modern, efficient techniques:
-   **🪶 4-bit Quantization (QLoRA)**: Drastically reduces memory footprint.
-   **⚡ Flash Attention 2**: Optimizes speed and memory usage for faster training.
-   **📦 Dataset Packing**: Maximizes GPU utilization and training efficiency.
-   **🤗 Hugging Face Integration**: Seamlessly handles model loading and pushing adapters to the Hub.

### 4. 🚀 Production Inference
**File: `CodeGemma_Inference.ipynb`**
A ready-to-use deployment module to test and showcase your fine-tuned model. It demonstrates how to:
-   Load your adapter from the Hugging Face Hub.
-   Format prompts for the code documentation task.
-   Generate high-quality documentation for new, unseen code snippets.

---

## 🚀 Quick Start

### Prerequisites
-   Python 3.10+
-   PyTorch 2.0+
-   A Hugging Face Account (for accessing gated models and pushing adapters)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gemma-lab.git
    cd gemma-lab
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Fine-Tuning Pipeline

For the best experience, run the `CodeGemma_Fine-Tuning.ipynb` notebook on a platform with free GPU access:

1.  **Upload to Kaggle or Google Colab.**
2.  **Add Your Data:** Upload the dataset you collected with the scraper.
3.  **Set Secrets:** Add your Hugging Face token as a secret named `HF_TOKEN` to authenticate.
4.  **Install Dependencies:** Run the initial cells to install all necessary libraries.
5.  **Run All Cells:** Execute the notebook to start the fine-tuning process and push your adapter to the Hub.

---

## 🎯 Key Features

-   **Educational Excellence**: Deep dive into transformer architecture from the ground up.
-   **Modern Training Techniques**: Implements QLoRA and Flash Attention for efficient training.
-   **Fully Customizable**: Easily adapt the code and data for various domains and tasks.
-   **Production Ready**: Integrated with the Hugging Face ecosystem for easy deployment.
-   **Memory Efficient**: Leverages 4-bit quantization to run on more accessible hardware.

---

## 🤝 Contributing

Contributions are welcome! We encourage you to:
-   Submit issues and feature requests.
-   Create pull requests for improvements and bug fixes.
-   Share your own fine-tuned models and datasets.
-   Improve documentation and examples.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

-   **Google** for the groundbreaking Gemma models.
-   **Hugging Face** for the incredible `transformers`, `peft`, and `trl` libraries that make this work possible.
-   **The Open-Source Community** for continuous inspiration and support.

---

**Gemma-Lab**: Where theoretical understanding meets practical implementation. Start your journey from fundamentals to state-of-the-art applications today! 🚀
