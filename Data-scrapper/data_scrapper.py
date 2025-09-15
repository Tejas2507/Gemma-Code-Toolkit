# create_dataset.py

# --- Step 1: Install required packages ---
# pip install PyGithub pandas GitPython requests beautifulsoup4 datasets transformers

import os
import ast
import json
import pandas as pd
import requests
import git
import shutil
import random
from github import Github
from pathlib import Path
import time
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GITHUB_TOKEN = "Your_Token_here"  # Get from https://github.com/settings/tokens
OUTPUT_DIR = Path("./ai_ml_dataset")
CLONE_DIR = Path("./cloned_repos")

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
CLONE_DIR.mkdir(exist_ok=True)


class AIMLDatasetCollector:
    def __init__(self, github_token: str):
        self.github = Github(github_token)
        
    def extract_functions_from_code(self, code_content: str, file_path: str = "") -> List[Dict]:
        """Extract Python functions with docstrings from code content."""
        functions = []
        try:
            tree = ast.parse(code_content)
        except (SyntaxError, ValueError, MemoryError):
            return functions
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                
                if docstring and len(docstring.strip()) > 50:
                    func_lines = code_content.split('\n')[node.lineno-1:node.end_lineno]
                    func_code = '\n'.join(func_lines)
                    
                    if self.is_ai_ml_relevant(func_code, docstring):
                        functions.append({
                            'code': func_code,
                            'docstring': docstring,
                            'function_name': node.name,
                            'file_path': file_path,
                        })
        return functions
    
    def is_ai_ml_relevant(self, code: str, docstring: str) -> bool:
        """Check if code/docstring is AI/ML relevant."""
        ai_ml_keywords = [
            'neural', 'model', 'train', 'loss', 'optimizer', 'dataset',
            'tensor', 'layer', 'embedding', 'attention', 'transformer',
            'gradient', 'sklearn', 'pytorch', 'tensorflow', 'keras', 'numpy'
        ]
        content = (code + " " + docstring).lower()
        return any(keyword in content for keyword in ai_ml_keywords)
    
    def fetch_repo_functions(self, repo_name: str) -> List[Dict]:
        """Clones a GitHub repository and extracts functions from its Python files."""
        repo_url = f"https://github.com/{repo_name}.git"
        local_repo_path = CLONE_DIR / repo_name.split('/')[-1]
        
        # 1. Clone the repository
        try:
            logger.info(f"Cloning {repo_name} to {local_repo_path}...")
            git.Repo.clone_from(repo_url, local_repo_path, depth=1)
        except git.GitCommandError as e:
            logger.error(f"Error cloning {repo_name}: {e}")
            if local_repo_path.exists():
                shutil.rmtree(local_repo_path)
            return []

        # 2. Walk through local files and extract functions
        functions = []
        try:
            for root, _, files in os.walk(local_repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(local_repo_path)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                code_content = f.read()
                                file_functions = self.extract_functions_from_code(
                                    code_content, str(relative_path)
                                )
                                functions.extend(file_functions)
                        except Exception as e:
                            logger.warning(f"Error reading or parsing {file_path}: {e}")
                            continue
        finally:
             # 3. Clean up by deleting the cloned repository
            logger.info(f"Cleaning up {local_repo_path}")
            shutil.rmtree(local_repo_path)

        logger.info(f"Collected {len(functions)} functions from {repo_name}")
        return functions

# Define target repositories by category
TARGET_REPOS = {
    "pytorch_examples": ["pytorch/examples", "pytorch/vision", "pytorch/ignite"],
    "tensorflow_code": ["keras-team/keras", "tensorflow/models"],
    "sklearn_functions": ["scikit-learn/scikit-learn"],
    "research_papers": ["huggingface/transformers", "google-research/bert"],
    "kaggle_solutions": ["abhishekkrthakur/approachingalmost"],
    "course_materials": ["fastai/fastbook"]
}

# Reduced counts for a faster run, increase as needed
TARGET_COUNTS = {
    "pytorch_examples": 1500,
    "tensorflow_code": 1000,
    "sklearn_functions": 1000,
    "research_papers": 1000,
    "kaggle_solutions": 600,
    "course_materials": 600,
}

class DatasetBuilder:
    def __init__(self, collector: AIMLDatasetCollector):
        self.collector = collector
        self.dataset = []
    
    def collect_from_repos(self):
        """Collect data from all target repositories."""
        for category, target_count in TARGET_COUNTS.items():
            logger.info(f"--- Collecting for category: {category} ---")
            repos = TARGET_REPOS.get(category, [])
            collected_count = 0
            
            for repo_name in repos:
                if collected_count >= target_count:
                    break
                    
                functions = self.collector.fetch_repo_functions(repo_name)
                
                for func in functions:
                    if collected_count >= target_count:
                        break
                    
                    sample = {
                        'category': category,
                        'repo_name': repo_name,
                        'code': func['code'],
                        'docstring': func['docstring'],
                        'input_prompt': self.create_input_prompt(func['code'], category),
                        'output_response': self.format_docstring(func['docstring']),
                    }
                    self.dataset.append(sample)
                    collected_count += 1
                
                logger.info(f"Progress for {category}: {collected_count}/{target_count}")

    def create_input_prompt(self, code: str, category: str) -> str:
        """Create a randomized input prompt for training."""
        templates = {
            "pytorch_examples": [
                "Generate comprehensive documentation for this PyTorch code snippet:",
                "Explain the following PyTorch function and create its docstring:",
                "Analyze and document this PyTorch implementation:",
            ],
            "tensorflow_code": [
                "Create a detailed docstring for this TensorFlow/Keras code:",
                "Document the following TensorFlow function, including its parameters and returns:",
                "Provide technical documentation for this Keras/TF code:",
            ],
            "default": [
                "Generate a high-quality docstring for the Python function below:",
                "Document this code:",
                "Write a detailed explanation and documentation for the following code snippet:",
            ]
        }
        
        prompt_template = random.choice(templates.get(category, templates["default"]))
        return f"{prompt_template}\n\n```python\n{code}\n```"
    
    def format_docstring(self, docstring: str) -> str:
        """Clean and format docstring for consistent output."""
        return '\n'.join([line.strip() for line in docstring.strip().split('\n')])


class DatasetProcessor:
    def __init__(self, dataset: list):
        self.dataset = dataset

    def convert_to_chat_format(self) -> list:
        """Convert to the chat messages format for fine-tuning."""
        converted = []
        for sample in self.dataset:
            messages = [
                {"role": "user", "content": sample['input_prompt']},
                {"role": "assistant", "content": sample['output_response']}
            ]
            converted.append({"messages": messages})
        return converted

    def save_dataset(self, samples: list, filename: str = "ai_ml_training_dataset.jsonl"):
        """Save processed dataset as a JSON Lines file."""
        output_file = OUTPUT_DIR / filename
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        logger.info(f"✅ Saved {len(samples)} samples to {output_file}")
        return output_file

    def save_dataset_csv(self, samples: list, filename: str = "ai_ml_training_dataset.csv"):
        """Save processed dataset as a CSV file."""
        output_file = OUTPUT_DIR / filename
        
        # Convert to DataFrame
        df_data = []
        for sample in samples:
            df_data.append({
                "category": sample.get("category", ""),
                "repo_name": sample.get("repo_name", ""),
                "code": sample.get("code", ""),
                "docstring": sample.get("docstring", ""),
                "input_prompt": sample.get("input_prompt", ""),
                "output_response": sample.get("output_response", "")
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False, escapechar='\\')
        
        logger.info(f"✅ Saved {len(samples)} samples to {output_file}")
        return output_file

    def save_chat_format_csv(self, chat_samples: list, filename: str = "ai_ml_training_chat.csv"):
        """Save chat format dataset as a CSV file."""
        output_file = OUTPUT_DIR / filename
        
        # Convert to DataFrame
        df_data = []
        for sample in chat_samples:
            user_msg = sample['messages'][0]['content']
            assistant_msg = sample['messages'][1]['content']
            df_data.append({
                "input": user_msg,
                "output": assistant_msg
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False, escapechar='\\')
        
        logger.info(f"✅ Saved {len(chat_samples)} chat samples to {output_file}")
        return output_file


def main():
    """Main execution pipeline for dataset creation."""
    collector = AIMLDatasetCollector(GITHUB_TOKEN)
    builder = DatasetBuilder(collector)
    
    logger.info("🚀 Starting AI/ML dataset collection...")
    
    # Step 1: Collect from GitHub repositories
    builder.collect_from_repos()
    raw_samples = builder.dataset
    logger.info(f"Collected {len(raw_samples)} raw samples")

    # Deduplication
    logger.info("Deduplicating the dataset...")
    df = pd.DataFrame(raw_samples)
    df_deduplicated = df.drop_duplicates(subset=['code'], keep='first')
    
    logger.info(f"Removed {len(df) - len(df_deduplicated)} duplicate samples.")
    logger.info(f"Dataset size after deduplication: {len(df_deduplicated)}")
    
    # Convert the cleaned DataFrame back to a list of dictionaries
    high_quality_samples = df_deduplicated.to_dict('records')

    # Step 2: Process and save the deduplicated dataset
    processor = DatasetProcessor(high_quality_samples)
    
    # Save raw dataset as CSV
    processor.save_dataset_csv(high_quality_samples, "ai_ml_raw_dataset.csv")
    
    # Convert to chat format and save as JSONL and CSV
    training_samples = processor.convert_to_chat_format()
    processor.save_dataset(training_samples)  # JSONL format
    processor.save_chat_format_csv(training_samples)  # CSV format
    
    logger.info("🎉 Dataset creation complete!")


if __name__ == "__main__":
    if GITHUB_TOKEN == "your_github_token_here" or not GITHUB_TOKEN:
        logger.error("❌ Please set your GITHUB_TOKEN in the script.")
    else:
        main()