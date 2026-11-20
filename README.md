# Python Code Generation with DistilGPT2

## Overview

An experimental project exploring how **synthetic training data and parameter-efficient fine-tuning** affect Python code generation using DistilGPT2.

The goal is to compare different training strategies and investigate whether combining original and synthetically generated examples can improve code generation.

## Approach

Four experimental setups were compared:

- **Baseline:** Pretrained DistilGPT2 without project-specific fine-tuning.
- **Original data:** Fine-tuning on a Python instruction dataset.
- **Synthetic data:** Fine-tuning on Python examples generated using an external LLM API.
- **Combined data:** Fine-tuning on both original and synthetic examples.

The implementation uses **LoRA/QLoRA** for parameter-efficient fine-tuning and evaluates generated code using metrics including BLEU, chrF, CodeBLEU, METEOR, ROUGE-L, and RUBY.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Datasets · PEFT · TRL · Accelerate · bitsandbytes

## Key Areas

- Large Language Model fine-tuning
- Synthetic data generation
- Parameter-efficient training with LoRA/QLoRA
- NLP and code-generation evaluation
- Experiment design and comparison

## Implementation

The complete workflow, including data preparation, synthetic data generation, model training, and evaluation, is available in the Jupyter notebook:

**[Explore the notebook](./GenerativeAI_DistilBert.ipynb)**

A GPU-enabled environment such as Google Colab is recommended for reproducing the training experiments.

## Notes

This is an experimental comparison. The current evaluation implementation has an aggregation issue, so its reported metric values should not be interpreted as reliable benchmark results until corrected and rerun.
