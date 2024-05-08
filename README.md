# PanoSent: A Panoptic Sextuple Extraction Benchmark for Multimodal Conversational Aspect-based Sentiment Analysis

# ACMMM 2024 Submission

## Overview
PanoSent introduces "Sentica", a MLLM backbone  designed to advance conversational aspect-based sentiment analysis (ABSA). This model innovatively addresses challenges by seamlessly integrating multimodality (text, images, audio, and video), conversation context, fine-grained analysis, and dynamic sentiment changes along with cognitive causal rationales. 

## Methodology

### Multimodal LLM Backbone
Sentica is developed as a robust backbone system combining the capabilities of Flan-T5 (XXL) and ImageBind. This architecture allows Sentica to interpret and synthesize multimodal information, facilitating a comprehensive understanding essential for nuanced sentiment analysis in conversational scenarios.

### CoS Reasoning Framework
The Chain-of-Sentiment (CoS) framework within Sentica strategically breaks down the sentiment analysis into manageable, sequential reasoning steps. This structured approach enhances the model's ability to process complex sentiment dynamics, ensuring detailed and accurate sentiment comprehension and output.

### Paraphrase-based Verification
To maintain reliability in Sentica’s output, we implement a Paraphrase-based Verification (PpV) mechanism. By paraphrasing the structured outputs into natural language and verifying these against the original context, Sentica can self-check its reasoning accuracy, significantly reducing potential errors.

## Training
Sentica undergoes a three-phase training regimen:
1. **Multimodal Understanding**: The model learns to encode and generate appropriate textual descriptions from multimodal inputs.
2. **CoS Reasoning**: Focused on sequentially mastering each step of the CoS framework to ensure reliable sentiment analysis.
3. **Paraphrase-based Verification**: Trains Sentica to verify its own outputs through the paraphrasing mechanism, enhancing result reliability and coherence.

## Evaluation
Our comprehensive evaluation setup includes:
1. **Fine-grained Sentiment Sextuple Extraction**: Assessing Sentica's performance through element-wise, pair-wise, and sextuple-wise evaluations using precision-recall and F1 scores.
2. **Sentiment Flipping Analysis**: Focusing on the identification of sentiment changes and their triggers, evaluated through exact match and macro F1 scores.

## Repository Structure
- `models/` - Implementation of Sentica and its components.
- `data/` - Scripts for processing and managing multimodal datasets.
- `train/` - Scripts for the staged training of Sentica.
- `evaluation/` - Tools and scripts for evaluating Sentica’s performance.
- `utils/` - Supporting utilities for various operations within the repository.

```bash
conda create -n PanoSent python=3.9

conda activate PanoSent

pip install -r requirements.txt
