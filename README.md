# 3B Multimodal Autoregressive LLM Architecture & HPC Training Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Architecture Phase](https://img.shields.io/badge/Status-Architecture_Phase-blue.svg)]()

This repository contains the comprehensive technical design, architecture, and project management documentation for developing a **~3-billion-parameter multimodal autoregressive large language model (LLM)**. It is specifically designed and optimized to be trained on the **Param Ganga HPC cluster at IIT Roorkee**, operating under tight constraints (e.g., 40 V100 GPUs).

## 📖 Overview

The proposed system spans the entire lifecycle of a modern LLM:
1. **Architecture**: A Transformer decoder backbone featuring Mixture-of-Experts (MoE) FFN layers, long-context attention (sliding window + periodic full attention up to 32,768 tokens), and deep multimodal vision integration.
2. **Data Pipeline**: Web-scale corpus collection, deduplication, toxicity filtering, and quality-controlled tokenization targeting 1–2 trillion tokens.
3. **HPC Pre-Training**: Scaled training using tensor, pipeline, and data parallelism, FP16 precision, gradient checkpointing, and memory-efficient attention (10²¹–10²² FLOPs).
4. **Alignment**: Constitutional AI and RLHF fine-tuning enabling robust agentic reasoning capabilities.
5. **Safety & Deployment**: A dedicated safety classifier, inference-time gating router, rigorous red-teaming, and production API deployment.

## 📂 Repository Structure

- [`design.md`](./design.md) - The core technical design document detailing hardware constraints, architecture specifics, model parallelism strategies, and tradeoffs made for V100 architectures.
- [`requirements.md`](./requirements.md) - High-level functional and non-functional requirements, success metrics, and hardware/software dependencies.
- [`tasks.md`](./tasks.md) - A granular task breakdown and project management tracking list covering all phases from data ingestion to deployment.

## 🚀 Key Technical Highlights

- **Hardware Target**: Optimized for NVIDIA V100 (32GB HBM2) infrastructure, addressing limitations in memory bandwidth, PCIe Gen3 interconnects, and lack of BF16 support.
- **Multimodal Integration**: Unified text and visual processing.
- **Long Context**: Capable of processing 32k context windows via chunked and sliding window attention mechanisms to bypass memory limitations on older architectures.
- **Cost-Effective Training**: Maximized Model FLOPs Utilization (MFU) through extreme memory-saving techniques like Zero Redundancy Optimizer (ZeRO), activation checkpointing, and optimal microbatching.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, feedback, and discussions regarding HPC training optimization and model architecture are welcome. Feel free to open an issue or submit a pull request.
