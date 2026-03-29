# 🚀 Hybrid-RCA: A Dual-Branch Heterogeneous Framework for 5G Root Cause Analysis

[![Zindi Medal](https://img.shields.io/badge/Zindi-Gold_Medal_&_3rd_Place-gold.svg)](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./Hybrid-RCA.pdf)
[![Model Architecture](https://img.shields.io/badge/Model-Qwen2.5--1.5B-blue)]()
[![ML Framework](https://img.shields.io/badge/ML-CatBoost-yellow)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 🥇 **3rd Place & Gold Medal solution** for Zindi's [AI Telco Troubleshooting Challenge](https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge). 
> 
> Hybrid-RCA is a dual-branch framework combining **CatBoost** and a **GRPO-finetuned Qwen2.5-1.5B LLM** for highly efficient 5G network root cause analysis.

<p align="center">
  <img src="./Certificate-ZhengShiji-The%20AI%20Telco%20Troubleshooting%20Challenge.png" alt="Zindi 3rd Place & Gold Medal Certificate - Zheng Shiji" width="800"/>
  <img src="./Certificate-Leyuan Liao-The%20AI%20Telco%20Troubleshooting%20Challenge.png" alt="Zindi 3rd Place & Gold Medal Certificate - Zheng Shiji" width="800"/>
  <img src="./Certificate-HeQize-The%20AI%20Telco%20Troubleshooting%20Challenge.png" alt="Zindi 3rd Place & Gold Medal Certificate - Zheng Shiji" width="800"/>
</p>


---

## 📄 Read the Paper

For a deep dive into our methodology, dynamic routing mechanism, and comprehensive ablation studies, please check out our full paper included in this repository:

👉 **[Download / View the Technical Paper (PDF)](./Hybrid-RCA.pdf)**

---

## 📖 Overview

As 5G networks grow in complexity, traditional expert-based troubleshooting lacks scalability, while pure Large Language Model (LLM) approaches suffer from high inference costs and hallucination risks. 

**Hybrid-RCA** introduces a divide-and-conquer approach to telecom Root Cause Analysis (RCA):
* **High Accuracy:** Achieves a peak accuracy of **96.76%** on complex, randomized fault data (TeleLogs Phase II dataset).
* **Extreme Efficiency:** Reduces average inference latency to **~150ms** per query (a 16x speedup compared to pure 32B LLM approaches).
* **Heterogeneous Architecture:** Synergizes a highly efficient CatBoost pipeline for routine statistical pattern recognition with a specialized Reasoning LLM for complex logic.

## 🧠 System Architecture

Our framework abandons the "one-model-fits-all" paradigm, operating on a dynamic, complexity-aware routing mechanism:

1. **Dynamic Routing Dispatcher:** Evaluates the predictive uncertainty (entropy) of an incoming sample. Routine faults (~80%) are routed to the lightweight ML pipeline, while complex queries are sent to the LLM.
2. **Branch A (Adaptive ML Pipeline):** Designed for speed. Utilizes **CatBoost** enhanced with rigorous domain-specific feature engineering (e.g., PCI mod 3 conflicts).
3. **Branch B (LLM with GRPO & Multi-Agent CoT):** Employs **Qwen2.5-1.5B**, optimized via SFT (Supervised Fine-Tuning) and **GRPO** (Group Relative Policy Optimization) to align the LLM's reasoning with telecom constraints.
4. **Ensemble Fusion Layer:** Dynamically fuses the probability vector from Branch A and the confidence-weighted output of Branch B for the final decision.

## 📊 Performance Comparison

Evaluated on the TeleLogs benchmark dataset against domain State-of-the-Art:

| Model Category | Model | Phase I (Standard) | Phase II (Randomized) |
| :--- | :--- | :--- | :--- |
| Traditional ML | CatBoost (w/ Feature Eng.) | 95.12% | 84.94% |
| Domain SOTA | Qwen2.5-RCA-32B | 95.86% | 93.23% |
| **Hybrid-RCA (Ours)** | **Hybrid w/ Qwen2.5-1.5B** | **96.81%** | **96.76%** |

## 🤖 Model Availability

The fine‑tuned Qwen2.5‑1.5B model used in the LLM branch of Hybrid‑RCA is publicly available on ModelScope:

👉 [Hybrid‑RCA Fine‑tuned Model](https://www.modelscope.cn/models/Leyuan123/JJ)

## 🚀 Quick Start & Reproducibility

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/Hybrid-RCA.git
cd Hybrid-RCA
pip install -r requirements.txt


