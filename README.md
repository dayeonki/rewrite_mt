# Rewriting Inputs for Translation with Large Language Models

This repository contains the code and dataset for our paper **Rewriting Inputs for Translation with Large Language Models**.

<div align="center">
[🤖 <b><a href=https://github.com/dayeonki/rewrite_mt/code>Code</a></b> / 🤗 <b><a href=https://huggingface.co/datasets/zoeyki/rewrite_mt_dataset>Dataset</a></b> / 📄 <b><a href=>Paper</a></b>]
</div>


## Abstract
Rewriting inputs is a common strategy to enhance translation quality exploited by end users and machine translation (MT) developers, particularly with dedicated MT architectures. How effective is this approach when translating with Large Language Models (LLMs)? We conduct an empirical study using the Tower multilingual LLM, primarily trained for translation-related tasks, on translation from English into German, Russian and Chinese. We find that MT-agnostic style rewrites do not uniformly improve translations. Consequently, we introduce re-ranking and fine-tuning strategies to guide rewrites with reference-free quality estimation. These techniques enhance translation quality according to both automatic and human evaluations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ef34c652-a880-47b1-9f5f-bd1586f2fd25" width="900">
</p>


## Quick Links
- [Overview](#overview)
- [Prompting](#prompting)
- [Re-ranking](#re-ranking)
- [Fine-tuning](#fine-tuning)
- [DPO](#dpo)


## Overview
In this work, we shed light on these issues by asking the following questions:

- Can we improve MT quality from LLMs by rewriting inputs for style?
- How can we guide models to rewrite inputs to improve their translatability? 

To address these questions, we first generate **MT-Agnostic rewrites** by prompting LLMs to simplify, paraphrase or change the style of the original input. Next, we design three strategies to obtain **MT-Aware rewrites**: using Chain-of-Thought prompting, selecting generic rewrites with reference-free quality estimation metrics, and fine-tuning LLMs to rewrite for translation. We conduct an empirical study of the <a href=https://arxiv.org/abs/2402.17733>Tower</a> LLM on translation from English into German, Russian and Chinese. Our findings suggest that rewriting inputs can improve translation quality and that quality estimation feedback helps generate inputs that are better translated.

## Prompting

## Re-ranking

## Fine-tuning

## DPO



## Citation
```
```
