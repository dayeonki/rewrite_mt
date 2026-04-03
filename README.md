<div align="center">

 # Automatic Input Rewriting Improves <br> Translation with Large Language Models

<p align="center">
<img width="900" alt="Screenshot 2025-03-21 at 1 48 51 PM" src="https://github.com/user-attachments/assets/af4275ae-47e1-4297-b585-64881d38c3f2" />
</p>

<a href=https://dayeonki.github.io/>Dayeon Ki</a>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a><br>
University of Maryland
<br>

This repository contains the code and dataset for our NAACL 2025 Main paper <br> **Automatic Input Rewriting Improves Translation with Large Language Models**.

<p>
  <a href="https://aclanthology.org/2025.naacl-long.542/" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
 <br>
  <a href="https://huggingface.co/datasets/zoeyki/rewrite_mt_dataset" target="_blank">
    <img src="https://img.shields.io/badge/🤗-Dataset-yellow?style=flat" alt="HuggingFace">
  </a>
</p>

</div>

---

## 👾 TL;DR
Can we improve machine translation with LLMs by _rewriting_ their inputs automatically? We present an empirical study of 21 input rewriting methods for translating from English into 6 target languages, showing text simplification as the most effective MT-agnostic rewrite strategy.


## 📰 News
- **`2025-01-22`** Our paper is accepted to **NAACL 2025**! See you in New Mexico!


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Data Preparation](#data-preparation)
  - [MT-Agnostic Rewrite](#mt-agnostic-rewrite)
  - [Task-Aware Rewrite](#task-aware-rewrite)
  - [Translation](#translation)
  - [Evaluation](#evaluation)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---

<a id="overview"></a>
## 🗺️ Overview

We ask the following questions: <br>

 (**1**) Can we improve MT quality from LLMs by rewriting inputs for style? <br>
 (**2**) How can we guide models to rewrite inputs to improve their translatability? 

We conduct an empirical study with 21 input rewriting methods with varying levels of **MT-awareness** on translation. <br>
We first **rewrite** the source sentence using different rewriting methods, **translate** each rewrite in the target language, and then **evaluate** the rewrites on the basis of (i) translatability and (ii) meaning preservation.


### Results
<div align="center">
<img width="800" height="616" alt="Screenshot 2026-04-03 at 1 00 59 PM" src="https://github.com/user-attachments/assets/5416d680-5fef-4530-ba34-2d0ae3900901" />
</div>



<a id="quick_start"></a>
## 🚀 Quick Start

### Data Preparation
We use the WMT-23 General MT task from [Tower-Eval](https://huggingface.co/datasets/Unbabel/TowerEval-Data-v0.1) dataset. For the main experiments, we focus on three language pairs: English-German (en-de), English-Russian (en-ru), and English-Chinese (en-zh). The dataset is in: `data/raw_{language_pair}.jsonl`.

We translate each source sentence in English to the respective target language using Tower-Instruct LLM. The translated dataset is in: `data/{language_pair}_mt_tower.jsonl`.


### MT-Agnostic Rewrite
MT-Agnostic rewrite methods leverage prior assumptions on what makes text _easier_ to translate and do not take as input any signal of translatability or knowledge about the end-task. We consider three prompting variants here, all inspired by prior works on source rewriting:

1. **Simplification:** replacing complex words with simpler ones, rephrasing complex syntactic structures, or shortening sentences.
    - code: `mt_agnostic/simple_{llm}.py`

2. **Paraphrasing:** LLMs might benefit MT by normalizing inputs using language patterns that are more frequent in LLM training data.
      - code: `mt_agnostic/paraphrase_{llm}.py`
3. **Stylistic transfer:** use an off-the-shelf text editing tool CoEdit to rewrite inputs according to diverse style specifications, including fixing the grammar, making the text more coherent, making it easier to understand, and rewriting the text more formally.
    - code: `mt_agnostic/dipper_paraphrase.py`
    - code: `mt_agnostic/coedit_{style}.py` where `style` can be coherent, formal, gec, paraphrase, understand

Each code accepts the following arguments:
  - `--model_name_hf`: The name or path of a transformers-based pre-trained checkpoint. You can directly refer to the Huggingface model. This argument is only required for 1 (Simplification) or 2 (Paraphrasing).
  - `--input_path`: Path to input data file
  - `--output_path`: Save path of output file (after rewriting)
  - `--cache_dir`: Cache directory of pre-trained model checkpoints


### Task-Aware Rewrite
We design prompts that account for the fact that rewrites are aimed at MT. Many prior works have shown that LLMs can post-edit errors in MT outputs and we were curious whether this ability can be extended to rewriting inputs to enhance translatability. We consider two prompting strategies:

1. **Easy translation:** prompt LLMs to rewrite inputs in a way that specifically facilitates translation in the target language.
    - code: `task_aware/easy_{llm}.py`

2. **Chain of thought (CoT):** prompt LLMs to handle the entire rewriting and translation process in one sequence.
    - code: `task_aware/cot_{llm}.py`


### Translation
We translate each generated rewrite into respective target language using `translate/translate_tower.py`.

```bash
python -u translate/translate_tower.py \
  --model_name_hf Unbabel/TowerInstruct-7B-v0.2 \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --tgt_lang $TARGET_LANGUAGE \
  --model_type $MODEL_TYPE \
  --cache_dir $PATH_TO_CACHE_DIR
```

Arguments for the translate code are as follows:
  - `--model_name_hf`: The name or path of a transformers-based pre-trained checkpoint. You can directly refer to the Huggingface model.
  - `--input_path`: Path to input data file.
  - `--output_path`: Save path of output file (after rewriting).
  - `--tgt_lang`: Target language (either German, Russian, or Chinese).
  - `--model_type`: Type of rewrite method (current code is set to simplification rewrite).
  - `--cache_dir`: Cache directory of pre-trained model checkpoints.


### Evaluation
We use [xCOMET](https://huggingface.co/Unbabel/XCOMET-XL) and [MetricX](https://github.com/google-research/metricx) to evaluate different aspects of rewrite quality. Using the two models, we examine three different evaluation metrics:
- **Translatability:** quality estimation (QE) score between the source and target
- **Meaning preservation:** QE score between the target and reference translation
- **Overall translation quality:** Reference-based score using source, target, and reference translation

We show an example of evaluation result for CoEdit style transfer evaluate. We can evaluate for other rewrite methods using the same code: `evaluate/xcomet_mt_coedit.py` (for translatability), `evaluate/xcomet_ref_coedit.py` (for meaning preservation), and `evaluate/xcomet_mtref_coedit.py` (for overall translation quality).

```bash
python -u evaluate/xcomet_mt_coedit.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --cache_dir $PATH_TO_CACHE_DIR
```

Arguments for the evaluation code are as follows:
  - `--input_path`: Path to input data file.
  - `--output_path`: Save path of output file (after evaluation).
  - `--cache_dir`: Cache directory of pre-trained model checkpoints.

---

<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
@inproceedings{ki-carpuat-2025-automatic,
    title = "Automatic Input Rewriting Improves Translation with Large Language Models",
    author = "Ki, Dayeon  and
      Carpuat, Marine",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.542/",
    doi = "10.18653/v1/2025.naacl-long.542",
    pages = "10829--10856",
    ISBN = "979-8-89176-189-6",
}
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
