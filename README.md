# Automatic Input Rewriting Improves Translation with Large Language Models

This repository contains the code and dataset for our NAACL 2025 Main paper **Automatic Input Rewriting Improves Translation with Large Language Models**.

<div align="center">
<img src="https://github.com/user-attachments/assets/b3415a65-ccac-4468-a291-07602cb95509" style="width: 15px;" alt="code"> <b><a href=https://github.com/dayeonki/rewrite_mt>Code</a></b> | <img src="https://github.com/user-attachments/assets/2bd9af9b-2182-4aef-83cd-6e9ca6189a39" style="width: 15px;" alt="data">
 <b><a href=https://huggingface.co/datasets/zoeyki/rewrite_mt_dataset>Dataset</a></b> | <img src="https://github.com/user-attachments/assets/fc2ca3c2-3e78-4ca4-a208-448c0a6c7068" style="width: 15px;" alt="paper"> <b><a href=https://arxiv.org/pdf/2502.16682>Paper</a></b>
</div>


## Abstract
Can we improve machine translation (MT) with LLMs by rewriting their inputs automatically? Users commonly rely on the intuition that wellwritten text is easier to translate when using off-the-shelf MT systems. LLMs can rewrite text in many ways but in the context of MT, these capabilities have been primarily exploited to rewrite outputs via post-editing. We present an empirical study of 21 input rewriting methods with 3 open-weight LLMs for translating from English into 6 target languages. We show that text simplification is the most effective MT-agnostic rewrite strategy and that it can be improved further when using quality estimation to assess translatability. Human evaluation further confirms that simplified rewrites and their MT outputs both largely preserve the original meaning of the source and MT. These results suggest LLM-assisted input rewriting as a promising direction for improving translations.

<p align="center">
<img width="800" alt="Screenshot 2025-03-21 at 1 48 51 PM" src="https://github.com/user-attachments/assets/af4275ae-47e1-4297-b585-64881d38c3f2" />
</p>


## Quick Links
- [Overview](#overview)
- [Preliminaries](#preliminaries)
- [MT-Agnostic](#mt-agnostic)
- [Task-Aware](#task-aware)
- [Translate](#translate)
- [Evaluate](#evaluate)


## Overview
In this work, we shed light on these issues by asking the following questions:

- Can we improve MT quality from LLMs by rewriting inputs for style?
- How can we guide models to rewrite inputs to improve their translatability? 

In order to answer these questions, we conduct an empirical study with 21 input rewriting methods with varying levels of MT-awareness on translation. As shown in the figure, given a triplet of source sentence, translation, and its reference translation, we first rewrite the source sentence using different types of rewriting methods, translate each rewrite in the target language, and then evaluate the rewrites on the basis of translatability and meaning preservations.

The following figure shows an example of two ways of improving MT. Given a pipeline of a source text going into the MT system and getting the machine translation output, one way to improve MT quality is **pre-editing** the source text before passing into the MT system and the other way is taking the MT output and **post-editing**. We explore the first variant:
<p align="center">
<img width="600" alt="Screenshot 2025-03-21 at 1 54 40 PM" src="https://github.com/user-attachments/assets/c9f9c78b-158f-4735-ac87-4a08f20b2189" />
</p>

## Preliminaries
### [1] Dataset
We use the WMT-23 General MT task from [Tower-Eval](https://huggingface.co/datasets/Unbabel/TowerEval-Data-v0.1) dataset. For the main experiments, we focus on three language pairs: English-German (en-de), English-Russian (en-ru), and English-Chinese (en-zh). The dataset is in the following: `data/raw_{language_pair}.jsonl`.

### [2] Translated dataset
We translate each source sentence in English to the respective target language using Tower-Instruct LLM. The translated dataset is in the following: `data/{language_pair}_mt_tower.jsonl`.

## MT-Agnostic
Within the process of source rewriting, the goal of a rewrite model is _to rewrite the original source sentence into another form that is easier to translate while preserving its intended meaning_. MT-Agnostic methods only reflect various prior assumptions on what makes text easier to translate and do not take as input any signal of translatability or knowledge about the end-task. We consider three prompting variants here, all inspired by prior works on source rewriting:

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

## Task-Aware
Next, we design prompts that account for the fact that rewrites are aimed at MT. Many prior works have shown that LLMs can post-edit errors in MT outputs and we were curious whether this ability can be extended to rewriting inputs to enhance translatability. We consider two prompting strategies:
1. **Easy translation:** prompt LLMs to rewrite inputs in a way that specifically facilitates translation in the target language.
    - code: `task_aware/easy_{llm}.py`

2. **Chain of thought (CoT):** prompt LLMs to handle the entire rewriting and translation process in one sequence.
    - code: `task_aware/cot_{llm}.py`


## Translate
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

## Evaluate
We use [xCOMET](https://huggingface.co/Unbabel/XCOMET-XL) and [MetricX](https://github.com/google-research/metricx) to evaluate different aspects of rewrite quality. Using the two models, we examine three different evaluation metrics:
- **Translatability:** quality estimation (QE) score between the source and target
- **Meaning preservation:** QE score between the target and reference translation
- **Overall translation quality:** Reference-based score using source, target, and reference translation

<p align="center">
<img width="550" alt="Screenshot 2025-03-21 at 2 39 07 PM" src="https://github.com/user-attachments/assets/e0dfc8dc-00e9-4611-a142-a4c63295b378" />
</p>

We show an evaluation example for CoEdit style transfer evaluate. We can evaluate for other rewrite methods using the same code: `evaluate/xcomet_mt_coedit.py` (for translatability), `evaluate/xcomet_ref_coedit.py` (for meaning preservation), and `evaluate/xcomet_mtref_coedit.py` (for overall translation quality).

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


## Citation
```
@misc{ki2025automaticinputrewritingimproves,
      title={Automatic Input Rewriting Improves Translation with Large Language Models}, 
      author={Dayeon Ki and Marine Carpuat},
      year={2025},
      eprint={2502.16682},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.16682}, 
}
```
