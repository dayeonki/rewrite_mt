import torch
import argparse
import datetime
import jsonlines
import random
import os
from huggingface_hub.hf_api import HfFolder
from transformers import pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer

random.seed(24)

own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop_tokens=None, prompt_len=0):
        super().__init__()
        if stop_tokens is None:
            stop_tokens = []
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist) + i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen


def generate_text(pipe, tokenizer, prompt):
    stop_token = f"Original:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:],
                                                                  prompt_len=input_ids.shape[1])])
    return pipe(prompt,
                stopping_criteria=stopping_criteria,
                return_full_text=False)[0]["generated_text"][:-len(stop_token)].strip()


def prompt_template(src):
    prompt = f"Original: {src}\nParaphrase:"
    return prompt


def main():
    start_time = datetime.datetime.now()

    hf_token = ""
    HfFolder.save_token(hf_token)

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, help="Name of the model on hugging face")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="")

    args = parser.parse_args()
    hf_model_name = args.model_name_hf

    # =========================================== Load Model ===========================================
    dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "auto": "auto",
    }["bf16"]
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=args.cache_dir)
    pipe = pipeline(
        model=hf_model_name,
        device_map="auto",
        torch_dtype=dtype,
        min_new_tokens=20,
        max_new_tokens=512,
        tokenizer=tokenizer,
        model_kwargs={"temperature": 0.0, "do_sample": False}
    )

    # =========================================== Load Dataset ===========================================    
    generations = []
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                prompts = []
                src = line["src"]

                prompts.append(f"Paraphrase the English sentence. Try to not directly copy but keep the meaning of the Original sentence.")
                query_prompt = prompt_template(src)
                prompts.append(query_prompt)
                whole_input = "\n\n".join([item for item in prompts])
                print(whole_input)

    # =========================================== Generation =============================================
                generation = generate_text(pipe, tokenizer, whole_input)
                if "\n" in generation:
                    generation = generation.split("\n")[0]
                else: pass
                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"paraphrase_llama3"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()