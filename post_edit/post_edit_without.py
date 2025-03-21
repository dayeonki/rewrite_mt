import torch
import os
from transformers import pipeline
import datetime
import argparse
import jsonlines

def remove_before(text, word):
    parts = text.split(word, 1)
    return parts[1] if len(parts) > 1 else text


if __name__ == "__main__":

    own_cache_dir = ""
    os.environ["HF_HOME"] = own_cache_dir
    os.environ["HF_DATASETS"] = own_cache_dir

    start_time = datetime.datetime.now()

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--target_lang", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--cache_dir", type=str, default="")

    args = parser.parse_args()
    hf_model_name = "Unbabel/TowerInstruct-7B-v0.2"

    pipe = pipeline("text-generation", model=hf_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                prompts = []
                mt = line[f"mt_{args.model_type}"]
                
                prompt = f"Rewrite the Original sentence. Keep the New sentence in {args.target_lang}.\n\nOriginal: {mt}\nNew:"
                print(prompt)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
                generation = outputs[0]["generated_text"]

                if "<|im_start|>assistant\n " in generation:
                    keyword = "<|im_start|>assistant\n "
                    generation = remove_before(generation, keyword)
                else: pass

                print(f"> {generation}")
                print("\n==================================\n")

                line[f"postedit_mt_{args.model_type}"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")
