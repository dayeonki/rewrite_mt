from transformers import AutoTokenizer, T5ForConditionalGeneration
import argparse
import jsonlines
import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-xl", cache_dir="")
    model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-xl", cache_dir="")

    start_time = datetime.datetime.now()

    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                source_text = line["src"]
                
                gec_prompt = "Fix grammatical errors in this sentence: "

                gec = gec_prompt + source_text
                print(gec)
                input_ids = tokenizer(gec, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_length=256)
                gec_generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"> {gec_generation}")
                print("---------------------------------------------------------------------------------\n")

                line["coedit_gec"] = gec_generation

                outfile.write(line)

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")