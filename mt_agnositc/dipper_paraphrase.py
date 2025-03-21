import time
import datetime
import torch
import jsonlines
import nltk
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xl', cache_dir="")
        self.model = T5ForConditionalGeneration.from_pretrained(model, cache_dir="")
        
        self.model.cuda()
        self.model.eval()

        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")


    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of '0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="")
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                source_text = line["src"]
                prompt = ""

                # L80 + O60
                output_l80_o60 = dp.paraphrase(source_text, lex_diversity=80, order_diversity=60, prefix=prompt, do_sample=False, max_length=512)
                print(f"Input: {prompt} <sent> {source_text} </sent>\n")
                print(f"Output (L80, O60): {output_l80_o60}\n")

                # L80 + O40
                output_l80_o40 = dp.paraphrase(source_text, lex_diversity=80, order_diversity=40, prefix=prompt, do_sample=False, max_length=512)
                print(f"Output (L80, O40): {output_l80_o40}\n")

                # L60 + O40
                output_l60_o40 = dp.paraphrase(source_text, lex_diversity=60, order_diversity=40, prefix=prompt, do_sample=False, max_length=512)
                print(f"Output (L60, O40): {output_l60_o40}\n")
                print("---------------------------------------------------------------------------------\n")

                line["dipper_l80_o60"] = output_l80_o60
                line["dipper_l80_o40"] = output_l80_o40
                line["dipper_l60_o40"] = output_l60_o40

                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")
