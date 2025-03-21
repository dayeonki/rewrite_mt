import datetime
import torch
import jsonlines
import argparse
from comet import download_model, load_from_checkpoint
from huggingface_hub.hf_api import HfFolder


hf_token = ""
HfFolder.save_token(hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="")
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    comet_model_path = download_model("Unbabel/XCOMET-XL", saving_directory=args.cache_dir)
    comet_model = load_from_checkpoint(comet_model_path).to(device)

    gecs, coherents, formals, understands, paraphrases = [], [], [], [], []

    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                ref = line["ref"]

                coedit_gec = line["coedit_gec"].strip()
                coedit_coherent = line["coedit_coherent"].strip()
                coedit_formal = line["coedit_formal"].strip()
                coedit_understand = line["coedit_understand"].strip()
                coedit_paraphrase = line["coedit_paraphrase"].strip()

                gec = {"src": coedit_gec, "mt": ref}
                gecs.append(gec)

                coherent = {"src": coedit_coherent, "mt": ref}
                coherents.append(coherent)

                formal = {"src": coedit_formal, "mt": ref}
                formals.append(formal)

                understand = {"src": coedit_understand, "mt": ref}
                understands.append(understand)

                paraphrase = {"src": coedit_paraphrase, "mt": ref}
                paraphrases.append(paraphrase)


    gec_output = comet_model.predict(gecs, batch_size=8, gpus=1)
    coherent_output = comet_model.predict(coherents, batch_size=8, gpus=1)
    formal_output = comet_model.predict(formals, batch_size=8, gpus=1)
    understand_output = comet_model.predict(understands, batch_size=8, gpus=1)
    paraphrase_output = comet_model.predict(paraphrases, batch_size=8, gpus=1)


    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for i, line in enumerate(file.iter()):
                line["xcomet_coedit_gec"] = round(float(gec_output.scores[i]), 3)
                line["xcomet_coedit_coherent"] = round(float(coherent_output.scores[i]), 3)
                line["xcomet_coedit_formal"] = round(float(formal_output.scores[i]), 3)
                line["xcomet_coedit_understand"] = round(float(understand_output.scores[i]), 3)
                line["xcomet_coedit_paraphrase"] = round(float(paraphrase_output.scores[i]), 3)
                outfile.write(line)
                print(line)
    
    print("---------------------------------------------------------------------------------\n")
    print("Coedit (GEC): ", round(float(sum(gec_output.scores) / len(gec_output.scores)), 3))
    print("Coedit (Coherent): ", round(float(sum(coherent_output.scores) / len(coherent_output.scores)), 3))
    print("Coedit (Formal): ", round(float(sum(formal_output.scores) / len(formal_output.scores)), 3))
    print("Coedit (Understand): ", round(float(sum(understand_output.scores) / len(understand_output.scores)), 3))
    print("Coedit (Paraphrase): ", round(float(sum(paraphrase_output.scores) / len(paraphrase_output.scores)), 3))
    print("---------------------------------------------------------------------------------\n")

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")
