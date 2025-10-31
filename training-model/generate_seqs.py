import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from modeling_hyena import StripedHyenaModelForCausalLM

from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast
)

random.seed(42)
np.random.seed(42)
set_seed(42)
transformers_set_seed(42)

checkpoint = 'model/mach-1-v00/checkpoint-16384'
model = StripedHyenaModelForCausalLM.from_pretrained(checkpoint)
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

# Move model to device and set eval mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Prepare EOS/PAD ids
eos_id = tokenizer.convert_tokens_to_ids("E")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = eos_id

generations_dir = "results/generations"
os.makedirs(generations_dir, exist_ok=True)

batch_size = 1
index = 1

if __name__ == "__main__":
    try:
        while True:
            prompt = "HS"

            # Encode prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Generate sequences directly with the model
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    num_return_sequences=batch_size,
                    max_new_tokens=65536,
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode generated sequences
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            output_file = os.path.join(generations_dir, f"generated_batch_{index}.txt")
            with open(output_file, "w") as f:
                for text in texts:
                    f.write(''.join(text) + "\n")
            index += 1

    except KeyboardInterrupt:
        print("Process interrupted.")
