import pandas as pd
#######################################################################################################################
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

device = 'cuda'

# load the pretrained base_model and hmm_model;
BASE_MODEL_PATH = f'ctrlg/tulu2-7b_writing-prompts'
HMM_MODEL_PATH = f'ctrlg/hmm_tulu2-7b_writing-prompts_32768'

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
base_model.half() # fp16 inference
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)

#######################################################################################################################
## MY VERSION ##
vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id

data = pd.read_csv('/media/data/bazaluk/ctrlg_tulu2/data.csv')
data['pred_graphs'] = ''

for i in range(3):
#for i in range(len(data)):
    prefix = data['prefix'].iloc[i]
    d_prompt = data['prompt'].iloc[i]
    suffix = '</s>'
    soft_constraint = '' # use empty string for no soft constraint
    prompt = f'<|user|>\n"{d_prompt}<|endoftext|>"{soft_constraint}:\n{prefix}\n<|assistant|>\n'

    prefix_ids = tokenizer.encode(prefix)[1:]
    suffix_ids = tokenizer.encode(suffix)[1:]
    prompt_ids = tokenizer.encode(prompt)

    ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
    eos_builder = ctrlg.EOSBuilder(vocab_size, eos_token_id)

    dfa_graphs = []

    keyphrases = [[' X ', ' Y ', ' V2 ', ' V3 ', ' V4 ', ' V5 '],
                [' ->'],
                 [' X ', ' Y ', ' V2 ', ' V3 ', ' V4 ', ' V5 ']]

    for keyphrase in keyphrases:
        patterns = [tokenizer.encode(x)[1:] for x in keyphrase]
        dfa_graphs.append(ac_builder.build(patterns))
    dfa_graphs.append(eos_builder.build())

    dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode='intersection')
    dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device)

    min_new_tokens = 1
    max_new_tokens = 32
    
    ############################################################################
    # initialze the constraints logits processor
    constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
        hmm_model, dfa_model,
        min_new_tokens, max_new_tokens,
        prompt_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids)


    # set the hmm_batch_size & temperature
    beam_size = 64 # sample 128 sequences
    temperature = 0.7
    constraint_logits_processor.hmm_batch_size = beam_size
    constraint_logits_processor.temperature = temperature


    # generate with sampling, temperature=0.7
    input_ids = torch.tensor([prompt_ids], device=device)
    outputs = base_model.generate(
            input_ids=input_ids, do_sample=True,
            num_return_sequences=beam_size, 
            min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
            logits_processor=LogitsProcessorList([constraint_logits_processor]),
            pad_token_id=tokenizer.eos_token_id,
        )


    # extract the generated ids; removing prompt ids; remove suffix ids that are (partially) generated
    generated_ids = ctrlg.extract_generated_ids(outputs.tolist(), prompt_ids, suffix_ids, eos_token_id)

    # filter 75% of the generated ids by how well they connect with the suffix
    generated_ids = ctrlg.rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids,
                                                suffix_logits_only=True, suffix_length_cap=5)[:32]
    # rank the generated ids by the base_model for higher quality
    generated_ids = ctrlg.rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids)

    # save top 10 outputs
    pred_graphs = []
    for idx, generated in enumerate(generated_ids[:10]):
        pred_graphs.append(f'{idx}. ' + tokenizer.decode(generated, skip_special_tokens=True) + \
              tokenizer.decode(suffix_ids, skip_special_tokens=True))
    data.at[i, 'pred_graphs'] = pred_graphs

data.to_csv('teste.csv')