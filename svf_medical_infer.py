import fire
import os
import torch
import torch.utils
import vllm
import json
import numpy as np
from typing import Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from datetime import datetime
import re

import fishfarm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.language_restricted_math import (
    LanguageRestrictedMathTask,
    MathSample,
)

from torch.nn.utils.rnn import pad_sequence

MODEL_ID = "/data/ldn/llm-models/Meta-Llama-3.1-8B-Instruct"
DECOMPOSED_PARAM_FILE = "/data/ldn/self-adaptive-llms/medical/llama3_decomposed_params.pt"
LLAMA3 = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
    "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
SYSTEM_MSG = (
    "Below is an instruction that describes a task."
    " Write a response that appropriately completes the request.\n\n"
)
CASE_NUM = 1



dispatch_dir =  "/data/ldn/self-adaptive-llms/medical/results/medical/dispatch/20250313-101649/learnable_params_latest.pt"
test_model_dir = {
    "骨科康复": "/data/ldn/self-adaptive-llms/medical/results/medical/骨科康复/20250313-092057/learnable_params_latest.pt",
    "脊髓损伤康复":"/data/ldn/self-adaptive-llms/medical/results/medical/脊髓损伤康复/20250313-083548/learnable_params_latest.pt",
    "内科康复":"/data/ldn/self-adaptive-llms/medical/results/medical/内科康复/20250313-090647/learnable_params_latest.pt",
    "言语、吞咽康复":"/data/ldn/self-adaptive-llms/medical/results/medical/言语、吞咽康复/20250313-094110/learnable_params_latest.pt",
    "卒中康复":"/data/ldn/self-adaptive-llms/medical/results/medical/卒中康复/20250313-095202/learnable_params_latest.pt",
}

def get_mask(p):
    return torch.sigmoid(p)


def compose_new_params(param_name, decomposed_params, learnable_params):
    mm = get_mask(learnable_params[param_name])
    return (
        decomposed_params[f"{param_name}.U"]
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)
        @ decomposed_params[f"{param_name}.V"].T
    ) * (
        decomposed_params[f"{param_name}.S"].sum()
        / (decomposed_params[f"{param_name}.S"] * mm).sum()
    )


@torch.no_grad()
def forward(model, base_params, decomposed_params, learnable_params):
    """Forward pass."""
    new_params = {}
    for k in base_params:
        # target_modules=['self_attn.q_proj', 'self_attn.v_proj']
        if "mlp" in k:
            new_params[k] = compose_new_params(k, decomposed_params, learnable_params)
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


def backward(model, base_params, decomposed_params, learnable_params):
    """Backward pass."""
    for k in base_params:
        if "mlp" in k:
            compose_new_params(k, decomposed_params, learnable_params).backward(
                model.get_parameter(k).grad
            )


def prepare_model_input(tokenizer, train_data, idx):
    """Return input_ids and label of batch"""
    text = train_data["text"][idx]
    input_ids = tokenizer.encode(text)
    response_template_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False
    )
    input_ids = torch.tensor(input_ids)
    response_template_ids = torch.tensor(response_template_ids)

    # Find where the full template sequence starts
    template_start = None
    for i in range(len(input_ids) - len(response_template_ids) + 1):
        if torch.all(
            input_ids[i : i + len(response_template_ids)] == response_template_ids
        ):
            template_start = i + len(response_template_ids)
            break

    labels = torch.full_like(input_ids, -100)
    labels[template_start:] = input_ids[template_start:]
    return input_ids, labels


def get_dataset(tokenizer, samples, ixs=None):
    context_msg = {"role": "system", "content": SYSTEM_MSG}
    if ixs is None:
        ixs = range(len(samples))
    lines = []
    for ix in ixs:
        user_msg = {"role": "user", "content": samples["question"][ix]}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=LLAMA3,
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = samples["answer"][ix]
        lines.append(prompt + answer)

    dataset = Dataset.from_dict({"text": lines})
    return dataset


def test_model(test_model_dir, model, tokenizer, base_params, decomposed_params, learnable_params):
    
    json_file = "/data/ldn/self-adaptive-llms/medical/data/5domains/test/dispatch.json"
    test_ds = load_dataset('json', data_files=json_file, split="train")

    def modify_question(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['question']}
        ]
        example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        return example
    
     
    test_ds = test_ds.map(modify_question)
    
    
    # 第一次推理
    batch_size = 8
    batched_test_ds = test_ds.batch(batch_size)

    learnable_params = torch.load(dispatch_dir)
    print("Learnable params loaded.")
    forward(model, base_params, decomposed_params, learnable_params)
    
    correct_count = 0
    results = []
    print("anwser and generation")
    for batch in batched_test_ds:
        model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda:1')
        generated_ids = model.generate(**model_inputs,max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for question, answer, answer1, raw_generation in zip(batch["question"],batch["answer"],batch["answer1"], response):
            answer = answer.split("####")[-1].strip()
            answer = answer.replace(".json","")
            print('raw_generation',raw_generation)
            generation = raw_generation.replace(".json","")
            print(answer, generation)
            if generation == answer:
                correct_count += 1
            results += [
                    {
                        "question": question,
                        "answer": answer,
                        "answer1": answer1,
                        "model_output": generation,
                    }
                ]
                

    print(f"Accuracy: {correct_count / len(test_ds)}")
    with open("results_dispatch.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return

def test_model_5domain(test_model_dir, model, tokenizer, base_params, decomposed_params, learnable_params):
    batch_size = 8
    
    json_file = "/data/ldn/self-adaptive-llms/medical/results_dispatch.json"
    test_ds = load_dataset(path='json', data_files=json_file, split="train")
    guke_ds = test_ds.filter(lambda x: "骨科康复" in x['model_output'])
    jisui_ds = test_ds.filter(lambda x: "脊髓损伤康复" in x['model_output'])
    neike_ds = test_ds.filter(lambda x: "内科康复" in x['model_output'])
    yanyu_ds = test_ds.filter(lambda x: "言语、吞咽康复" in x['model_output'])
    zuzhong_ds = test_ds.filter(lambda x: "卒中康复" in x['model_output'])
    total_size = len(guke_ds) + len(jisui_ds) + len(neike_ds) + len(yanyu_ds) + len(zuzhong_ds)
    all_ds = {
        "骨科康复": guke_ds,
        "脊髓损伤康复": jisui_ds,
        "内科康复": neike_ds,
        "言语、吞咽康复": yanyu_ds,
        "卒中康复": zuzhong_ds,
    }


    def modify_question(example):# -> Any:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['question']}
        ]
        example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        return example
    

    correct_count = 0
    results = []
    for domain, path in test_model_dir.items():
        print(f"Testing {domain}...")
        # load model
        learnable_params = torch.load(path)
        print("Learnable params loaded.")
        forward(model, base_params, decomposed_params, learnable_params)
        # load data
        test_ds = all_ds[domain]
        test_ds = test_ds.map(modify_question)
        batched_test_ds = test_ds.batch(batch_size)
        print("anwser and generation")
        for batch in batched_test_ds:
            model_inputs = tokenizer(batch["question"], return_tensors="pt", padding="longest").to('cuda:1')
            generated_ids = model.generate(**model_inputs,max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for question, answer, answer1, raw_generation in zip(batch["question"],batch["answer"],batch["answer1"], response):
                answer1 = answer1.split("####")[-1].strip()
                print('raw_generation',raw_generation)
                generation = raw_generation
                print(answer1, generation)
                if answer1 in generation:
                    correct_count += 1
                results += [
                        {
                            "question": question,
                            "answer": answer,
                            "answer1": answer1,
                            "model_output_1": generation,
                        }
                    ]
                

    print(f"Accuracy: {correct_count / total_size}")
    with open("results_5domain.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    return


def main(
    num_iters: int = 50,
    lr: float = 2e-3,
    batch_size: int = 8,
    seed: int = 42,
    case_num: int = 1,
    init_val: float = 0.1,
    test_only: bool = True,
    use_wandb: bool = False,
    custom_prefix: str = "骨科康复",
) :


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:1"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"



    gpu = torch.device("cuda:1")


    # Load model and tokenizer.
    base_params = model.state_dict()
    # Load decomposed parameters.
    if not os.path.exists(DECOMPOSED_PARAM_FILE):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            if "mlp" in k:
                print(k)
                U, S, V = torch.svd(v.to(torch.float32))
                decomposed_params[f"{k}.U"] = U
                decomposed_params[f"{k}.S"] = S
                decomposed_params[f"{k}.V"] = V
        torch.save(decomposed_params, DECOMPOSED_PARAM_FILE)
    else:
        print("Decomposed params found. Loading...")
        decomposed_params = torch.load(DECOMPOSED_PARAM_FILE)
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(gpu)

    # Create learnable parameters.
    learnable_params = {}
    num_params = 0
    for k, v in base_params.items():
        if "mlp" in k:
            learnable_params[k] = torch.nn.Parameter(
                data=(
                    torch.randn(
                        min(v.shape),
                        device=gpu,
                        dtype=torch.bfloat16,
                    )
                    * 0.01
                    + init_val
                    # torch.ones(min(v.shape), device=gpu, dtype=torch.bfloat16)
                ),
                requires_grad=True,
            )
            num_params += learnable_params[k].numel()
    print(f"#params={num_params}")

    # inference and test
    if test_only:
        model.eval()
        # test_model(
        #     test_model_dir, model, tokenizer, base_params, decomposed_params, learnable_params
        # )
        test_model_5domain(test_model_dir, model, tokenizer, base_params, decomposed_params, learnable_params)
        exit(0)




if __name__ == "__main__":
    fire.Fire(main)
