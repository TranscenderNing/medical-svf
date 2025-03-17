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



def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer




def get_vllm_model() -> VLLMModel:
    """Load a vLLM model."""
    model = vllm.LLM(
        MODEL_ID,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="bfloat16",
    )
    # This may change with vLLM versions.
    m = model.llm_engine.model_executor.driver_worker.model_runner.model
    for _, param in m.named_parameters():
        param.requires_grad = False
    vllm_model = VLLMModel(
        model,
        sampling_params=vllm.SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=512,
            stop=["Instruction:", "Instruction", "Response:", "Response"],
            repetition_penalty=1.0,
        ),
        chat_template=LLAMA3,
    )
    return vllm_model


def get_evaluator() -> Tuple:
    res = []
    for split in ["train", "test"]:
        dataset = load_dataset("gsm8k", "main", split=split)
        samples = []
        for sample in dataset:
            answer = sample["answer"]
            answer = extract_answer_number(answer)
            answer = int(answer) if answer is not None else None
            samples.append(
                MathSample(
                    problem=sample["question"],
                    answer=answer,
                )
            )
        res.append(
            LanguageRestrictedMathTask(
                samples=samples,
                context_messages=[
                    fishfarm.Message("system", SYSTEM_MSG),
                ],
                languages=[],
            )
        )
    return tuple(res)


def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """Load weights from HF transformer model to vLLM model."""

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    num_layers = model.config.num_hidden_layers

    # Load embeddings layer weights.
    model_param = model.get_parameter("model.embed_tokens.weight")
    model_param.copy_(
        param["model.embed_tokens.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )
    model_param = model.get_parameter("lm_head.weight")
    model_param.copy_(
        param["lm_head.weight"][: model_param.shape[0]]
        .to(model_param.dtype)
        .to(model_param.device)
    )

    # Load the final layernorm weights.
    model_param = model.get_parameter("model.norm.weight")
    model_param.copy_(
        param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
    )

    for i in range(num_layers):
        # Load qkv_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.qkv_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.self_attn.q_proj.weight"],
                    param[f"model.layers.{i}.self_attn.k_proj.weight"],
                    param[f"model.layers.{i}.self_attn.v_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load gate_up_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.mlp.gate_up_proj.weight")
        model_param.copy_(
            torch.cat(
                [
                    param[f"model.layers.{i}.mlp.gate_proj.weight"],
                    param[f"model.layers.{i}.mlp.up_proj.weight"],
                ],
                dim=0,
            )
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load o_proj and down_proj weights.
        model_param = model.get_parameter(f"model.layers.{i}.self_attn.o_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.self_attn.o_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(f"model.layers.{i}.mlp.down_proj.weight")
        model_param.copy_(
            param[f"model.layers.{i}.mlp.down_proj.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        # Load layer_norm weights.
        model_param = model.get_parameter(f"model.layers.{i}.input_layernorm.weight")
        model_param.copy_(
            param[f"model.layers.{i}.input_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )
        model_param = model.get_parameter(
            f"model.layers.{i}.post_attention_layernorm.weight"
        )
        model_param.copy_(
            param[f"model.layers.{i}.post_attention_layernorm.weight"]
            .to(model_param.dtype)
            .to(model_param.device)
        )


def eval_model(vllm_model, evaluator, ix=None):
    result = evaluator.evaluate(vllm_model, sample_ids=ix)
    print(result.aggregate_metrics)
    return result


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
    
    ds = load_dataset("openai/gsm8k", "main")

    def modify_question(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['question']}
        ]
        example['question'] = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        return example
    
    test_ds = ds["test"]
    test_ds = test_ds.map(modify_question)
    batch_size = 8
    batched_test_ds = test_ds.batch(batch_size)

    learnable_params = torch.load(test_model_dir)
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
        for question, answer, raw_generation in zip(batch["question"],batch["answer"], response):
            answer = answer.split("####")[-1].strip()
            print('raw_generation',raw_generation)
            generation = extract_answer_number(sentence=raw_generation)
            print(answer, generation)
            if (
                abs(float(extract_answer_number(answer)) - generation)
                <= 0.001
            ):
                correct_count += 1
            results += [
                    {
                        "question": question,
                        "answer": answer,
                        "model_output": generation,
                    }
                ]
                

    print(f"Accuracy: {correct_count / len(test_ds)}")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return


def main(
    num_iters: int = 50,
    lr: float = 2e-3,
    batch_size: int = 8,
    seed: int = 42,
    case_num: int = 1,
    init_val: float = 0.1,
    test_only: bool = False,
    test_model_dir: str = "/data/ldn/self-adaptive-llms/medical/results/gsm8k/only-svf/20250312-145732/learnable_params_latest.pt",
    use_wandb: bool = False,
    custom_prefix: str = "骨科康复",
):
    """Main function."""

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:1"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_wandb:
        import wandb

        _ = wandb.init(
            project="proj-ayu-v2",
            name=f"{custom_prefix}_case_num_{case_num}-init_val_{init_val}-lr_{lr}-bs_{batch_size}",
            config={
                "lr": lr,
                "seed": seed,
                "batch_size": batch_size,
                "custom_prefix": custom_prefix,
                "init_val": init_val,
            },
        )

    # Create log dir.
    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d-%H%M%S")
    log_dir = f"results/medical/{custom_prefix}/{datetime_str}"
    os.makedirs(log_dir, exist_ok=True)

    
    json_file = f"/data/ldn/self-adaptive-llms/medical/data/5domains/train/{custom_prefix}.json"
    train_val_data = load_dataset('json', data_files=json_file, split="train")
    print(train_val_data)
    train_val_data = get_dataset(tokenizer, train_val_data)
    train_size = len(train_val_data)

    train_ix = range(0, train_size, 2)
    gpu = torch.device("cuda:1")
    np_random = np.random.RandomState(seed)

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
    learnable_params_list = list(learnable_params.values())
    optimizer = torch.optim.Adam(learnable_params_list, lr=lr)

    # inference and test
    if test_only:
        model.eval()
        test_model(
            test_model_dir, model, tokenizer, base_params, decomposed_params, learnable_params
        )
        exit(0)

    model.eval()
    for k in learnable_params:
        model.get_parameter(k).requires_grad_(True)

    # Training loop.
    for i in range(num_iters):
        batch_ix = np_random.choice(train_ix, size=batch_size, replace=False)

        # fetch data for this batch
        batch_inputs = [
            prepare_model_input(tokenizer, train_val_data, i) for i in batch_ix
        ]

        # Update model params
        forward(model, base_params, decomposed_params, learnable_params)

        step_loss = 0.0
        for batch_input in batch_inputs:
            input_ids, labels = batch_input
            input_ids = input_ids.unsqueeze(0).to(model.device)
            labels = labels.unsqueeze(0).to(model.device)

            # check the model input format match
            loss = model(input_ids=input_ids, labels=labels).loss
            step_loss += loss.item()
            loss.backward()

        # Backpropogate and update.
        backward(model, base_params, decomposed_params, learnable_params)

        grad_mean = learnable_params_list[0].grad.mean().item()
        torch.nn.utils.clip_grad_norm_(learnable_params_list, 1.0)  # use default value
        grad_norm_mean = learnable_params_list[0].grad.mean().item()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        # Inaccurate logging.
        print(
            f"Iter {i}, loss = {step_loss / batch_size} "
            + f"param={learnable_params_list[0].mean()}, "
            + f"grad={grad_mean}"
        )

    # 结束保存可训练参数
    forward(model, base_params, decomposed_params, learnable_params)
    torch.save(learnable_params, f"{log_dir}/learnable_params_latest.pt")


if __name__ == "__main__":
    fire.Fire(main)
