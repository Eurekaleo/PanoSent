import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def create_verification_prompt(data, prompt_template):
    description = prompt_template.format(**data)
    return description


def verify_claims(data, model, tokenizer, prompt_template):
    prompt = create_verification_prompt(data, prompt_template)
    verification_result = model(prompt, tokenizer)
    return verification_result


def execute_cos_reasoning_with_ppv(
    dialogue, multimodal_data, model, tokenizer, stage_data
):
    prompt_templates = [
        "In this dialogue, participants discussed various targets and their corresponding aspects, including {aspects}. Please based on the dialogue, verify whether these descriptions are consistent with the dialogue content and provide `1' for `yes' or `0' for `no' judgment.",
        "In this dialogue, different participants expressed their opinions towards various aspects of targets, including {opinions}. Please based on the dialogue, verify whether these descriptions are consistent with the dialogue content and provide `1' for `yes' or `0' for `no' judgment.",
        "In this dialogue, the analysis has identified sentiments and rationales behind opinions, including {sentiments}. Please based on the dialogue, verify whether these descriptions are consistent with the dialogue content and provide `1' for `yes' or `0' for `no' judgment.",
        "In this dialogue, instances of sentiment flipping and their triggers have been identified, including {flips}. Please based on the dialogue and your commonsense knowledge, verify whether these descriptions accurately capture the emotional dynamics and their triggers in the dialogue and provide `1' for `yes' or `0' for `no' judgment.",
    ]

    verification_results = []
    for index, data in enumerate(stage_data):
        result = verify_claims(data, model, tokenizer, prompt_templates[index])
        verification_results.append(result)
        if not result:
            print("Inconsistency found, needs re-evaluation or manual correction")
            continue

    return verification_results


def handle_inconsistency(data, model, tokenizer, prompt_template):
    print("Regenerating k-tuples for:", data)
    regenerated_data = regenerate_data(data)
    new_prompt = create_verification_prompt(regenerated_data, prompt_template)
    new_result = model(new_prompt, tokenizer)
    return new_result


def regenerate_data(data):
    return data


stage_data = [
    {"aspects": "a1 of t1, a2 of t2"},
    {"opinions": "o1 of h1 on a1 of t1, o2 of h2 on a2 of t2"},
    {"sentiments": "s1 with r1 of o1 on a1 of t1"},
    {"flips": "z1 to f1 due to t1 of h1 on a1 of t1"},
]
verification_results = execute_cos_reasoning_with_ppv(
    "Example Dialogue", None, model, tokenizer, stage_data
)
