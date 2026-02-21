# task2_report_generation/report_generator.py

import torch
from task2_report_generation.prompt_builder import build_prompt
from task2_report_generation.config import MAX_NEW_TOKENS


def generate_report(image, processor, model, strategy="structured"):

    prompt = build_prompt(strategy)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    input_len = inputs.input_ids.shape[1]
    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    return response
