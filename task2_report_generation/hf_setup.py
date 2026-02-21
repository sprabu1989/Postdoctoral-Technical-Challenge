# task2_report_generation/hf_setup.py

from huggingface_hub import login


def hf_login():
    print("Please enter your HuggingFace token:")
    login()
