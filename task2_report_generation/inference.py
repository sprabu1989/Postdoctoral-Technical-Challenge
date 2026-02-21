# task2_report_generation/inference.py

from task2_report_generation.report_generator import generate_report
from task2_report_generation.evaluator import extract_prediction


def run_single_inference(image, processor, model):

    report = generate_report(image, processor, model)
    pred = extract_prediction(report)

    return report, pred
