# task2_report_generation/evaluator.py

from tqdm import tqdm


def extract_prediction(report):

    report_lower = report.lower()

    if "pneumonia" in report_lower:
        return 1
    elif "normal" in report_lower:
        return 0
    else:
        return -1


def evaluate_subset(test_dataset, processor, model, N=50):

    results = []

    for i in tqdm(range(N)):
        img, label = test_dataset[i]

        report = generate_report(
            image=img,
            processor=processor,
            model=model,
            strategy="structured"
        )

        pred = extract_prediction(report)

        results.append({
            "index": i,
            "ground_truth": int(label),
            "vlm_prediction": pred,
            "report": report
        })

    return results
