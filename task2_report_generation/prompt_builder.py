# task2_report_generation/prompt_builder.py

def build_prompt(strategy="structured"):

    if strategy == "structured":
        return (
            "Analyze this chest X-ray and provide:\n"
            "1. Findings\n"
            "2. Impression\n"
            "3. Possible Diagnosis\n"
            "Also clearly state whether this is Normal or Pneumonia."
        )

    elif strategy == "simple":
        return "Describe the abnormalities in this chest X-ray."

    else:
        return "Provide a medical report for this image."
