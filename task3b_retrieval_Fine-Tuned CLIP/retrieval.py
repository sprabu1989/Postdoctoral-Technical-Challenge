# task3_clip_finetuned_retrieval/retrieval.py

def image_search(query_idx, embeddings, index, k=5):

    query = embeddings[query_idx].reshape(1, -1)
    distances, indices = index.search(query, k+1)

    return indices[0][1:], distances[0][1:]


def text_search(query_text, processor, model, index, device, k=5):

    inputs = processor(text=[query_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)

    text_features = text_features / text_features.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    text_features = text_features.cpu().numpy().astype("float32")

    distances, indices = index.search(text_features, k)

    return indices[0], distances[0]
