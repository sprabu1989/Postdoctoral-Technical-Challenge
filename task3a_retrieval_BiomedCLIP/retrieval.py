# task3_biomedclip_retrieval/retrieval.py

def search_image(query_idx, embeddings, index, k=5):

    query = embeddings[query_idx:query_idx + 1]
    D, I = index.search(query, k + 1)

    return I[0][1:], D[0][1:]
