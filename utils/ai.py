import numpy as np
import openai
from numpy import ndarray
from tenacity import retry, wait_random_exponential, stop_after_attempt
from scipy.spatial.distance import cosine


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def vector_similarity(x: list[float], y: list[float]) -> ndarray:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def vector_similarity_2(x: list[float], y: list[float]) -> float:
    return 1 - cosine(x, y)


if __name__ == '__main__':
    import ast
    import time
    q = 'How can we use pre-trained language models?'
    # qe = get_embedding(q)
    with open('qe.txt', 'r') as f:
        qe_str = f.read()

    qe = np.array(ast.literal_eval(qe_str), dtype='float32')

    with open('ee.txt', 'r') as f:
        ee_str = f.read()

    ee = np.array(ast.literal_eval(ee_str), dtype='float32')

    t1_start = time.process_time()
    for i in range(100000):
        a = vector_similarity(qe, ee)
    t1_stop = time.process_time()
    print("Elapsed time:", t1_stop-t1_start)
    print(a)

    t1_start = time.process_time()
    for i in range(100000):
        a = vector_similarity_2(qe, ee)
    t1_stop = time.process_time()
    print("Elapsed time:", t1_stop - t1_start)
    print(a)