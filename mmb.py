import algo
import pandas as pd
import time
import numpy as np

from tqdm import tqdm

def benchmark(A, B, methods):
    result = []
    for method in methods:
        start = time.time()
        _ = method(A, B)
        end = time.time()

        runtime = start - end
        result.append(runtime)

    return result


def gen_matrix(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    return A, B

if __name__ == "__main__":
    methods = [algo.straightway, algo.strassen, algo.winograd]
    methods_names = ['Straight way', 'Strassen', 'Winograd']
    report = pd.DataFrame(columns=methods_names)

    for n in tqdm(range(20, 1000, 20)):
        A, B = gen_matrix(n)
        report.loc[n] = benchmark(A, B, methods)

    report.to_csv('report.csv')

