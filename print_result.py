from PIL import Image
import os
os.system('clear')
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from tqdm import tqdm
import argparse

def compute_top_k_accuracy(results, k_list=[1, 5]):
    """
    Computes the top-k accuracy for the result indices.
    
    Args:
        results (list): A list of dictionaries containing 'qa_list', 'indicies', and 'all_page_md_str'.

    Returns:
        dict: A dictionary where keys are k values, and values are corresponding accuracy scores.
    """
    top_k_acc = {k: [] for k in k_list}

    for meta in results:
        true_index = [x['page_index'] for x in meta['qa_list']]
        # file_name = meta['file_name']

        result_index = meta['indicies']  # Predicted ranked lists

        for k in k_list:
            if isinstance(result_index, list) and (not result_index or all(isinstance(sublist, list) and not sublist for sublist in result_index)):
                continue
            acc = [true_i in pred[:k] for true_i, pred in zip(true_index, result_index)]
            top_k_acc[k] += acc

    # Compute final average accuracy for each k
    top_k_acc = {k: np.mean(v) for k, v in top_k_acc.items()}

    for k in k_list:
        print(f'top-{k} accuracy: {100 * top_k_acc[k]:.2f} %')


    return f'{100 * top_k_acc[1]:.2f} & {100 * top_k_acc[5]:.2f}'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default='/Users/chenjian/Projects/M4Doc/M4Doc_local', help="work_dir", type=str)
    args = parser.parse_args()

    for retriever in ['CLIP', 'SigLIP', 'VLM2vec', 'VisRAG', 'GEM', 'ColPhi', 'ColPali', 'ColInternVL2']:
        for qa_type in ['table', 'text', 'figure']:
            print('>' * 100)
            print(f'{retriever=}, {qa_type=}')
            results = json.load(open(os.path.join(args.work_dir, f'results/{retriever}_{qa_type}.json'), 'r'))
            acc = compute_top_k_accuracy(results, k_list=[1, 5])

    print('>' * 100)
    print('text only')
    print('>' * 100)

    for retriever in ['BM25', 'SBERT', 'bge_large', 'BGE_M3', 'NV-Embed']:
        for qa_type in ['table', 'text', 'figure']:
            print('>' * 100)
            print(f'{retriever=}, {qa_type=}')
            results = json.load(open(os.path.join(args.work_dir, f'results/{retriever}_{qa_type}.json'), 'r'))
            acc = compute_top_k_accuracy(results, k_list=[1, 5])
            