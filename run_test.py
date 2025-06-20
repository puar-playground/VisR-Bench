from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from tqdm import tqdm
os.system('clear')
from utils.retrieval_model_util import CLIPRetriever, SigLIPRetriever, NVEmbedMultiRetriever, NVEmbedRetriever
from utils.retrieval_model_util import SentenceBERTRetriever, BGE_M3Retriever, BGE_LargeRetriever, BM25Retriever
from utils.retrieval_model_util import ColPhiRetriever, ColInternVL2Retriever, ColPaliRetriever, VLM2VecRetriever, GmeRetriever, VisRAGRetriever
import argparse
from print_result import compute_top_k_accuracy


parser = argparse.ArgumentParser()
parser.add_argument("--retriever", default='BM25', 
                    choices=['CLIP', 'SigLIP', 'BM25', 'SBERT', 'bge_large', 'BGE_M3', 
                             'VLM2vec', 'VisRAG', 'NV-Embed',
                             'ColPali', 'ColPhi', 'ColInternVL2', 'GEM'], 
                    help="retriever name", 
                    type=str)
parser.add_argument("--type", default='multilingual', choices=['figure', 'table', 'text', 'multilingual'], help="QA type", type=str)
parser.add_argument("--work_dir", default='./', help="/path/to/code/folder", type=str)
args = parser.parse_args()


if __name__ == "__main__":

    print(f'{args.retriever}, {args.type}')
    
    if args.retriever == 'CLIP':
        retriever = CLIPRetriever(checkpoint="ViT-L/14")
    elif args.retriever == 'SigLIP':
        retriever = SigLIPRetriever(checkpoint="google/siglip-so400m-patch14-384")
    elif args.retriever == 'NV-Embed':
        retriever = NVEmbedRetriever(model_name="nvidia/NV-Embed-v2")
    elif args.retriever == 'BM25':
        retriever = BM25Retriever()
    elif args.retriever == 'SBERT':
        retriever = SentenceBERTRetriever()
    elif args.retriever == 'BGE_M3':
        retriever = BGE_M3Retriever()
    elif args.retriever == 'bge_large':
        retriever = BGE_LargeRetriever()
    elif args.retriever == 'GEM':
        retriever = GmeRetriever()
    elif args.retriever == 'VLM2vec':
        retriever = VLM2VecRetriever()
    elif args.retriever == 'VisRAG':
        retriever = VisRAGRetriever()
    elif args.retriever == 'ColPali':
        retriever = ColPaliRetriever()
    elif args.retriever == 'ColPhi':
        model_checkpoint = 'puar-playground/Col-Phi-3-V'
        retriever = ColPhiRetriever(model_name=model_checkpoint)
    elif args.retriever == 'ColInternVL2':
        model_checkpoint = 'puar-playground/Col-InternVL2-4B'
        retriever = ColInternVL2Retriever(model_name=model_checkpoint)

    qa_data = json.load(open(f'./QA/{args.type}_QA.json', 'r'))

    top_k = 1

    acc_all = []
    pbar = tqdm(qa_data)

    results = []
    for qa in pbar:

        file_name = qa['file_name']
        question_list = [x['question'] for x in qa['qa_list']]
        page_index_list = [x['page_index'] for x in qa['qa_list']]

        assert len(qa['all_page_images']) == len(qa['all_page_md_str'])

        image_list = [os.path.join(args.work_dir, 'data', file_name, x) for x in qa['all_page_images']]

        if retriever.multimodel:
            _, indicies = retriever.retrieve(query_list=question_list, image_list=image_list)
            indicies = indicies.tolist()
        else:
            _, indicies_raw = retriever.retrieve(query_list=question_list, md_list=qa['all_page_md_str'])
            indicies = []
            for ind in indicies_raw.tolist():
                indicies.append([x for x in ind if qa['all_page_md_str'][x] != 'None'])

        qa['indicies'] = indicies
        results.append(qa.copy())

        # break

    json.dump(results, open(os.path.join(args.work_dir, f'results/{args.retriever}_{args.type}.json'), 'w'), indent=2)

    acc = compute_top_k_accuracy(results, k_list=[1, 5])