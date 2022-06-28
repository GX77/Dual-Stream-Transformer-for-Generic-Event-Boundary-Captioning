import sys
import json
import pdb
import traceback
from bdb import BdbQuit
import argparse
import os

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.eval import EvalCap

def save_json(path,data):
    with open(path, "w") as f:
        return json.dump(data, f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def classify(data):
    index = {}
    index['BEF'] = []
    index['AFT'] = []
    index['SUB'] = []
    ann = {}
    for name,value in data.items():
        mark = value['sentence'][1:4]
        ann[name] = [value['sentence'][6:]]
        index[mark].append(int(name))
    return index,ann

def evaluate(args):
    tokenizer = PTBTokenizer  # for English
    annos = load_json(args.references[0])
    index,annos = classify(annos)
    data = json.load(open(args.submission, 'r'))['results']
    rests = []
    for name, value in data.items():
        temp = {}
        temp['image_id'] = str(name)
        temp['caption'] = value[0]['sentence']
        rests.append(temp)
    # annos_other = json.load(open(args.references[0], 'r'))
    #n2vid = json.load(open("/mnt/bd/gxvolume8/Try_asr/densevid_eval/msrvtt_data/msr_test_number2vid.json",'r'))
    #annos,rests = deal(annos,rests,n2vid)
    # annos_fake = json.load(open('/mnt/bd/gxvolume8/Try_asr/densevid_eval/caption_eval/ann.json', 'r'))
    # rests_fake = json.load(open('/mnt/bd/gxvolume8/Try_asr/densevid_eval/caption_eval/ref.json', 'r'))
    all_score = {}
    for cla,ind in index.items():
        rests_tmp = [rests[i] for i in ind]  # if i<len(rests)]
        annos_tmp = {k: v for k, v in annos.items() if int(k) in ind}  # and int(k)<len(rests)}
        eval_cap = EvalCap(annos_tmp, rests_tmp, tokenizer,use_scorers=['CIDEr', 'SPICE','ROUGE_L'])#, use_scorers=['CIDEr', 'SPICE','ROUGE_L'])  # , use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'])

        eval_cap.evaluate()

        scores = {}
        for metric, score in eval_cap.eval.items():
            # print('%s: %.1f' % (metric, score*100))
            scores[metric] = score
        all_score[cla] = scores
    json.dump(all_score, open(args.output, 'w'))


def main():
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str, default='/mnt/bd/gxvolume8/Try_asr/result_msr/layer3_asrsents/region_8_kg0/msr_single_2019FDO/model_best_greedy_pred_test.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default=["/mnt/bd/gxvolume8/Try_asr/densevid_eval/msrvtt_data/msr_test_anet_format_anntidy.json"],
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('-o', '--output', type=str, default='/mnt/bd/gxvolume8/result.json', help='output file with final language metrics.')
    parser.add_argument('-v', '--verbose', default=True, help='Print intermediate steps.')
    parser.add_argument('--time', '--t', action='store_true',help='Count running time.')
    parser.add_argument('--all_scorer', '--a', action='store_true',help='Use all scorer.')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
