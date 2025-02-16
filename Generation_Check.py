from argparse import Namespace, ArgumentParser
from pickle import load as p_load
from typing import List, Tuple
from automatic_metrics import AutomaticMetrics
from numpy import mean as np_mean

import re
import pandas as pd
import datasets

FAILED_TOKEN = "<UNK>"
def parse_args() -> Namespace :
    parser = ArgumentParser()
    parser.add_argument("type", default="all",
                        required=True,
                        type=str, help="qag, qg, dg_1, dg_all",
                        choices=["qag", "qg", "dg_1", "dg_all", "all"])
    parser.add_argument("path",
                        type="str",
                        help="pickle path for reading result",
                        default="./generated/dg_1_strict/dg_1_multiple_correct.pickle", )
    parser.add_argument("dataset_path",
                        type="str",
                        help="Dataset path",
                        default="VosLannack/race_id_uncombined")
    parser.add_argument("question_path", type=str,
                        default="VosLannack/squad_id_512")
    parser.add_argument("tyqdqa_path", type=str,
                        default="VosLannack/tydqa_id")
    parser.add_argument("distractor_1_path", type=str,
                        default="VosLannack/race_id_uncombined")
    parser.add_argument("distractor_all_path", type=str,
                        default="VosLannack/race_id")
    return parser.parse_args()
def remove_brackets(text: str) -> str:
    pattern = "<[^>]+>"
    return re.sub(pattern, "", text)

def _clean_qag(text: str) -> str:
    try:
        answer, question = text.split("<question>")
        answer = remove_brackets(answer)
        question = remove_brackets(question)
        return answer.strip(), question.strip()
    except:
        return FAILED_TOKEN, FAILED_TOKEN

def _clean_qg(text: str) -> str :   
    try: 
        return remove_brackets(text).strip()
    except:
        return FAILED_TOKEN

def _clean_dg_all(text: str) -> List[str]:
    try:
        list_of_distractor = []
        list_of_dg = text.split("<sep>")
        for distractor in list_of_dg:
            ds  = distractor.split("</s>")
            if ds != []:
                for d in ds:
                    if d != " " and d != "":
                        list_of_distractor.append(remove_brackets(d).strip())
            else:
                if distractor != " " and distractor != "":
                    list_of_distractor.append(remove_brackets(distractor).strip())
        return list_of_distractor
    except:
        return [FAILED_TOKEN]

def _clean_dg_1(text: str) -> List[str]:
    try:
        list_of_distractor = []
        list_of_dg = text.split("</s>")
        for distractor in list_of_dg:
            list_of_distractor.append(remove_brackets(distractor).strip())
        return list_of_distractor

    except:
        return FAILED_TOKEN


def qag_cleaner(qag_pairs: List[str]) -> List[str]:
    qag_answer: List[str] = []
    qag_question: List[str] = []
    for text in qag_pairs:
        ans, question = _clean_qag(text)
        qag_answer.append(ans)
        qag_question.append(question)

    return qag_answer, qag_question

def qg_cleaner(qg_list: List[str]) -> List[str]:
    qgs: List[str] = []
    for text in qg_list:
        qgs.append(_clean_qg(text))
    return qgs

def dg_all_cleaner(dg_all: List[str]) -> List[List[str]]:
    dgs: List[List[str]] = []
    for distractors in dg_all:
        dgs.append(_clean_dg_all(distractors))
    return dgs

def dg_1_cleaner(dg_1: List[str]) -> List[List[str]]:
    dgs : List[str] = []
    for distractor in dg_1:
        dgs.append(_clean_dg_1(distractor))
    return dgs

def make_dg_result(distractors: List[List[str]], n_result: int = 3) -> pd.DataFrame:
    return pd.DataFrame(distractors, columns=[f"incorrect_{i + 1}_gen" for i in range(n_result)])

def make_qag_result(questions: List[str], answers: List[str] ) -> pd.DataFrame:
    q_df = pd.DataFrame(questions, columns=["question_gen"])
    a_df = pd.DataFrame(answers, columns=["answer_gen"])
    return pd.concat([q_df, a_df],axis=1)

def make_qg_result(questions: List[str]) -> pd.DataFrame:
    return pd.DataFrame(questions, columns=["question_gen"])

def get_automatic_result(df: pd.DataFrame, target_column: str, pred_column: str ):
    
    bl_1, bl_2, bl_3, bl_4, rouge_l = [], [] ,[] , [], []
    for indx, item in df.iterrows():
        target = item[target_column]
        pred = item[pred_column]
        b1, b2, b3 , b4 = AutomaticMetrics()._calculate_bleu(target, pred)
        _, _, _, r = AutomaticMetrics()._calculate_rouge(target, pred)

        bl_1.append(b1)
        bl_2.append(b2)
        bl_3.append(b3)
        bl_4.append(b4)
        rouge_l.append(r)
    return bl_1, bl_2, bl_3, bl_4, rouge_l  

def get_bert_scores(df: pd.DataFrame, target_column: str, pred_column: str, model: str):
    references = list(df[target_column].values)
    candidates = list(df[pred_column].values)
    P, R, F1 = AutomaticMetrics().calculate_bert_score(references, candidates, model=model)
    return P, R, F1

def get_bert_scores(df: pd.DataFrame, target_column: str, pred_column: str, model: str):
    references = list(df[target_column].values)
    candidates = list(df[pred_column].values)
    cos = AutomaticMetrics().calculate_sbert_score(references, candidates, model)
    return cos

def print_metric_result(b1, b2, b3, b4, r, sbert ):
    print("Bleu 1 : ", np_mean(b1) * 100)
    print("Bleu 2 : ", np_mean(b2) * 100 )
    print("Bleu 3 : ", np_mean(b3) * 100)
    print("Bleu 4 : ", np_mean(b4) * 100)
    print("Rouge-L :", np_mean(r) * 100)
    print("SBert-Score; : ", np_mean(sbert) )

def load_generation_result(path: str):
    with open(path, "rb") as f:
        gen_result = p_load(f)
    return gen_result

def fetch_dataset(path: str, key_name: str) -> pd.DataFrame:
    dataset = datasets.load_dataset(path)
    return pd.DataFrame(dataset[key_name])


