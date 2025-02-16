from nltk.translate.bleu_score import sentence_bleu
from numpy import mean as nmean
from rouge_score import rouge_scorer
from typing import List, Tuple
from bert_score import score
from nltk.tokenize import word_tokenize
class AutomaticMetrics():
    
    def calculate_bleu(this, references: List[str], generated_candidates: List[str]):
        b_1s = []
        b_2s = []
        b_3s = []
        b_4s = []

        for reference, candidate in zip(references, generated_candidates):
            b_1, b_2, b_3, b_4 = this._calculate_bleu(reference, candidate)

            b_1s.append(b_1)
            b_2s.append(b_2)
            b_3s.append(b_3)
            b_4s.append(b_4)
        
        return nmean(b_1s), nmean(b_2s), nmean(b_3s), nmean(b_4s)
    
    def calculate_bert_score(this, references: List[str], candidates: List[str], model: str):
        P, R, F1 = score(candidates, references, lang="id", model_type=model)
        return P, R, F1

    

    def calculate_rouge(this, references: List[str], candidates: List[str]):
        r_1s = []
        r_2s = []
        r_3s = []
        r_Ls = []

        for reference, candidate in zip(references, candidates):
            r_1, r_2, r_3, r_L = this._calculate_rouge(reference, candidate)

            r_1s.append(r_1)
            r_2s.append(r_2)
            r_3s.append(r_3)
            r_Ls.append(r_L)
        
        return nmean(r_1s), nmean(r_2s), nmean(r_3s), nmean(r_Ls)


    def _calculate_bleu(this, reference: str, candidate: str) -> Tuple[float, float, float, float]:

        reference_token = word_tokenize(reference.lower())
        candidate_token = word_tokenize(candidate.lower())

        b_1_score = sentence_bleu([reference_token], candidate_token, weights=(1, 0, 0, 0))
        b_2_score = sentence_bleu([reference_token], candidate_token, weights=(0.50, 0.50, 0, 0))
        b_3_score = sentence_bleu([reference_token], candidate_token, weights=(0.33, 0.33, 0.33, 0))
        b_4_score = sentence_bleu([reference_token], candidate_token, weights=(0.25, 0.25, 0.25, 0.25))

        return b_1_score, b_2_score, b_3_score, b_4_score
    
        
    def _calculate_rouge(this, reference: str, candidate: str):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3','rougeL'], use_stemmer=False)
        rouge_scores = scorer.score(reference, candidate)
        return rouge_scores["rouge1"].fmeasure, rouge_scores["rouge2"].fmeasure, rouge_scores["rouge3"].fmeasure, rouge_scores["rougeL"].fmeasure


"""
if __name__ == "__main__":

    # Example reference and candidate sentences
    reference = ['this is a test']
    candidate = ['this a test']

    # Initialize ROUGE scorer
    
    m1, m2, m3, m4 = AutomaticMetrics().calculate_rouge(reference, candidate)
    print(m1, m2, m3, m4)
"""