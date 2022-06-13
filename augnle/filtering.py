from nlgeval import NLGEval

class MatchingScorer(object):
    def __init__(self, metric="GreedyMatchingScore"):
        matching_metrics = [
            'Bleu_1',
            'EmbeddingAverageCosineSimilarity',
            'VectorExtremaCosineSimilarity',
            'GreedyMatchingScore', 
            'METEOR', 'ROUGE_L', 'CIDEr', 'SkipThoughtCS']

        if "Bleu_" == metric[:-1]:
            except_metrics = matching_metrics[1:]
        else:
            idx = matching_metrics.index(metric)
            except_metrics = matching_metrics[:idx] + matching_metrics[idx+1:]
        self.metric = metric
        self.evaluator = NLGEval(metrics_to_omit=except_metrics)
    
    def forward(self, hyp, ref):
        if type(ref) is str:
            ref = [ref]
        scores = self.evaluator.compute_individual_metrics(hyp=hyp, ref=ref)
        return scores[self.metric]
    

if __name__=="__main__":
    scorer = MatchingScorer(metric="EmbeddingAverageCosineSimilarity")
    print(scorer.forward("banana", "only banana"))