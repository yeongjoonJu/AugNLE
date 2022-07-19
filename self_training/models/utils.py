import re
import json
import  os, sys
import re
import torch
from torch.nn import CrossEntropyLoss
import json
import torch.nn.functional as F

from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')



contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def prep_ans(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def proc_ans(ans):

    ans_prob_dict = {}

    for ans_ in ans:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    confident_answer = max(ans_prob_dict, key=ans_prob_dict.get)
    return confident_answer

def proc_ques(ques):
    words = re.sub(r"([.,'!?\"()*#:;])",'',ques.lower()).replace('-', ' ').replace('/', ' ')
    return words

import torch
import torch.nn.functional as F

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def pseudo_labeling(mode, output_data, pseudo_labels_pth, iter):
    if os.path.exists(pseudo_labels_pth):
        print(pseudo_labels_pth)
        annotations = json.load(open(pseudo_labels_pth, "r"))
    else:
        annotations = {}
    # Only for VQA-X 
    if mode == "teacher":
        for batch in output_data:
            for smp in batch:
                img_name = str(smp["qid"])
                try:
                    question, answer = smp["input"].split("the answer is")
                except:
                    print(smp["input"])
                    continue
                answer = answer.split("<|endoftext|>")[0]
                
                if img_name not in annotations.keys():
                    annotations[img_name] = {}
                    annotations[img_name]["question"] = question
                    annotations[img_name]["answers"] = [{'answer': answer, "answer_confidence": f"iter{iter}", "answer_id":1}]
                    annotations[img_name]["image_id"] = img_name
                    annotations[img_name]["explanation"] = [smp["GT"]]
                else:
                    annotations[img_name]["question"] = question
                    annotations[img_name]["answers"].append({'answer': answer, "answer_confidence": "iter1", "answer_id":len(annotations[img_name]["answers"])+1})
                    annotations[img_name]["explanation"].append(smp["GT"])
    elif mode == "student":
        for batch in output_data:
            for smp in batch:
                question, answer = smp["input"].split("<|endoftext|>")
                img_name = str(smp["qid"])
                try:
                    answer, explanation = answer.split("because")
                except:
                    print(smp["input"])
                answer = answer.replace("the answer is ", "")
                
                if img_name not in annotations.keys():
                    annotations[img_name] = {}
                    annotations[img_name]["question"] = question
                    annotations[img_name]["answers"] = [{'answer': answer, "answer_confidence": f"iter{iter}", "answer_id":1}]
                    annotations[img_name]["image_id"] = img_name
                    annotations[img_name]["explanation"] = [explanation]
                else:
                    annotations[img_name]["question"] = question
                    annotations[img_name]["answers"].append({'answer': answer, "answer_confidence": "iter1", "answer_id":len(annotations[img_name]["answers"])+1})
                    annotations[img_name]["explanation"].append(explanation)
    else:
        raise NotImplementedError("Only for Student and Teacher mode")
    # new_path = os.path.join(data_dir, f"pseudo_labeling_{iter}")
    with open(pseudo_labels_pth, "w") as fout:
            json.dump(annotations, fout, indent=2)
    return pseudo_labels_pth


def split_dataset(data_lst,num):
    return [data_lst[i: i+num] for i in range(0, len(data_lst), num)]


def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions, test_anno_path):
    annFileExp = 'cococaption/annotations/vqaX_test_annot_exp.json'
    annFileFull = 'cococaption/annotations/vqaX_test_annot_full.json'
    all_file = json.load(open(test_anno_path, 'r'))
    
    gt_answers = {}
    for key,value in all_file.items():
        gt_answers[int(key)] = proc_ans(value['answers'])
        
    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    correct_keys = []
    for key,value in pred_answers.items():
        gt_answer = gt_answers[key]
        # to measure accuracy for VQA, please change "==" to "in" (if value in gt_answer:)
        if value == gt_answer:
            correct_keys.append(key)

    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]
    
    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)

    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)