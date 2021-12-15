import torch
import torch.nn.functional as F
from datasets import load_metric

acc, f1 = None, None
#bertscore = load_metric("bertscore")

def multi_acc(y, y_hat):
    """
    y_hat : logits of size  (bs, n_class) or predicted class of size (bs,)
    y : expected class of size (bs,)
    """
    #if y_hat.dim() == 2 :
    #    y_hat = torch.log_softmax(y_hat, dim=1).argmax(dim=1)
    return 100. * (y_hat == y).float().mean().item()
    #return 100. * acc.compute(predictions=y_hat.view(-1), references=y.view(-1))

def get_compute_metrics_lm(task):
    assert task in ["clm", "mlm"]
    if task == "clm" :
        def compute_metrics_lm(y, logits, mask_token_index=None,  prefix=""):
            results = {}
            y = y.cpu()
            y_hat = torch.log_softmax(logits.cpu().detach(), dim=-1).argmax(dim=-1)
            results['%s%s'%(prefix,"acc")] = multi_acc(y=y, y_hat=y_hat)
            #bertscore.compute(predictions=y_hat, references=y, lang="en")
            return results
    else :
        def compute_metrics_lm(y, logits, mask_token_index,  prefix=""):
            results = {}
            y = y.cpu()[:,mask_token_index]
            logits = logits.cpu().detach()[:,mask_token_index,:]
            y_hat = torch.log_softmax(logits, dim=-1).argmax(dim=-1)
            results['%s%s'%(prefix,"acc")] = multi_acc(y=y, y_hat=y_hat)
            #bertscore.compute(predictions=y_hat, references=y, lang="en")
            return results
        
    return compute_metrics_lm

def compute_metrics_bin_clf(y, logits, prefix=""):
    results = {}
    y = y.cpu()
    y_hat = torch.sigmoid(logits.cpu().detach()).round().int()
    #results['%s%s'%(prefix,"clf_acc")] = multi_acc(y=y, y_hat=y_hat)
    results['%s%s'%(prefix,"clf_acc")] = 100. * acc.compute(predictions=y_hat, references=y)["accuracy"]
    results['%s%s'%(prefix,"clf_f1")] = 100. * f1.compute(predictions=y_hat, references=y)["f1"]
    return results

def compute_metrics_multi_clf(y, logits, prefix=""):
    results = {}
    y = y.cpu()
    y_hat = F.softmax(logits.cpu().detach(), dim=-1).max(1)[0]
    #results['%s%s'%(prefix,"clf_acc")] = multi_acc(y=y, y_hat=y_hat)
    results['%s%s'%(prefix,"clf_acc")] = 100. * acc.compute(predictions=y_hat, references=y)["accuracy"]
    results['%s%s'%(prefix,"clf_f1")] = 100. * f1.compute(predictions=y_hat, references=y)["f1"]
    return results

def get_compute_metrics_clf(n_labels): 
    global acc, f1
    assert n_labels > 1
    acc = load_metric("accuracy")
    f1 = load_metric("f1")
    if n_labels == 2 :
        return compute_metrics_bin_clf
    else :
        return compute_metrics_multi_clf

    