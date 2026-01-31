"""
Captioning metrics: BLEU, METEOR, ROUGE; optional CLIPScore.
"""

from typing import Any, Dict, List, Optional, Union


def compute_bleu(
    references: Union[List[str], List[List[str]]],
    hypotheses: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU score(s) (sacrebleu).

    Args:
        references: List of reference captions, or list of list of refs per hypothesis.
        hypotheses: List of predicted captions.

    Returns:
        Dict with bleu score and optional precisions.
    """
    import sacrebleu
    if not references:
        refs = []
    elif isinstance(references[0], str):
        refs = [references]
    else:
        refs = list(references)
    if refs and len(refs) != len(hypotheses):
        if len(refs) == 1:
            refs = [refs[0]] * len(hypotheses)
        else:
            refs = refs[: len(hypotheses)]
    if not refs or not hypotheses:
        return {"bleu": 0.0, "bleu_precisions": []}
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return {"bleu": bleu.score, "bleu_precisions": list(bleu.precisions)}


def compute_meteor(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """
    Compute METEOR score (nltk).

    Args:
        references: List of reference captions (one per hypothesis).
        hypotheses: List of predicted captions.

    Returns:
        Dict with meteor score.
    """
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([r.split()], h.split()) for r, h in zip(references, hypotheses)]
    return {"meteor": sum(scores) / len(scores) if scores else 0.0}


def compute_rouge(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-L (and optionally ROUGE-1/2) via rouge_score.

    Args:
        references: List of reference captions.
        hypotheses: List of predicted captions.

    Returns:
        Dict with rouge1, rouge2, rougeL (f) scores.
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for r, h in zip(references, hypotheses):
        s = scorer.score(r, h)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(r1) or 1
    return {"rouge1": sum(r1) / n, "rouge2": sum(r2) / n, "rougeL": sum(rl) / n}


def compute_metrics(
    references: List[str],
    hypotheses: List[str],
    include_bleu: bool = True,
    include_meteor: bool = True,
    include_rouge: bool = True,
) -> Dict[str, float]:
    """
    Aggregate BLEU, METEOR, ROUGE for caption evaluation.

    Args:
        references: Reference captions (one per hypothesis).
        hypotheses: Predicted captions.
        include_bleu: Compute BLEU.
        include_meteor: Compute METEOR.
        include_rouge: Compute ROUGE.

    Returns:
        Dict of metric names to scores.
    """
    out: Dict[str, float] = {}
    refs_bleu = [references] if references else []
    if include_bleu and refs_bleu and hypotheses:
        out.update(compute_bleu(refs_bleu, hypotheses))
    if include_meteor and references and hypotheses:
        out.update(compute_meteor(references, hypotheses))
    if include_rouge and references and hypotheses:
        out.update(compute_rouge(references, hypotheses))
    return out
