from nltk.translate.bleu_score import sentence_bleu

ref = ["je", "suis", "fatigué"]
hyp = "je suis fatigue".split()

score = sentence_bleu([ref], hyp)
print(f"BLEU Score: {score:.4f}")