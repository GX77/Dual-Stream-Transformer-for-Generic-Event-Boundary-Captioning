from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer, use_scorers=['Bleu', 'METEOR', 'ROUGE_L', 'CIDEr']):#,'SPICE']):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': c} for c in self.annos[imgId]]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        use_scorers = self.use_scorers
        scorers = []
        if 'Bleu' in use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in use_scorers:
            scorers.append((Spice(), "SPICE"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.1f" % (m, sc*100))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.1f" % (method, score*100))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
