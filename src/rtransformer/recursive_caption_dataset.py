import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO)


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    BEF_TOKEN = "[BEF]"
    AFT_TOKEN = "[AFT]"
    SUB_TOKEN = "[SUB]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    BEF = 7
    AFT = 8
    SUB = 9
    IGNORE = -1  # used to calculate loss

    """
    recurrent: if True, return recurrent data
    """

    def __init__(self, dset_name, data_dir, video_feature_dir, region_feature_dir, region_num, word2idx_path,
                 max_t_len, max_v_len, max_n_sen, mode="train", recurrent=True, untied=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.video_feature_dir = video_feature_dir
        self.region_feature_dir = region_feature_dir
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen
        self.max_asr_len = 5

        self.n = region_num
        self.max_r_len = 20 * self.n + 2
        self.max_len = self.max_r_len + self.max_asr_len + 22

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None  # number of sentence for each video, set in self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logging.info("Mode {}".format(mode))
        self.mode = mode
        if self.dset_name == "kin":
            if mode == "train":
                data_path = os.path.join(self.data_dir, "kin_train_anet_format_independent.json")
            elif mode == "val":
                data_path = os.path.join(self.data_dir, "kin_val_anet_format_independent.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        else:
            raise ValueError
        self._load_data(data_path)

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):
            video_name = e["video_id"]  # e["name"][2:] if self.dset_name == "anet" else e["name"]
            cur_path = os.path.join(self.video_feature_dir+"2d_clip/", "{}.npy".format(video_name))
            for p in [cur_path]:
                if not os.path.exists(p):
                    self.missing_video_names.append(e["video_id"])
        print("Missing {} features (clips/sentences) from {} videos".format(len(self.missing_video_names),
                                                                            len(set(self.missing_video_names))))
        # print("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["video_id"] not in self.missing_video_names]

    def _load_data(self, data_path):
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in tqdm(raw_data.items()):
            line["name"] = k
            line["sentences"] = [line["sentence"]]
            data.append(line)

        if self.recurrent:  # recurrent
            self.data = data
        else:  # non-recurrent single sentence
            singel_sentence_data = []
            for d in data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                singel_sentence_data.extend([
                    {
                        "name": d['name'],
                        "sentence": d["sentences"][idx],
                        "video_id": d['video_name'].split(".")[0],
                        "idx": idx,
                        "time":d['video_time'],
                        "type":d['type']
                    } for idx in range(num_sen)])
            self.data = singel_sentence_data

        logging.info("Loading complete! {} examples".format(len(self)))

    def convert_example_to_features(self, example):
        name = example["video_id"]
        start_f, end_f = 4*int(example["time"][0]), 4*int(example["time"][1]+1)  # [start_f:end_f]
        if start_f >= 4:
            start_f = start_f - 4
        video_feature = np.load(self.video_feature_dir+"/rt_kin_frame/myself_frames_224/{}.npy".format(name), allow_pickle=True)[start_f:end_f]
        index = np.linspace(0, video_feature.shape[0]-1, self.max_v_len, endpoint=True).astype(np.int).tolist()
        video_feature = video_feature[index]

        region_path = os.path.join(self.region_feature_dir, "{}.npy".format(name))
        temp = np.load(region_path, allow_pickle=True).tolist()
        region_feature = temp['x'][int(start_f/2):int(end_f/2)][:, :self.n]
        tags = temp['score'][int(start_f/2):int(end_f/2)][:, :self.n]
        rf_300d = None
        asr = example['type']
        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(example["name"],
                                                                   example["timestamps"][clip_idx],
                                                                   example["sentences"][clip_idx],
                                                                   video_feature,
                                                                   region_feature,
                                                                   tags)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature
            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["sentence"],
                                                 video_feature,
                                                 region_feature,
                                                 rf_300d,
                                                 tags,
                                                 asr)
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, sentence, video_feature, region_fature, rf_300d, tags, asr):
        max_r_len = self.max_r_len
        frm2sec = 0.5  # self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]
        # if str(type(sentence)) == "<class 'list'>":
        #    sentence = self.choose(sentence,ratio)
        # video + text tokens
        feat, region, rf, tag, video_tokens, region_token, video_mask, region_mask, m = self._load_indexed_video_feature(
            video_feature, region_fature, rf_300d, tags, frm2sec)

        text_tokens, text_mask = self._tokenize_pad_sentence(sentence, self.max_t_len)

        asr = asr.lower().split(" ")[:3]
        asr_tokens, asr_mask = [self.BOS_TOKEN] + asr + [self.EOS_TOKEN] + [self.PAD_TOKEN]*(self.max_asr_len-len(asr)-2), [1] * (len(asr)+2) + [0] * (self.max_asr_len-len(asr)-2)

        input_tokens = video_tokens + asr_tokens + text_tokens
        input_tokens2 = region_token + asr_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        input_ids2 = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens2]

        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + [self.IGNORE]

        input_mask = video_mask + asr_mask + text_mask
        input_mask2 = region_mask + asr_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_asr_len + [1] * self.max_t_len
        token_type_ids2 = [0] * max_r_len + [1] * self.max_asr_len + [1] * self.max_t_len

        data = dict(
            name=name,
            input_tokens=input_tokens,
            input_tokens2=input_tokens2,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64),
            input_ids2=np.array(input_ids2).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            input_mask2=np.array(input_mask2).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            token_type_ids2=np.array(token_type_ids2).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            region_feature=region.astype(np.float32),
            tag=np.array(tag).astype(np.int64),
        )
        meta = dict(
            # meta
            name=name,
            sentence=sentence
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature(self, raw_feat, region_feat, rf_300d, tags, frm2sec):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """

        if len(raw_feat.shape) == 1:
            raw_feat = raw_feat.reshape(1, -1)
        n = self.n
        max_v_l = self.max_v_len
        max_r_l = self.max_r_len

        feat = np.zeros((self.max_v_len + self.max_asr_len + self.max_t_len, 768))
        region = np.zeros((max_r_l + self.max_asr_len + self.max_t_len, 2048))
        rf = np.zeros((max_r_l + self.max_asr_len + self.max_t_len, 300))
        tag = np.zeros((max_r_l + self.max_asr_len + self.max_t_len))

        valid_l = int(raw_feat.shape[0]/2)
        if valid_l>28:
            downsamlp_indices = np.linspace(0, valid_l - 1, 28, endpoint=True).astype(np.int).tolist()
            raw_feat = raw_feat[downsamlp_indices]
            valid_l = 28
        valid_d = region_feat.shape[0]
        if valid_d>20:
            # print(valid_d)
            downsamlp_indices = np.linspace(0, valid_d-1, 20, endpoint=True).astype(np.int).tolist()
            region_feat = region_feat[downsamlp_indices]
            tags = tags[downsamlp_indices]
            valid_d = 20

        # feat[1:valid_l + 1] = raw_feat  # ,raw_feat.shape[2],raw_feat.shape[3]
        region[1:valid_d * n + 1] = region_feat.reshape(-1, 2048)
        tag[1:valid_d * n + 1] = tags.reshape(-1)
        video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (
                    max_v_l - valid_l - 2)
        region_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_d * n + [self.SEP_TOKEN] + [self.PAD_TOKEN] * (
                    max_r_l - valid_d * n - 2)
        mask_video = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l - 2)
        mask_region = [1] * (valid_d * n + 2) + [0] * (max_r_l - valid_d * n - 2)
        m = valid_d * n

        return feat, region, rf, tag, video_tokens, region_tokens, mask_video, mask_region, m


    def _tokenize_pad_sentence(self, sentence, max_t_len):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        mark = sentence[:5]
        sentence = sentence[6:]
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [mark] + sentence_tokens + [self.EOS_TOKEN]
        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
