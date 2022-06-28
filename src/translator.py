""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from src.rtransformer.model import NonRecurTransformer
from src.rtransformer.masked_transformer import MTransformer
from src.rtransformer.beam_search import BeamSearch
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def copy_for_memory(*inputs):
    return [copy.deepcopy(e) for e in inputs]


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero(as_tuple = False)
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.model_config.max_t_len
        self.max_v_len = self.model_config.max_v_len
        self.num_hidden_layers = self.model_config.num_hidden_layers
        self.region_num = opt.region_num

        if model is None:
            if opt.recurrent:
                if opt.xl:
                    logger.info("Use recurrent model - TransformerXL")
                    # model = TransformerXL(self.model_config).to(self.device)
                else:
                    logger.info("Use recurrent model - Mine")
                    model = RecursiveTransformer(self.model_config).to(self.device)
            else:
                if opt.untied:
                    logger.info("Use untied non-recurrent single sentence model")
                    # model = NonRecurTransformerUntied(self.model_config).to(self.device)
                elif opt.mtrans:
                    logger.info("Use masked transformer -- another non-recurrent single sentence model")
                    model = MTransformer(self.model_config).to(self.device)
                else:
                    logger.info("Use non-recurrent single sentence model")
                    model = NonRecurTransformer(self.model_config).to(self.device)
            # model = RecursiveTransformer(self.model_config).to(self.device)
            model = nn.DataParallel(model, device_ids=[0]).to("cuda")
            model.load_state_dict(checkpoint["model"])
        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

        # self.eval_dataset = eval_dataset

    def translate_batch_greedy(self, input_ids_list, video_features_list, input_masks_list, token_type_ids_list,
                               rt_model):
        def greedy_decoding_step(prev_ms, input_ids, video_features, input_masks, token_type_ids,
                            model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """RTransformer The first few args are the same to the input to the forward_step func
            Note:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1
                # if dec_idx < max_v_len + 5:
                #     logger.info("prev_ms {} {}".format(type(prev_ms[0]), prev_ms[0]))
                copied_prev_ms = copy.deepcopy(prev_ms)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features, input_masks, token_type_ids)
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(
                prev_ms, input_ids, video_features, input_masks, token_type_ids)

            # logger.info("input_ids[:, max_v_len:] {}".format(input_ids[:, max_v_len:]))
            # import sys
            # sys.exit(1)

            return cur_ms, input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked"

        config = rt_model.config
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_seq_list = []
            for idx in range(step_size):
                prev_ms, dec_seq = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list

    def translate_batch_single_sentence_greedy(self, input_ids, input_ids2, video_features, region_features, input_masks, input_masks2, token_type_ids, token_type_ids2, model, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
        max_r_len = 20*self.region_num+2     # 30+5+22
        max_asr_len = 5
        start_symbol = input_ids[:, -22].clone()
        input_ids[:, -22:] = 0.
        input_masks[:, -22:] = 0.
        input_ids2[:, max_r_len+max_asr_len:max_r_len+max_asr_len+22] = 0.
        input_masks2[:, max_r_len+max_asr_len:max_r_len+max_asr_len+22] = 0.
        assert torch.sum(input_masks[:, self.max_v_len+1+max_asr_len:]) == 0, "Initially, all text tokens should be masked"
        config = model.module.config
        max_v_len = config.max_v_len+max_asr_len
        max_t_len = config.max_t_len
        bsz = len(input_ids)
        next_symbols = start_symbol  # (N, )

        for dec_idx in range(max_v_len, max_v_len + max_t_len):
            input_ids[:, dec_idx] = next_symbols
            input_ids2[:, dec_idx-max_v_len+max_r_len+max_asr_len] = next_symbols
            input_masks[:, dec_idx] = 1
            input_masks2[:, dec_idx-max_v_len+max_r_len+max_asr_len] = 1
            _, pred_scores = model.forward(input_ids, input_ids2, video_features, region_features, input_masks, input_masks2, token_type_ids, token_type_ids2, None)
            pred_scores[:, :, unk_idx] = -1e10
            # next_words = pred_scores.max(2)[1][:, dec_idx]
            next_words = pred_scores[:, dec_idx-max_v_len].max(1)[1]  # TODO / NOTE changed
            next_symbols = next_words
        return input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        input_ids_list, input_ids_list2, video_features_list, region_features_list, input_masks_list, input_masks_list2, token_type_ids_list, token_type_ids_list2 = model_inputs
        return self.translate_batch_single_sentence_greedy(
                input_ids_list, input_ids_list2, video_features_list, region_features_list, input_masks_list, input_masks_list2, token_type_ids_list, token_type_ids_list2,
                self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids, n):
        """ replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        input_ids = input_ids.clone()
        input_masks = input_masks.clone()
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == n
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks
