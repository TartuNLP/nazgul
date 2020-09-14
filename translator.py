"""Sockeye model loading and inference"""

import os
import sys
import sockeye
import mxnet as mx
import sentencepiece as spm
import json
import time

from typing import List

from truecaser import applytc
from log import log

from collections import namedtuple
from sockeye.translate import inference


class Translator:
    models = None
    spm_model = spm.SentencePieceProcessor()
    tc_model = None

    def __init__(self, sockeye_model_folder: str, spm_model_path: str, tc_model_path: str, use_cpu: bool):
        self.mx_ctx = mx.cpu() if use_cpu else mx.gpu()
        self.use_truecaser = True if tc_model_path is not None else False
        self.constrains = None  # Legacy thing
        self.load_models(sockeye_model_folder, spm_model_path, tc_model_path)

    def _preprocess(self, sent: str, lang_factor: str):
        # TODO: add constraints support with indexes (used in previous versions of this)
        # TODO: add style/domain factors support
        if self.use_truecaser:
            truecased_sentence = applytc.processLine(self.tc_model, sent)
            pieces = self.spm_model.EncodeAsPieces(truecased_sentence)
        else:
            pieces = self.spm_model.EncodeAsPieces(sent)
        spm_sent = " ".join(pieces)
        n_words = len(pieces)

        preproc_json = {"text": spm_sent, "factors": [" ".join([lang_factor] * n_words)]}
        log(f"Preprocessed: {sent} into {spm_sent}.")
        return json.dumps(preproc_json)

    def _postprocess(self, sent: str):
        postproc_sent = self.spm_model.DecodePieces(sent.split()).capitalize()
        log(f"Postprocessed: {sent} into {postproc_sent}.")
        return postproc_sent

    def _forward(self, sents: List[str]):
        translation_inputs = [
            inference.make_input_from_json_string(sentence_id=i, json_string=sent, translator=self.models)
            for (i, sent) in enumerate(sents)]
        outputs = self.models.translate(translation_inputs)
        return [(output.translation, output.score) for output in outputs]

    def load_models(self, sockeye_model_folder: str, spm_model_path: str, tc_model_path: str):
        """Load translation and segmentation models."""

        self.models = load_sockeye_v1_translator_models([sockeye_model_folder, ], self.mx_ctx)

        self.spm_model.Load(spm_model_path)

        if self.use_truecaser:
            self.tc_model = applytc.loadModel(tc_model_path) if os.path.exists(tc_model_path) else None

    def translate(self, sents: List[str], out_langs: List[str]):
        clean_inputs = [self._preprocess(sent, lang) for (sent, lang) in zip(sents, out_langs)]

        scored_translations = self._forward(clean_inputs)

        translations, scores = zip(*scored_translations)

        postproc_translations = [self._postprocess(sent) for sent in translations]
        return postproc_translations, scores, clean_inputs, translations

    def process_config(self):
        # TODO: Do we need this at all? (Initial thought: This is for domains parsing)
        pass


def load_sockeye_v1_translator_models(model_folders, ctx=mx.gpu()):
    log(f"Loading Sockeye models. MXNET context: {ctx}")
    models, source_vocabs, target_vocab = inference.load_models(
        context=ctx,
        max_input_len=None,
        beam_size=3,
        batch_size=16,
        model_folders=model_folders,
        checkpoints=None,
        softmax_temperature=None,
        max_output_length_num_stds=2,
        decoder_return_logit_inputs=False,
        cache_output_layer_w_b=False)

    return inference.Translator(context=ctx,
                                ensemble_mode="linear",
                                bucket_source_width=10,
                                length_penalty=inference.LengthPenalty(1.0, 0.0),
                                beam_prune=0,
                                beam_search_stop='all',
                                models=models,
                                source_vocabs=source_vocabs,
                                target_vocab=target_vocab,
                                restrict_lexicon=None,
                                store_beam=False,
                                strip_unknown_words=False)




