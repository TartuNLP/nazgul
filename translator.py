"""Sockeye model loading and inference"""

import os
import sys
import sockeye
import mxnet as mx
import sentencepiece as spm
import json

from truecaser import applytc
from log import log

from collections import namedtuple
from sockeye.translate import inference


def _preprocess(sentence, index, lang_factor, style_factor, models, constraints):
    if models.truecaser:
        truecased_sentence = applytc.processLine(models.truecaser, sentence)
        pieces = models.segmenter.EncodeAsPieces(truecased_sentence)
    else:
        pieces = models.segmenter.EncodeAsPieces(sentence)
    segmented_sentence = ' '.join(pieces)

    rawlen = len(pieces)
    # TODO: Legacy code - factors (should come from config or something
    # prejsson = {'text': segmented_sentence, 'factors': [" ".join([lang_factor] * rawlen), " ".join([style_factor] * rawlen), " ".join(['f0'] * rawlen), " ".join(['g0'] * rawlen)]}

    prejsson = {'text': segmented_sentence, 'factors': [" ".join([lang_factor] * rawlen)]}

    try:
        if constraints and constraints[index]:
            prejsson['avoid'] = constraints[index]
    except IndexError as e:
        sys.stderr.write(str(constraints) + ", " + str(index))
        raise e

    jsson = json.dumps(prejsson)
    log("PREPROC received '" + sentence + "', turned it into '" + segmented_sentence + "'")
    return jsson


def _doMany(many, func, args):
    return [func(one, idx, *args) for idx, one in enumerate(many)]


def _postprocess(sentence, idx, models):
    de_segmented_sentence = models.segmenter.DecodePieces(sentence.split())
    try:
        de_truecased_sentence = de_segmented_sentence[0].upper() + de_segmented_sentence[1:]
    except:
        de_truecased_sentence = de_segmented_sentence

    log("POSTPROC received '" + sentence + "', turned it into '" + de_truecased_sentence + "'")

    return de_truecased_sentence


def _forward(sentences, models):
    trans_inputs = [
        inference.make_input_from_json_string(sentence_id=i, json_string=sentence, translator=models.translator) for
        i, sentence in enumerate(sentences)]
    outputs = models.translator.translate(trans_inputs)
    return [(output.translation, output.score) for output in outputs]


def _loadTranslator(model_folders, ctx=mx.gpu()):
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


def load_models(sockeye_model_path, spm_model_path, tc_model_path=None, use_cpu=False):
    """Load translation, truecasing and segmentation models and return them as a named tuple"""
    mx_ctx = mx.cpu() if use_cpu else mx.gpu()

    sockeye_model = _loadTranslator([sockeye_model_path, ], mx_ctx)

    segmenter_model = spm.SentencePieceProcessor()
    segmenter_model.Load(spm_model_path)

    tc_model = applytc.loadModel(tc_model_path) if os.path.exists(tc_model_path) else None

    Models = namedtuple("Models", ["translator", "segmenter", "truecaser"])

    return Models(sockeye_model, segmenter_model, tc_model)


def translate(models, sentences, outputLanguage, outputStyle, constraints):
    """Take list of sentences, output language and style as well as a list of constraints,
    and feed them through a set of loaded NMT models.
	Return list of translations, list of scores, list of preprocessed input sentences and
	list of raw translations prior to postprocessing."""

    cleaninputs = _doMany(sentences, _preprocess, (outputLanguage, outputStyle, models, constraints))

    scoredTranslations = _forward(cleaninputs, models)
    translations, scores = zip(*scoredTranslations)

    postprocessed_translations = _doMany(translations, _postprocess, (models,))

    return postprocessed_translations, scores, cleaninputs, translations
