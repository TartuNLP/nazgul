#!/usr/bin/python3

import sys
import sockeye
import mosestokenizer
import html
import json
import socket

import mxnet as mx
import sentencepiece as spm

import ipywidgets as widgets

from truecaser import applytc
from time import time
from datetime import datetime
from sockeye.translate import inference
from nltk.tokenize import sent_tokenize
from traceback import format_exc

sntId = 0

supportedStyles = { 'fml', 'inf' }
supportedOutLangs = { 'et', 'lv', 'en' }
defaultStyle = 'inf'
defaultOutLang = 'en'

SOCKEYE_MODEL_FOLDER_ENETLV = ['en-et-lv-model']
TRUECASE_MODEL_ENETLV = 'preprocessing-models/joint-truecase-enetlv.tc'
SENTENCEPIECE_MODEL_ENETLV = 'preprocessing-models/sp.model'

my_tokenizer = mosestokenizer.MosesTokenizer('en')
my_detokenizer = mosestokenizer.MosesDetokenizer('en')
my_truecaser_enetlv = applytc.loadModel(TRUECASE_MODEL_ENETLV)
my_segmenter_enetlv = spm.SentencePieceProcessor()
my_segmenter_enetlv.Load(SENTENCEPIECE_MODEL_ENETLV)

def log(msg, skip = False):
    global sntId
    if skip:
        msg = "[DEBUG {0}] {1}\n".format(datetime.now(), msg)
    else:
        msg = "[DEBUG {0} {1}] {2}\n".format(sntId, datetime.now(), msg)
    for channel in (sys.stderr, sys.stdout):
        channel.write(msg)

def get_translator(model_folders):
    #ctx = mx.cpu()
    ctx = mx.gpu()
    models, source_vocabs, target_vocab = inference.load_models(
        context=ctx,
        max_input_len=None,
        beam_size=5,
        batch_size=10,
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


my_translator_enetlv = get_translator(SOCKEYE_MODEL_FOLDER_ENETLV)

def preprocess(sentence, lang_factor, style_factor,
               tokenizer, truecaser, segmenter):
    tokenized_sentence = html.unescape(' '.join(tokenizer(sentence.replace("\n", " "))))
    truecased_sentence = applytc.processLine(truecaser,
                                             tokenized_sentence).replace("@-@", "-")
    segmented_sentence = ' '.join([x
                                   for x in segmenter.EncodeAsPieces(truecased_sentence)])
    factored_sentence = ' '.join([x + '|' + lang_factor + '|' + style_factor
                                  for x in segmented_sentence.split()])
    log("PREPROC received '" + sentence + "', turned it into '" + segmented_sentence + "'")
    return factored_sentence

def doMany(many, func, args):
	return [func(one, *args) for one in many]

def postprocess(sentence, segmenter, detokenizer):
    de_segmented_sentence = segmenter.DecodePieces(sentence.split())
    de_truecased_sentence = de_segmented_sentence[0].upper() + de_segmented_sentence[1:]
    de_tokenized_sentence = detokenizer(de_truecased_sentence.split())
    
    log("POSTPROC received '" + sentence + "', turned it into '" + de_tokenized_sentence + "'")
    
    return de_tokenized_sentence

def forward(sentences, t):
    trans_inputs = [inference.make_input_from_factored_string(sentence_id=i, factored_string=sentence, translator=t) for i, sentence in enumerate(sentences)]
    outputs = t.translate(trans_inputs)
    return [(output.translation, output.score) for output in outputs]

def translate(sentences, lang_factor, style_factor,
              tokenizer = my_tokenizer, detokenizer = my_detokenizer, truecaser = my_truecaser_enetlv, segmenter = my_segmenter_enetlv,
              translator = my_translator_enetlv):
    
    #livesubs = "|" in text
    #sentences = text.split("|") if livesubs else sent_tokenize(text)
    cleaninputs = doMany(sentences, preprocess, (lang_factor, style_factor, tokenizer, truecaser, segmenter))
    
    scoredTranslations = forward(cleaninputs, translator)
    translations, scores = zip(*scoredTranslations)
    
    postprocessed_translations = doMany(translations, postprocess, (segmenter, detokenizer))
    #findelim = "|" if livesubs else " "
    #postprocessed_translation = findelim.join(postprocessed_translations)
    
    return postprocessed_translations, scores

def parseInput(rawText):
	global supportedStyles, defaultStyle, supportedOutLangs, defaultOutLang
	
	try:
		rawStyle, rawOutLang, fullText = rawText.split(",", maxsplit=2)
		
		livesubs = "|" in fullText
		
		sentences = fullText.split("|") if livesubs else sent_tokenize(fullText)
		delim = "|" if livesubs else " "
		
	except AttributeError:
		sentences = rawText['sentences']
		rawStyle = rawText['outStyle']
		rawOutLang = rawText['outLang']
		delim = False
	
	if rawStyle not in supportedStyles:
		#raise ValueError("style bad: " + rawStyle)
		rawStyle = defaultStyle
	
	if rawOutLang not in supportedOutLangs:
		#raise ValueError("out lang bad: " + rawOutLang)
		rawOutLang = defaultOutLang
	
	outputStyle = 'to-osubs' if rawStyle == 'inf' else 'to-eparl'
	outputLang = 'to-' + rawOutLang
	
	#print("LOGGG", sentences, outputLang, outputStyle, delim)
	return sentences, outputLang, outputStyle, delim
	
		
	#sentence = sentence.replace("\n", " ")
	#if froml == 'fml':
	#	froml = 'eparl'
	#else:
	#	froml = 'osubs'
	#
	#if not sntl.strip()[-1] in ".?!":
	#	sntl = sntl.strip() + "."

def send(rawtext,
         tokenizer = my_tokenizer, detokenizer = my_detokenizer, truecaser = my_truecaser_enetlv, segmenter = my_segmenter_enetlv,
         translator = my_translator_enetlv):
    #lang_dict = {'EN': 'to-en',
    #             'ET': 'to-et',
    #             'LV': 'to-lv',
    #             'DE': 'to-de',
    #             'FR': 'to-fr'}
    #style_dict = {'Informal': 'to-osubs',
    #              'Official': 'to-eparl',
    #              'Legal': 'to-jrcac',
    #              'Medical': 'to-emea'}
    #isLive = 'live' in sentenceStruc
    
    inputSntList, outputLang, outputStyle, delim = parseInput(rawtext)
    
    start = time()
    preres, scores = translate(inputSntList, outputLang, outputStyle,
                     tokenizer, detokenizer, truecaser, segmenter,
                     translator)
    sntcount = len(inputSntList)
    res = delim.join(preres) if delim else preres
    endtime = time() - start
    log("input: '{0}', output: '{1}', snt. count {2}, scores {3}, comp. time: {4}".format(rawtext, res, sntcount, "/".join(["{0:.2}".format(s) for s in scores]), endtime))

    return res, scores

def startServ():
	global sntId
	log("starting server", skip=True)
	
	s = socket.socket()
	host = '172.17.37.223' #socket.gethostname()  # Get local machine name
	port = 5678  # Reserve a port for your service.
	s.bind((host, port))
	
	while True:
		try:
			s.listen(5)
			c, a = s.accept()
			xx = c.recv(4096)
			log("check " + str(xx))
			c.send(b"okay")
			gotthis = c.recv(4096).decode('utf-8')
			info = json.loads(gotthis)
			#log(gotthis)
			response, scores = send(info['src'])
			msg = json.dumps({'raw_trans': ['-'],
				'raw_input': ['-'],
				'final_trans': response})
			c.send(bytes(msg, 'utf-8'))
			sntId += 1
		except:
			log("ERROR: " + format_exc())
			msg = json.dumps({'raw_trans': ['-'],
				'raw_input': ['-'],
				'final_trans': "(Some error, sorry)"})
			c.send(bytes(msg, 'utf-8'))
			c.close()

def prtr(lines, outputLang, outputDomain):
	trs, scs = send({ 'sentences': lines, 'outLang': outputLang, 'outStyle': outputDomain })
	for tr, sc in zip(trs, scs):
		print(str(sc) + "\t" + tr)

def translateStdinInBatches():
	outputLang = sys.argv[1]
	try:
		outputDomain = sys.argv[2]
	except IndexError:
		outputDomain = 'inf'
	lines = []
	
	batchMaxSize = 48
	for line in sys.stdin:
		lines.append(line.strip())
		if len(lines) >= batchMaxSize:
			prtr(lines, outputLang, outputDomain)
			lines = []
	if lines:
		prtr(lines, outputLang, outputDomain)

if __name__ == "__main__":
	#TODO implement an upper limit on server batches
	sntId = 0
	if len(sys.argv) > 1:
		translateStdinInBatches()
	else:
		startServ()
