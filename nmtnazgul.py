#!/usr/bin/python3

import sock
import translator

import sys
import json
from time import time
from nltk import sent_tokenize

from constraints import getPolitenessConstraints as getCnstrs
from log import log

import configargparse
import yaml



# TODO: Add this to a real config file
### Legacy stuff
# supportedStyles = { 'fml', 'inf', 'auto' }
# styleToDomain = { 'fml': 'ep', 'inf': 'os', 'auto': 'pc' }
# supportedStyles = { "os", "un", "dg", "jr", "ep", "pc", "em", "nc" }
# extraSupportedOutLangs = { 'est': 'et', 'lav': 'lv', 'eng': 'en', 'rus': 'ru', 'fin': 'fi', 'lit': 'lt', 'ger': 'de' }
# defaultStyle = 'auto'


def get_conf(raw_conf):
    style = 'auto'
    outlang = 'en'

    for field in raw_conf.split(','):
        if field in supportedStyles:
            style = field
        if field in supportedOutLangs:
            outlang = field
        #if field in extraSupportedOutLangs:
        #    outlang = extraSupportedOutLangs[field]

    return style, outlang


def parse_input(raw_msg):
    global supportedStyles, defaultStyle, supportedOutLangs, defaultOutLang

    try:
        fullText = raw_msg['src']
        raw_style, raw_out_lang = get_conf(raw_msg['conf'])

        livesubs = "|" in fullText

        sentences = fullText.split("|") if livesubs else sent_tokenize(fullText)
        delim = "|" if livesubs else " "

    except KeyError:
        sentences = raw_msg['sentences']
        raw_style = raw_msg['outStyle']
        raw_out_lang = raw_msg['outLang']
        delim = False

    if raw_style not in supportedStyles:
        # raise ValueError("style bad: " + rawStyle)
        raw_style = defaultStyle

    if raw_out_lang not in supportedOutLangs:
        # raise ValueError("out lang bad: " + rawOutLang)
        raw_out_lang = defaultOutLang

    outputLang = raw_out_lang
    # outputStyle = styleToDomain[rawStyle]
    outputStyle = None

    return sentences, outputLang, outputStyle, delim


def decode_request(raw_message):
    struct = json.loads(raw_message.decode('utf-8'))

    segments, output_lang, output_style, delim = parse_input(struct)

    return segments, output_lang, output_style, delim


def encode_response(translation_list, delim):
    translation_text = delim.join(translation_list)

    # TODO: Check what to do with raw_trans and raw_input? Some legacy thing?
    result = json.dumps({'raw_trans': ['-'],
                         'raw_input': ['-'],
                         'final_trans': translation_text})

    return bytes(result, 'utf-8')


def server_translation_func(raw_message, sockeye_models):
    segments, output_lang, output_style, delim = decode_request(raw_message)

    translations, _, _, _ = translator.translate(sockeye_models, segments, output_lang, output_style, getCnstrs())

    return encode_response(translations, delim)


def start_translation_server(sockeye_models, ip, port):
    log("started server")

    # start listening as a socket server; apply serverTranslationFunc to incoming messages to generate the response
    sock.startServer(server_translation_func, (sockeye_models,), port=port, host=ip)

#### TODO: Add this functionality separately
"""
def translateStdinInBatches(models, outputLang, outputStyle):
    '''Read lines from STDIN and treat each as a segment to translate;
	translate them and print out tab-separated scores (decoder log-prob)
	and the translation outputs'''

    # read STDIN as a list of segments
    lines = [line.strip() for line in sys.stdin]

    # translate segments and get translations and scores
    translations, scores, _, _ = translator.translate(models, lines, outputLang, outputStyle, getCnstrs())

    # print each score and translation, separated with a tab
    for translation, score in zip(translations, scores):
        print("{0}\t{1}".format(score, translation))
"""

#############################################################################################
################################## Cmdline and main block ###################################
#############################################################################################


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        description="Backend NMT server for Sockeye models. Further info: http://github.com/tartunlp/nazgul",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=["config/config.ini"]
    )
    parser.add_argument('--config', type=yaml.safe_load)
    parser.add_argument("--model", "-m", required=True, type=str, help="Path to Sockeye model folder")
    parser.add_argument("--spm_model", "-s", required=True, type=str,
                        help="Path to trained Google SentencePiece model file")
    parser.add_argument("--tc_model", "-t", type=str, default=None,
                        help="Path to trained TartuNLP truecaser model file")
    parser.add_argument("--cpu", default=False, action="store_true",
                        help="Use CPU-s instead of GPU-s for serving")

    parser.add_argument("--port", "-p", type=int, default=12345,
                        help="Port to run the service on.")
    parser.add_argument("--ip", "-i", type=str, default="127.0.0.1",
                        help="IP to run the service on")

    parser.add_argument("--langs", "-l", type=str, action="append",
                        help="Comma separated string on supported languages.")
    parser.add_argument("--domains", "-d", type=str, action="append",
                        help="Comma separated string on supported domains.")

    args = parser.parse_args()
    print(args)

    ### Legacy stuff
    # read translation and preprocessing model paths off cmdline
    # modelPaths = readCmdlineModels()

    # read output language and style off cmdline -- both are optional and will be "None" if not given
    # olang, ostyle = readLangAndStyle()
    supportedOutLangs = args.langs
    defaultOutLang = "et"
    # load translation and preprocessing models using paths
    models = translator.load_models(args.model, args.spm_model, args.tc_model, args.cpu)

    start_translation_server(models, args.ip, args.port)
