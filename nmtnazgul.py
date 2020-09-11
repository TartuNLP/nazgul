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



### Legacy stuff
# supportedStyles = { 'fml', 'inf', 'auto' }
# styleToDomain = { 'fml': 'ep', 'inf': 'os', 'auto': 'pc' }
# supportedStyles = { "os", "un", "dg", "jr", "ep", "pc", "em", "nc" }
# extraSupportedOutLangs = { 'est': 'et', 'lav': 'lv', 'eng': 'en', 'rus': 'ru', 'fin': 'fi', 'lit': 'lt', 'ger': 'de' }
# defaultStyle = 'auto'

supportedOutLangs = {'et', 'fi', 'vro', 'sme', 'sma'}
defaultOutLang = 'et'


#############################################################################################
###################################### STDIN and Server #####################################
#############################################################################################

# TODO: Add this to a real config file
def getConf(rawConf):
    style = 'auto'
    outlang = 'en'

    for field in rawConf.split(','):
        if field in supportedStyles:
            style = field
        if field in supportedOutLangs:
            outlang = field
        #if field in extraSupportedOutLangs:
        #    outlang = extraSupportedOutLangs[field]

    return style, outlang


def parseInput(rawText):
    global supportedStyles, defaultStyle, supportedOutLangs, defaultOutLang

    try:
        fullText = rawText['src']
        rawStyle, rawOutLang = getConf(rawText['conf'])

        livesubs = "|" in fullText

        sentences = fullText.split("|") if livesubs else sent_tokenize(fullText)
        delim = "|" if livesubs else " "

    except KeyError:
        sentences = rawText['sentences']
        rawStyle = rawText['outStyle']
        rawOutLang = rawText['outLang']
        delim = False

    if rawStyle not in supportedStyles:
        # raise ValueError("style bad: " + rawStyle)
        rawStyle = defaultStyle

    if rawOutLang not in supportedOutLangs:
        # raise ValueError("out lang bad: " + rawOutLang)
        rawOutLang = defaultOutLang

    outputLang = rawOutLang
    # outputStyle = styleToDomain[rawStyle]
    outputStyle = None

    return sentences, outputLang, outputStyle, delim


def decodeRequest(rawMessage):
    struct = json.loads(rawMessage.decode('utf-8'))

    segments, outputLang, outputStyle, delim = parseInput(struct)

    return segments, outputLang, outputStyle, delim


def encodeResponse(translationList, delim):
    translationText = delim.join(translationList)

    result = json.dumps({'raw_trans': ['-'],
                         'raw_input': ['-'],
                         'final_trans': translationText})

    return bytes(result, 'utf-8')


def serverTranslationFunc(rawMessage, models):
    segments, outputLang, outputStyle, delim = decodeRequest(rawMessage)

    translations, _, _, _ = translator.translate(models, segments, outputLang, outputStyle, getCnstrs())

    return encodeResponse(translations, delim)


def startTranslationServer(models, ip, port):
    log("started server")

    # start listening as a socket server; apply serverTranslationFunc to incoming messages to genereate the response
    sock.startServer(serverTranslationFunc, (models,), port=port, host=ip)

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

    # load translation and preprocessing models using paths
    models = translator.load_models(args.model, args.spm_model, args.tc_model, args.cpu)

    startTranslationServer(models, args.ip, args.port)
