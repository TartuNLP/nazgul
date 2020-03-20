# Nazgul
Multilingual multi-domain Neural machine translation server based on Amazon SockEye. One or several Nazgul instances can serve translations to [Sauron](https://github.com/TartuNLP/sauron).

## Requirements:

+ https://github.com/TartuNLP/truecaser

```
pip3 install mxnet sentencepiece sockeye mosestokenizer estnltk
```

## Usage in command-line:

```
cat input_text | ./nmtnazgul.py  translation_model  truecaser_model  segmenter_model [output_lang [output_style]]

translation_model: path to a trained Sockeye model folder
truecaser_model: path to a trained TartuNLP truecaser model file
segmenter_model: path to a trained Google SentencePiece model file

output_lang: output language (one of the following: lt, fi, de, ru, lv, et, en)
output_style: output style (one of the following: inf, fml, auto; default: auto)
```

## Usage as a socket server:

```
./nmtnazgul.py translation_model  truecaser_model  segmenter_model
```

The server receives requests from [Sauron](https://github.com/TartuNLP/sauron) and sends back translations.
