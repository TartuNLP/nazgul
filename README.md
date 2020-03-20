# Nazgul
Multilingual multi-domain Neural machine translation server based on Amazon SockEye. One or several Nazgul instances can serve translations to [Sauron](https://github.com/TartuNLP/sauron).

## Requirements:

+ https://github.com/TartuNLP/truecaser

```
pip3 install mxnet sentencepiece sockeye mosestokenizer estnltk
```

## Usage in command-line:

```
head -3 estonian-text | ./nmtserver.py lv fml > latvian-output
```

## Usage as a socket server:

```
./nmtserver.py
```

