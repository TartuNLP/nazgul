# SockEyeServer
Neural machine translation as a server, using SockEye

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
