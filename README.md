# SockEyeServer
Neural machine translation as a server, using SockEye

## Requirements:

+ https://github.com/TartuNLP/truecaser

```
pip3 install mxnet sentencepiece sockeye estnltk
```

+ `mosestokenizer-py` fork:

```
pip install https://github.com/inoryy/mosestokenizer-py/archive/master.zip
```

## Usage in command-line:

```
head -3 estonian-text | ./nmtserver.py lv fml > latvian-output
```

## Usage as a socket server:

```
./nmtserver.py
```
