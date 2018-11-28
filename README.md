# SockEyeServer
Neural machine translation as a server, using SockEye

## Requirements:

```
pip3 install mxnet sockeye mosestokenizer estnltk
```

+ https://github.com/TartuNLP/truecaser

## Usage in command-line:

```
head -3 estonian-text | ./nmtserver.py lv fml > latvian-output
```

## Usage as a socket server:

```
./nmtserver.py
```
