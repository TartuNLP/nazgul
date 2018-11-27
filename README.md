# SockEyeServer
Neural machine translation as a server, using SockEye

Model files are in 'en-et-lv-model', preprocessing models in 'preprocessing-models'.

If you have all dependencies installed, translation should run fine from the notebook 'translation.ipynb',
if you download the whole folder together. There is a widget version and a code version,
so installing ipywidgets is not necessary.

I added one function to the applytc.py module from https://github.com/TartuNLP/truecaser
to be able to call it from the notebook.
