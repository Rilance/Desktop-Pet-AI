# Preface

Do you want an AI that is always by your side, ready to answer your questions? This project is a simple implementation.

## Acknowledgements:

LLM: Thanks to the X.ai
TTS: Thanks to the GPT-Sovits project
PUI: Thanks to the MyFlowingFireflyWife project
I’m not sure who exactly to thank… Anyway, I won’t do anything unethical. Thanks to them.

## Explanation

I can’t stand the upload mechanism of Github repositories… so I deleted the files… There are a total of 2 parts, and the source code can be found in the following repositories:

[TTS](https://github.com/RVC-Boss/GPT-SoVITS)
[PUI](https://github.com/PYmili/MyFlowingFireflyWife)

Building from Source:

## TTS:
Unzip TTS.zip to TTS (root directory, do not place the entire folder into TTS, about 23 items), then deploy according to the official version. I used Python’s venv virtual environment, not Conda:

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Otherwise, you need to modify the batch file.
TTS is using the V2 Models.
You can download the model here: [Huggingface](https://huggingface.co/RaidenSilver/TTS/tree/main)

If TTS reports an error, please try the following command:

```sh
.\venv\Scripts\activate
python
```

```python
import nltk
nltk.set_proxy('http://127.0.0.1:10809') #Replace it with your proxy port
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

## LLM:
I will use Grok's API to obtain LLM support.
You can register and obtain your Grok API at [X](https://x.ai/).
### IMPORTANT:
Fill in the Grok API key yourself, on line 2 of the .\PUI\config.py file.

## PUI:
Unzip PUI.zip to PUI (similarly, do not place the folder, but the files, about 13 items)

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

This completes the three parts (Maybe?).

Finally, confirm if all models are in place…

## STT
This is a local update for converting speech to text. The installation method is as follows:
```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
At the same time, download the model to the root directory.
```git
git clone https://huggingface.co/Systran/faster-whisper-large-v3
```
Now, you can double-click the Main.batfile to start the STT API service.

## About Errors

I think I really need to know where the problems are and then fix them.

## About Updates

I will update when I have time.

## About Versions

I’m afraid I won’t set versions.

## About Usage

I’m not familiar with various open-source licenses, just don’t use it for illegal activities. Oh, and no reselling (although no one would probably resell it, but I always see “reselling is prohibited”).

Find the integration package here: [huggingface](https://huggingface.co/RaidenSilver/desktop-pet-AI)
