# Preface

Do you want an AI that is always by your side, ready to answer your questions? This project is a simple implementation.

## Acknowledgements:

LLM: Thanks to the text-generation-webui project
TTS: Thanks to the GPT-Sovits project
PUI: Thanks to the MyFlowingFireflyWife project
I’m not sure who exactly to thank… Anyway, I won’t do anything unethical. Thanks to them.

## Explanation

I can’t stand the upload mechanism of Github repositories… so I deleted the files… There are a total of 3 parts, and the source code can be found in the following repositories:

[TTS](https://github.com/RVC-Boss/GPT-SoVITS)
[LLM](https://github.com/oobabooga/text-generation-webui)
[PUI](https://github.com/PYmili/MyFlowingFireflyWife)

Building from Source:

### TTS:
Unzip TTS.zip to TTS (root directory, do not place the entire folder into TTS, about 23 items), then deploy according to the official version. I used Python’s venv virtual environment, not Conda:

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Otherwise, you need to modify the batch file.
~~TTS uses the V1 model… because I only trained the V1 model…~~
TTS is using the V2 Models.
You can download the model here: [huggingface](https://huggingface.co/RaidenSilver/TTS/tree/main)

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

### LLM:
Unzip LLM.zip to LLM (similarly, do not place the folder, but the files, about 48 items)

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dialogue model here: 1, for low-end computers: 
[Llama-2-7B-Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf?download=true)
Double-click the API.bat file to see if it works.

Please note: The LLM module needs to use my updated server.py file, otherwise errors may occur and it may be unstable.

### PUI:
Unzip PUI.zip to PUI (similarly, do not place the folder, but the files, about 13 items)

```sh
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

This completes the three parts (Maybe?).

Finally, confirm if all models are in place…

Then double-click API.bat in the LLM folder and API.bat in the TTS folder. After both are loaded, open Start_With_API.bat in the PUI folder. Then you can enjoy it.

## About Errors

I think I really need to know where the problems are and then fix them.

## About Updates

I will update when I have time.

## About Usage

I’m not familiar with various open-source licenses, just don’t use it for illegal activities. Oh, and no reselling (although no one would probably resell it, but I always see “reselling is prohibited”).

## Conclusion

Find the integration package here: [Huggingface](https://huggingface.co/RaidenSilver/desktop-pet-AI)

## IMPORTANT:
In the LLM folder of the integrated package, the API.bat file needs to be modified as follows:

.\venv\Scripts\python server.py --model-dir "models" --model llama-2-7b-chat.Q4_K_S.gguf --listen --listen-port 7860 --character Copilot --gradio-auth Raiden:13252123393Wf --api
pause

󠁪Otherwise, an error will occur.
