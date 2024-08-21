Preface

The reason for creating this project is my curiosity and some unknown psychological activities. There are many details that haven’t been perfected yet… Anyway, this is my first work… I hope no one sees this repository, as I have chosen it to be private.

Acknowledgements:

LLM: Thanks to the text-generation-webui project
TTS: Thanks to the GPT-Sovits project
PUI: Thanks to the MyFlowingFireflyWife project
I’m not sure who exactly to thank… Anyway, I won’t do anything unethical. Thanks to them.

Explanation

I can’t stand the upload mechanism of Github repositories… so I deleted the files… There are a total of 3 parts, and the source code can be found in the following repositories:

TTS:https://github.com/RVC-Boss/GPT-SoVITS
LLM:https://github.com/oobabooga/text-generation-webui
PUI:https://github.com/PYmili/MyFlowingFireflyWife

Building from Source:

TTS:
Unzip TTS.zip to TTS (root directory, do not place the entire folder into TTS, about 23 items), then deploy according to the official version. I used Python’s venv virtual environment, not Conda:

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

Otherwise, you need to modify the batch file.
TTS uses the V1 model… because I only trained the V1 model…
You can download the model here: https://huggingface.co/RaidenSilver/TTS/tree/main

LLM:
Unzip LLM.zip to LLM (similarly, do not place the folder, but the files, about 48 items)

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

Then download the dialogue model here: 1, for low-end computers: 
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf?download=true
Double-click the API.bat file to see if it works.

PUI:
Unzip PUI.zip to PUI (similarly, do not place the folder, but the files, about 13 items)

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

This completes the three parts (May be?).

Finally, confirm if all models are in place…

Then double-click API.bat in the LLM folder and API.bat in the TTS folder. After both are loaded, open Start_With_API.bat in the PUI folder. Then you can enjoy it.

About Errors

I think I really need to know where the problems are and then fix them.

About Updates

I will update when I have time.

About Versions

I’m afraid I won’t set versions.

About Usage

I’m not familiar with various open-source licenses, just don’t use it for illegal activities. Oh, and no reselling (although no one would probably resell it, but I always see “reselling is prohibited”).

Conclusion

I hope no one sees this repository, but this Readme.md seems like an introduction to others… Let’s use it as an introduction for my future self.

Find the integration package here: https://huggingface.co/RaidenSilver/desktop-pet-AI
