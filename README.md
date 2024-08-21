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




## Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, Korean, Cantonese and Chinese.

4. **WebUI Tools:** Integrated tools include voice accompaniment separation, automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

**Check out our [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw) here!**

Unseen speakers few-shot fine-tuning demo:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**User guide: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## Installation

For users in China, you can [click here](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official) to use AutoDL Cloud Docker to experience the full functionality online.

### Tested Environments

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.2.2, macOS 14.4.1 (Apple silicon)
- Python 3.9, PyTorch 2.2.2, CPU devices

_Note: numba==0.56.4 requires py<3.11_

### Windows

If you are a Windows user (tested with win>=10), you can [download the integrated package](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true) and double-click on _go-webui.bat_ to start GPT-SoVITS-WebUI.

**Users in China can [download the package here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO).**

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**

1. Install Xcode command-line tools by running `xcode-select --install`.
2. Install FFmpeg by running `brew install ffmpeg`.
3. Install the program by running the following commands:

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
pip install -r requirements.txt
```

### Install Manually

#### Install FFmpeg

##### Conda Users

```bash
conda install ffmpeg
```

##### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root.

Install [Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) (Korean TTS Only)

##### MacOS Users
```bash
brew install ffmpeg
```

#### Install Dependences

```bash
pip install -r requirements.txt
```

### Using Docker

#### docker-compose.yaml configuration

0. Regarding image tags: Due to rapid updates in the codebase and the slow process of packaging and testing images, please check [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) for the currently packaged latest images and select as per your situation, or alternatively, build locally using a Dockerfile according to your own needs.
1. Environment Variables：

- is_half: Controls half-precision/double-precision. This is typically the cause if the content under the directories 4-cnhubert/5-wav32k is not generated correctly during the "SSL extracting" step. Adjust to True or False based on your actual situation.

2. Volumes Configuration，The application's root directory inside the container is set to /workspace. The default docker-compose.yaml lists some practical examples for uploading/downloading content.
3. shm_size： The default available memory for Docker Desktop on Windows is too small, which can cause abnormal operations. Adjust according to your own situation.
4. Under the deploy section, GPU-related settings should be adjusted cautiously according to your system and actual circumstances.

#### Running with docker compose

```
docker compose -f "docker-compose.yaml" up -d
```

#### Running with docker command

As above, modify the corresponding parameters based on your actual situation, then run the following command:

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## Pretrained Models

**Users in China can [download all these models here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#nVNhX).**

1. Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

2. Download G2PW models from [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip), unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.(Chinese TTS Only)

3. For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

4. For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/asr/models`.

5. For English or Japanese ASR (additionally), download models from [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) and place them in `tools/asr/models`. Also, [other models](https://huggingface.co/Systran) may have the similar effect with smaller disk footprint. 

## Dataset Format

The TTS annotation .list file format:

```
vocal_path|speaker_name|language|text
```

Language dictionary:

- 'zh': Chinese
- 'ja': Japanese
- 'en': English
- 'ko': Korean
- 'yue': Cantonese
  
Example:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## Finetune and inference

 ### Open WebUI

 #### Integrated Package Users

 Double-click `go-webui.bat`or use `go-webui.ps`
 if you want to switch to V1,then double-click`go-webui-v1.bat` or use `go-webui-v1.ps`

 #### Others

 ```bash
 python webui.py <language(optional)>
 ```

 if you want to switch to V1,then

 ```bash
 python webui.py v1 <language(optional)>
 ```
Or maunally switch version in WebUI

 ### Finetune

 #### Path Auto-filling is now supported

     1.Fill in the audio path

     2.Slice the audio into small chunks

     3.Denoise(optinal)

     4.ASR

     5.Proofreading ASR transcriptions

     6.Go to the next Tab, then finetune the model

 ### Open Inference WebUI

 #### Integrated Package Users

 Double-click `go-webui-v2.bat` or use `go-webui-v2.ps` ,then open the inference webui at  `1-GPT-SoVITS-TTS/1C-inference` 

 #### Others

 ```bash
 python GPT_SoVITS/inference_webui.py <language(optional)>
 ```
 OR

 ```bash
 python webui.py
 ```
then open the inference webui at `1-GPT-SoVITS-TTS/1C-inference`

 ## V2 Release Notes

New Features:

1. Support Korean and Cantonese

2. An optimized text frontend

3. Pre-trained model extended from 2k hours to 5k hours

4. Improved synthesis quality for low-quality reference audio 

    [more details](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7) ) 

Use v2 from v1 environment: 

1. `pip install -r requirements.txt` to update some packages

2. Clone the latest codes from github.

3. Download v2 pretrained models from [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) and put them into `GPT_SoVITS\pretrained_models\gsv-v2final-pretrained`.

    Chinese v2 additional: [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（Download G2PW models,  unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.
     
## Todo List

- [x] **High Priority:**

  - [x] Localization in Japanese and English.
  - [x] User guide.
  - [x] Japanese and English dataset fine tune training.

- [ ] **Features:**
  - [x] Zero-shot voice conversion (5s) / few-shot voice conversion (1min).
  - [x] TTS speaking speed control.
  - [ ] ~~Enhanced TTS emotion control.~~
  - [ ] Experiment with changing SoVITS token inputs to probability distribution of GPT vocabs (transformer latent).
  - [x] Improve English and Japanese text frontend.
  - [ ] Develop tiny and larger-sized TTS models.
  - [x] Colab scripts.
  - [ ] Try expand training dataset (2k hours -> 10k hours).
  - [x] better sovits base model (enhanced audio quality)
  - [ ] model mix

## (Additional) Method for running from the command line
Use the command line to open the WebUI for UVR5
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
<!-- If you can't open a browser, follow the format below for UVR processing,This is using mdxnet for audio processing
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
``` -->
This is how the audio segmentation of the dataset is done using the command line
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips> 
    --hop_size <step_size_for_computing_volume_curve>
```
This is how dataset ASR processing is done using the command line(Only Chinese)
```
python tools/asr/funasr_asr.py -i <input> -o <output>
```
ASR processing is performed through Faster_Whisper(ASR marking except Chinese)

(No progress bars, GPU performance may cause time delays)
```
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```
A custom list save path is enabled

## Credits

Special thanks to the following projects and contributors:

### Theoretical Research
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
### Pretrained Models
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
### Text Frontend for Inference
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [LangSegment](https://github.com/juntaosun/LangSegment)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)
### WebUI Tools
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)

Thankful to @Naozumi520 for providing the Cantonese training set and for the guidance on Cantonese-related knowledge.

## Thanks to all contributors for their efforts

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>




LLM:
Unzip LLM.zip to LLM (similarly, do not place the folder, but the files, about 48 items)

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

Then download the dialogue model here: 1, for low-end computers: 
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf?download=true
Double-click the API.bat file to see if it works.



# Text generation web UI

A Gradio web UI for Large Language Models.

Its goal is to become the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) of text generation.

|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_instruct.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_chat.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/print_default.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/print_parameters.png) |

## Features

* 3 interface modes: default (two columns), notebook, and chat.
* Multiple model backends: [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
* Dropdown menu for quickly switching between different models.
* Large number of extensions (built-in and user-contributed), including Coqui TTS for realistic voice outputs, Whisper STT for voice inputs, translation, [multimodal pipelines](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/multimodal), vector databases, Stable Diffusion integration, and a lot more. See [the wiki](https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions) and [the extensions directory](https://github.com/oobabooga/text-generation-webui-extensions) for details.
* [Chat with custom characters](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#character).
* Precise chat templates for instruction-following models, including Llama-2-chat, Alpaca, Vicuna, Mistral.
* LoRA: train new LoRAs with your own data, load/unload LoRAs on the fly for generation.
* Transformers library integration: load models in 4-bit or 8-bit precision through bitsandbytes, use llama.cpp with transformers samplers (`llamacpp_HF` loader), CPU inference in 32-bit precision using PyTorch.
* OpenAI-compatible API server with Chat and Completions endpoints -- see the [examples](https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples).

## How to install

1) Clone or [download](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) the repository.
2) Run the `start_linux.sh`, `start_windows.bat`, `start_macos.sh`, or `start_wsl.bat` script depending on your OS.
3) Select your GPU vendor when asked.
4) Once the installation ends, browse to `http://localhost:7860/?__theme=dark`.
5) Have fun!

To restart the web UI in the future, just run the `start_` script again. This script creates an `installer_files` folder where it sets up the project's requirements. In case you need to reinstall the requirements, you can simply delete that folder and start the web UI again.

The script accepts command-line flags. Alternatively, you can edit the `CMD_FLAGS.txt` file with a text editor and add your flags there.

To get updates in the future, run `update_wizard_linux.sh`, `update_wizard_windows.bat`, `update_wizard_macos.sh`, or `update_wizard_wsl.bat`.

<details>
<summary>
Setup details and information about installing manually
</summary>

### One-click-installer

The script uses Miniconda to set up a Conda environment in the `installer_files` folder.

If you ever need to install something manually in the `installer_files` environment, you can launch an interactive shell using the cmd script: `cmd_linux.sh`, `cmd_windows.bat`, `cmd_macos.sh`, or `cmd_wsl.bat`.

* There is no need to run any of those scripts (`start_`, `update_wizard_`, or `cmd_`) as admin/root.
* To install the requirements for extensions, you can use the `extensions_reqs` script for your OS. At the end, this script will install the main requirements for the project to make sure that they take precedence in case of version conflicts.
* For additional instructions about AMD and WSL setup, consult [the documentation](https://github.com/oobabooga/text-generation-webui/wiki).
* For automated installation, you can use the `GPU_CHOICE`, `USE_CUDA118`, `LAUNCH_AFTER_INSTALL`, and `INSTALL_EXTENSIONS` environment variables. For instance: `GPU_CHOICE=A USE_CUDA118=FALSE LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh`.

### Manual installation using Conda

Recommended if you have some experience with the command-line.

#### 0. Install Conda

https://docs.conda.io/en/latest/miniconda.html

On Linux or WSL, it can be automatically installed with these two commands ([source](https://educe-ubc.github.io/conda.html)):

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1. Create a new conda environment

```
conda create -n textgen python=3.11
conda activate textgen
```

#### 2. Install Pytorch

| System | GPU | Command |
|--------|---------|---------|
| Linux/WSL | NVIDIA | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121` |
| Linux/WSL | CPU only | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.6` |
| MacOS + MPS | Any | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2` |
| Windows | NVIDIA | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121` |
| Windows | CPU only | `pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2` |

The up-to-date commands can be found here: https://pytorch.org/get-started/locally/.

For NVIDIA, you also need to install the CUDA runtime libraries:

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
```

If you need `nvcc` to compile some library manually, replace the command above with

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda
```

#### 3. Install the web UI

```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r <requirements file according to table below>
```

Requirements file to use:

| GPU | CPU | requirements file to use |
|--------|---------|---------|
| NVIDIA | has AVX2 | `requirements.txt` |
| NVIDIA | no AVX2 | `requirements_noavx2.txt` |
| AMD | has AVX2 | `requirements_amd.txt` |
| AMD | no AVX2 | `requirements_amd_noavx2.txt` |
| CPU only | has AVX2 | `requirements_cpu_only.txt` |
| CPU only | no AVX2 | `requirements_cpu_only_noavx2.txt` |
| Apple | Intel | `requirements_apple_intel.txt` |
| Apple | Apple Silicon | `requirements_apple_silicon.txt` |

### Start the web UI

```
conda activate textgen
cd text-generation-webui
python server.py
```

Then browse to

`http://localhost:7860/?__theme=dark`

##### AMD GPU on Windows

1) Use `requirements_cpu_only.txt` or `requirements_cpu_only_noavx2.txt` in the command above.

2) Manually install llama-cpp-python using the appropriate command for your hardware: [Installation from PyPI](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration).
    * Use the `LLAMA_HIPBLAS=on` toggle.
    * Note the [Windows remarks](https://github.com/abetlen/llama-cpp-python#windows-remarks).

3) Manually install AutoGPTQ: [Installation](https://github.com/PanQiWei/AutoGPTQ#install-from-source).
    * Perform the from-source installation - there are no prebuilt ROCm packages for Windows.

##### Older NVIDIA GPUs

1) For Kepler GPUs and older, you will need to install CUDA 11.8 instead of 12:

```
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-runtime
```

2) bitsandbytes >= 0.39 may not work. In that case, to use `--load-in-8bit`, you may have to downgrade like this:
    * Linux: `pip install bitsandbytes==0.38.1`
    * Windows: `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl`

##### Manual install

The `requirements*.txt` above contain various wheels precompiled through GitHub Actions. If you wish to compile things manually, or if you need to because no suitable wheels are available for your hardware, you can use `requirements_nowheels.txt` and then install your desired loaders manually.

### Alternative: Docker

```
For NVIDIA GPU:
ln -s docker/{nvidia/Dockerfile,nvidia/docker-compose.yml,.dockerignore} .
For AMD GPU: 
ln -s docker/{amd/Dockerfile,intel/docker-compose.yml,.dockerignore} .
For Intel GPU:
ln -s docker/{intel/Dockerfile,amd/docker-compose.yml,.dockerignore} .
For CPU only
ln -s docker/{cpu/Dockerfile,cpu/docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
#Create logs/cache dir : 
mkdir -p logs cache
# Edit .env and set: 
#   TORCH_CUDA_ARCH_LIST based on your GPU model
#   APP_RUNTIME_GID      your host user's group id (run `id -g` in a terminal)
#   BUILD_EXTENIONS      optionally add comma separated list of extensions to build
# Edit CMD_FLAGS.txt and add in it the options you want to execute (like --listen --cpu)
# 
docker compose up --build
```

* You need to have Docker Compose v2.17 or higher installed. See [this guide](https://github.com/oobabooga/text-generation-webui/wiki/09-%E2%80%90-Docker) for instructions.
* For additional docker files, check out [this repository](https://github.com/Atinoda/text-generation-webui-docker).

### Updating the requirements

From time to time, the `requirements*.txt` change. To update, use these commands:

```
conda activate textgen
cd text-generation-webui
pip install -r <requirements file that you have used> --upgrade
```
</details>

<details>
<summary>
List of command-line flags
</summary>

```txt
usage: server.py [-h] [--multi-user] [--character CHARACTER] [--model MODEL] [--lora LORA [LORA ...]] [--model-dir MODEL_DIR] [--lora-dir LORA_DIR] [--model-menu] [--settings SETTINGS]
                 [--extensions EXTENSIONS [EXTENSIONS ...]] [--verbose] [--chat-buttons] [--idle-timeout IDLE_TIMEOUT] [--loader LOADER] [--cpu] [--auto-devices]
                 [--gpu-memory GPU_MEMORY [GPU_MEMORY ...]] [--cpu-memory CPU_MEMORY] [--disk] [--disk-cache-dir DISK_CACHE_DIR] [--load-in-8bit] [--bf16] [--no-cache] [--trust-remote-code]
                 [--force-safetensors] [--no_use_fast] [--use_flash_attention_2] [--use_eager_attention] [--load-in-4bit] [--use_double_quant] [--compute_dtype COMPUTE_DTYPE] [--quant_type QUANT_TYPE]
                 [--flash-attn] [--tensorcores] [--n_ctx N_CTX] [--threads THREADS] [--threads-batch THREADS_BATCH] [--no_mul_mat_q] [--n_batch N_BATCH] [--no-mmap] [--mlock]
                 [--n-gpu-layers N_GPU_LAYERS] [--tensor_split TENSOR_SPLIT] [--numa] [--logits_all] [--no_offload_kqv] [--cache-capacity CACHE_CAPACITY] [--row_split] [--streaming-llm]
                 [--attention-sink-size ATTENTION_SINK_SIZE] [--gpu-split GPU_SPLIT] [--autosplit] [--max_seq_len MAX_SEQ_LEN] [--cfg-cache] [--no_flash_attn] [--no_xformers] [--no_sdpa]
                 [--cache_8bit] [--cache_4bit] [--num_experts_per_token NUM_EXPERTS_PER_TOKEN] [--triton] [--no_inject_fused_mlp] [--no_use_cuda_fp16] [--desc_act] [--disable_exllama]
                 [--disable_exllamav2] [--wbits WBITS] [--groupsize GROUPSIZE] [--no_inject_fused_attention] [--hqq-backend HQQ_BACKEND] [--cpp-runner] [--deepspeed]
                 [--nvme-offload-dir NVME_OFFLOAD_DIR] [--local_rank LOCAL_RANK] [--alpha_value ALPHA_VALUE] [--rope_freq_base ROPE_FREQ_BASE] [--compress_pos_emb COMPRESS_POS_EMB] [--listen]
                 [--listen-port LISTEN_PORT] [--listen-host LISTEN_HOST] [--share] [--auto-launch] [--gradio-auth GRADIO_AUTH] [--gradio-auth-path GRADIO_AUTH_PATH] [--ssl-keyfile SSL_KEYFILE]
                 [--ssl-certfile SSL_CERTFILE] [--subpath SUBPATH] [--api] [--public-api] [--public-api-id PUBLIC_API_ID] [--api-port API_PORT] [--api-key API_KEY] [--admin-key ADMIN_KEY] [--nowebui]
                 [--multimodal-pipeline MULTIMODAL_PIPELINE] [--model_type MODEL_TYPE] [--pre_layer PRE_LAYER [PRE_LAYER ...]] [--checkpoint CHECKPOINT] [--monkey-patch]

Text generation web UI

options:
  -h, --help                                     show this help message and exit

Basic settings:
  --multi-user                                   Multi-user mode. Chat histories are not saved or automatically loaded. Warning: this is likely not safe for sharing publicly.
  --character CHARACTER                          The name of the character to load in chat mode by default.
  --model MODEL                                  Name of the model to load by default.
  --lora LORA [LORA ...]                         The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.
  --model-dir MODEL_DIR                          Path to directory with all the models.
  --lora-dir LORA_DIR                            Path to directory with all the loras.
  --model-menu                                   Show a model menu in the terminal when the web UI is first launched.
  --settings SETTINGS                            Load the default interface settings from this yaml file. See settings-template.yaml for an example. If you create a file called settings.yaml, this
                                                 file will be loaded by default without the need to use the --settings flag.
  --extensions EXTENSIONS [EXTENSIONS ...]       The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.
  --verbose                                      Print the prompts to the terminal.
  --chat-buttons                                 Show buttons on the chat tab instead of a hover menu.
  --idle-timeout IDLE_TIMEOUT                    Unload model after this many minutes of inactivity. It will be automatically reloaded when you try to use it again.

Model loader:
  --loader LOADER                                Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2,
                                                 AutoGPTQ, AutoAWQ.

Transformers/Accelerate:
  --cpu                                          Use the CPU to generate text. Warning: Training on CPU is extremely slow.
  --auto-devices                                 Automatically split the model across the available GPU(s) and CPU.
  --gpu-memory GPU_MEMORY [GPU_MEMORY ...]       Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values
                                                 in MiB like --gpu-memory 3500MiB.
  --cpu-memory CPU_MEMORY                        Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.
  --disk                                         If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.
  --disk-cache-dir DISK_CACHE_DIR                Directory to save the disk cache to. Defaults to "cache".
  --load-in-8bit                                 Load the model with 8-bit precision (using bitsandbytes).
  --bf16                                         Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
  --no-cache                                     Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.
  --trust-remote-code                            Set trust_remote_code=True while loading the model. Necessary for some models.
  --force-safetensors                            Set use_safetensors=True while loading the model. This prevents arbitrary code execution.
  --no_use_fast                                  Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast.
  --use_flash_attention_2                        Set use_flash_attention_2=True while loading the model.
  --use_eager_attention                          Set attn_implementation= eager while loading the model.

bitsandbytes 4-bit:
  --load-in-4bit                                 Load the model with 4-bit precision (using bitsandbytes).
  --use_double_quant                             use_double_quant for 4-bit.
  --compute_dtype COMPUTE_DTYPE                  compute dtype for 4-bit. Valid options: bfloat16, float16, float32.
  --quant_type QUANT_TYPE                        quant_type for 4-bit. Valid options: nf4, fp4.

llama.cpp:
  --flash-attn                                   Use flash-attention.
  --tensorcores                                  NVIDIA only: use llama-cpp-python compiled with tensor cores support. This may increase performance on newer cards.
  --n_ctx N_CTX                                  Size of the prompt context.
  --threads THREADS                              Number of threads to use.
  --threads-batch THREADS_BATCH                  Number of threads to use for batches/prompt processing.
  --no_mul_mat_q                                 Disable the mulmat kernels.
  --n_batch N_BATCH                              Maximum number of prompt tokens to batch together when calling llama_eval.
  --no-mmap                                      Prevent mmap from being used.
  --mlock                                        Force the system to keep the model in RAM.
  --n-gpu-layers N_GPU_LAYERS                    Number of layers to offload to the GPU.
  --tensor_split TENSOR_SPLIT                    Split the model across multiple GPUs. Comma-separated list of proportions. Example: 60,40.
  --numa                                         Activate NUMA task allocation for llama.cpp.
  --logits_all                                   Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower.
  --no_offload_kqv                               Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance.
  --cache-capacity CACHE_CAPACITY                Maximum cache capacity (llama-cpp-python). Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed.
  --row_split                                    Split the model by rows across GPUs. This may improve multi-gpu performance.
  --streaming-llm                                Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.
  --attention-sink-size ATTENTION_SINK_SIZE      StreamingLLM: number of sink tokens. Only used if the trimmed prompt does not share a prefix with the old prompt.

ExLlamaV2:
  --gpu-split GPU_SPLIT                          Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7.
  --autosplit                                    Autosplit the model tensors across the available GPUs. This causes --gpu-split to be ignored.
  --max_seq_len MAX_SEQ_LEN                      Maximum sequence length.
  --cfg-cache                                    ExLlamav2_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader.
  --no_flash_attn                                Force flash-attention to not be used.
  --no_xformers                                  Force xformers to not be used.
  --no_sdpa                                      Force Torch SDPA to not be used.
  --cache_8bit                                   Use 8-bit cache to save VRAM.
  --cache_4bit                                   Use Q4 cache to save VRAM.
  --num_experts_per_token NUM_EXPERTS_PER_TOKEN  Number of experts to use for generation. Applies to MoE models like Mixtral.

AutoGPTQ:
  --triton                                       Use triton.
  --no_inject_fused_mlp                          Triton mode only: disable the use of fused MLP, which will use less VRAM at the cost of slower inference.
  --no_use_cuda_fp16                             This can make models faster on some systems.
  --desc_act                                     For models that do not have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig.
  --disable_exllama                              Disable ExLlama kernel, which can improve inference speed on some systems.
  --disable_exllamav2                            Disable ExLlamav2 kernel.
  --wbits WBITS                                  Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.
  --groupsize GROUPSIZE                          Group size.

AutoAWQ:
  --no_inject_fused_attention                    Disable the use of fused attention, which will use less VRAM at the cost of slower inference.

HQQ:
  --hqq-backend HQQ_BACKEND                      Backend for the HQQ loader. Valid options: PYTORCH, PYTORCH_COMPILE, ATEN.

TensorRT-LLM:
  --cpp-runner                                   Use the ModelRunnerCpp runner, which is faster than the default ModelRunner but doesn't support streaming yet.

DeepSpeed:
  --deepspeed                                    Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.
  --nvme-offload-dir NVME_OFFLOAD_DIR            DeepSpeed: Directory to use for ZeRO-3 NVME offloading.
  --local_rank LOCAL_RANK                        DeepSpeed: Optional argument for distributed setups.

RoPE:
  --alpha_value ALPHA_VALUE                      Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress_pos_emb, not both.
  --rope_freq_base ROPE_FREQ_BASE                If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63).
  --compress_pos_emb COMPRESS_POS_EMB            Positional embeddings compression factor. Should be set to (context length) / (model's original context length). Equal to 1/rope_freq_scale.

Gradio:
  --listen                                       Make the web UI reachable from your local network.
  --listen-port LISTEN_PORT                      The listening port that the server will use.
  --listen-host LISTEN_HOST                      The hostname that the server will use.
  --share                                        Create a public URL. This is useful for running the web UI on Google Colab or similar.
  --auto-launch                                  Open the web UI in the default browser upon launch.
  --gradio-auth GRADIO_AUTH                      Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".
  --gradio-auth-path GRADIO_AUTH_PATH            Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.
  --ssl-keyfile SSL_KEYFILE                      The path to the SSL certificate key file.
  --ssl-certfile SSL_CERTFILE                    The path to the SSL certificate cert file.
  --subpath SUBPATH                              Customize the subpath for gradio, use with reverse proxy

API:
  --api                                          Enable the API extension.
  --public-api                                   Create a public URL for the API using Cloudfare.
  --public-api-id PUBLIC_API_ID                  Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.
  --api-port API_PORT                            The listening port for the API.
  --api-key API_KEY                              API authentication key.
  --admin-key ADMIN_KEY                          API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.
  --nowebui                                      Do not launch the Gradio UI. Useful for launching the API in standalone mode.

Multimodal:
  --multimodal-pipeline MULTIMODAL_PIPELINE      The multimodal pipeline to use. Examples: llava-7b, llava-13b.
```

</details>

## Documentation

https://github.com/oobabooga/text-generation-webui/wiki

## Downloading models

Models should be placed in the folder `text-generation-webui/models`. They are usually downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* GGUF models are a single file and should be placed directly into `models`. Example:

```
text-generation-webui
└── models
    └── llama-2-13b-chat.Q4_K_M.gguf
```

* The remaining model types (like 16-bit transformers models and GPTQ models) are made of several files and must be placed in a subfolder. Example:

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download it via the command-line with 

```
python download-model.py organization/model
```

Run `python download-model.py --help` to see all the options.

## Google Colab notebook

https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb

## Community

* Subreddit: https://www.reddit.com/r/oobabooga/
* Discord: https://discord.gg/jwZCF2dPQN

## Acknowledgment

In August 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) provided a generous grant to encourage and support my independent work on this project. I am **extremely** grateful for their trust and recognition.






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
