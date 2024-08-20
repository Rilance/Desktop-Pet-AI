前言
创建这个项目的原因是我的好奇心与不知名的心理活动，有很多细节暂时没有完善......不管怎么说，这都是我的第一个作品......希望这个存储库没人看到，我可是选择了私人的。
鸣谢：
LLM：感谢text-generation-webui项目
TTS：感谢GPT-Sovits项目
PUI：感谢MyFlowingFireflyWife项目
具体该感谢谁我不知道......反正我不会做出什么亏心事。感谢他们。
解释
我受不了Github存储库的上传机制......所以我将文件删了......总共3个部分，源码可以从以下存储库找到：
TTS:https://github.com/RVC-Boss/GPT-SoVITS
LLM:https://github.com/oobabooga/text-generation-webui
PUI:https://github.com/PYmili/MyFlowingFireflyWife
我将我使用的三个源码压缩上传，其中有一部分应该修改，看下面的部分吧

从源代码构建：

1.TTS:
将TTS.zip解压缩至TTS（根目录，不要将一整个文件夹放入TTS，约23个项目），之后按照官方的版本进行部署。我使用python的venv虚拟环境，没有使用Conda：
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
否则你需要修改批处理文件
TTS用的是V1模型...因为我只训练了V1模型...
你可以在这里下载模型：
https://huggingface.co/RaidenSilver/TTS/tree/main
中国大陆的用户可以在其他地方下载......
双击API.bat,看看是否有效果

2.LLM：
将LLM.zip解压至LLM（同样不要将文件夹放入，而是文件，约48个项目）
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
然后在这里下载对话模型：
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
我选择的是llama-2-7b-chat.Q4_K_S.gguf，为低配电脑选择的：
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf?download=true
中国大陆的用户去找镜像站吧
将模型文件放在Model文件夹
双击API.bat文件尝试是否有效果

3.PUI：
将PUI.zip解压至PUI（同样不要将文件夹放入，而是文件，约13个项目）
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

这样三个部分就完成了（应该？）
最后确认一下模型是否都在......

然后双击LLM文件夹中的API.bat和TTS文件夹中的API.bat，最后等两个加载完后打开PUI中的Start_With_API.bat
然后就可以尽情享受了

关于错误
我想，我应该很需要知道哪里有问题，然后修改

关于更新
有时间就更新

关于版本
恐怕不会设置版本

关于使用
我不清楚各类开源协议，只要不拿去犯法就行。对了，不支持倒卖（虽然应该没有人会去倒卖，但我总是看到“倒钩死马（或全家）”......）

结尾
我不希望有人看到这个存储库，但这个Read me.md却像是向别人介绍......用来对之后的我作为介绍吧
在这里找到整合包：
https://huggingface.co/RaidenSilver/desktop-pet-AI
