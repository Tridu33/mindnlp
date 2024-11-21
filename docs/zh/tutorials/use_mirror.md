# 使用镜子下载模型和数据集

> 警告：本文由机器翻译生成，可能导致质量不佳或信息有误，请谨慎阅读！


虽然官方的拥抱面孔存储库提供了许多高质量的模型和数据集，但由于网络问题，它们可能总是无法访问的。 为了使访问更轻松，MindNLP使您可以从各种拥抱面镜或其他模型存储库中下载模型和数据集。

在这里，我们向您展示如何设置所需的镜子。

您可以通过环境变量设置拥抱面镜，或者在本地更大，指定镜子 `from_pretrained` 下载模型时的方法。

## 通过环境变量设置拥抱面镜

通过Mindnlp使用的拥抱表镜可以通过 `HF_ENDPOINT` 环境变量。

您可以在终端中设置此变量，然后再借鉴Python脚本：
```bash
export HF_ENDPOINT="https://hf-mirror.com" 
```
或使用 `os` 包裹：


```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

如果是 `HF_ENDPOINT` 用户未明确设置变量，MindNLP默认情况下将使用'https://hf-mirror.com'。 您可以将其更改为官方的HuggingFace存储库“ https://huggingface.co'。

 **重要的：** 

URL不应包括最后一个'/'。 将Varialble设置为“ https://hf-mirror.com”将有效，同时将其设置为'https://hf-mirror.com/'将导致错误。

 **重要的：** 

作为 `HF_ENDPOINT` 在MindNLP的初始导入期间读取变量，设置 `HF_ENDPOINT` 在导入MindNLP之前。 如果您在Jupyter笔记本电脑中，并且已经导入MindNLP软件包，则可能需要重新启动笔记本以进行更改才能生效。

现在您可以下载所需的模型，例如：


```python
from mindnlp.transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
```

## 指定拥抱面镜 `from_pretrained` 方法

您也可以在环境变量上全球设置拥抱面镜，而是可以在 `from_pretrained` 方法。

例如：


```python
from mindnlp.transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', mirror='modelscope', revision='master')
```

MindNLP接受以下选项 `mirror` 争论：

*'huggingface'

从通过指定的拥抱面镜下载 `HF_ENDPOINT` 环境变量。 默认情况下，它指向 [HF MIRROR](https://hf-mirror.com).

*'ModelsCope'

从 [ModelsCope](https://www.modelscope.cn).

*'Wisemodel'

从 [始智ai](https://www.wisemodel.cn).

*'gitee'

从 [Gitee Ai拥抱脸部存储库](https://ai.gitee.com/huggingface).

*'aifast'

从 [ai快站](https://aifasthub.com).

请注意，并非所有型号都可以从单个镜像中找到，您可能需要检查要下载的型号是否由您选择的镜像提供。

除了指定镜子外，您还需要指定 `revision` 争论。 这 `revision` 根据您选择的镜子，参数可以是“主”或“主”。 默认情况下，`revision='main'`.

*如果是 `mirror` 是“拥抱面”，“ Wisemodel”或“ Gitee”`revision='main'`.

*如果是 `mirror` 是“ ModelsCope”，设置 `revision='master'`.

*如果是 `mirror` 是“ aifast”，`revision` 不需要指定。
