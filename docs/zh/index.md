---
隐藏：
  - navigation
---

> 警告：本文由机器翻译生成，可能导致质量不佳或信息有误，请谨慎阅读！

# <center> Mindnlp

<p align ="center">
<a href ="https://mindnlp.cqu.ai/en/latest/">
<img alt ="docs" src ="https://img.shields.io/badge/docs-latest-blue">
</a>
<a href ="https://github.com/mindspore-lab/mindnlp/blob/master/LICENSE">
<img alt ="GitHub" src ="https://img.shields.io/github/license/mindspore-lab/mindnlp.svg">
</a>
<a href ="https://github.com/mindspore-lab/mindnlp/pulls">
<img alt ="PRs Welcome" src ="https://img.shields.io/badge/PRs-welcome-pink.svg">
</a>
<a href ="https://github.com/mindspore-lab/mindnlp/issues">
<img alt ="open issues" src ="https://img.shields.io/github/issues/mindspore-lab/mindnlp">
</a>
<a href ="https://github.com/mindspore-lab/mindnlp/actions">
<img alt ="ci" src ="https://github.com/mindspore-lab/mindnlp/actions/workflows/ci_pipeline.yaml/badge.svg">
</a>
</p>


### 新闻📢

*🔥 **最新功能** 

*🤗拥抱 *拥抱面 *生态系统，我们使用 **数据集** lib作为支持支持的默认数据集加载程序
有用数据集的安装。
*📝MINDNLP支持NLP任务，例如 *语言模型 *， *机器翻译 *， *问题回答 *， *情感分析 *， *序列标签 *， *摘要 *等。您可以通过 [例子](https://github.com/mindspore-lab/mindnlp/examples/).
*🚀MindNLP当前支持行业领先的大语言模型（LLMS），包括 **骆驼**，，，，**Glm**，，，，**RWKV** 等等。用于与大语言模型有关的支持，包括 ***预训练***，，***微调***， 和 **推理** 演示示例，您可以在 ["llm"目录](https://github.com/mindspore-lab/mindnlp/llm/).
*🤗验证的模型支持 ***拥抱面变压器般的API***， 包括 **60+** 类似的模型 **[伯特](https://github.com/mindspore-lab/mindnlp/mindnlp/transformers/models/bert)**，，，，**[罗伯塔](https://github.com/mindspore-lab/mindnlp/mindnlp/transformers/models/roberta)**，，，，**[GPT2](https://github.com/mindspore-lab/mindnlp/mindnlp/transformers/models/gpt2)**，，，，**[T5](https://github.com/mindspore-lab/mindnlp/mindnlp/transformers/models/t5)**， ETC。
您可以通过以下代码段来轻松使用它们：
```python
    from mindnlp.transformers import AutoModel

    model = AutoModel.from_pretrained('bert-base-cased')
    ```

### 安装

#### 从PYPI安装

您可以安装上传到PYPI的MindNLP的官方版本。

```bash
pip install mindnlp
```

#### 每日构建

您可以从中下载MindNLP每日轮 [这里](https://repo.mindspore.cn/mindspore-lab/mindnlp/newest/any/).

#### 从源安装

要从源安装MindNLP，请运行：

```bash
pip install git+https://github.com/mindspore-lab/mindnlp.git
# or
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
bash scripts/build_and_reinstall.sh
```

#### 版本兼容性

|MindNLP版本|Mindspore版本|支持的Python版本|
|-----------------|-------------------|--------------------------|
|掌握|每日构建|> = 3.7.5，<= 3.9|
|0.1.1|> = 1.8.1，<= 2.0.0|> = 3.7.5，<= 3.9|
|0.2.x|> = 2.1.0|> = 3.8，<= 3.9|

### 介绍

MindNLP是基于Mindspore的开源NLP库。 它支持一个平台来解决自然语言处理任务，其中包含NLP中许多常见方法。 它可以帮助研究人员和开发人员更方便，迅速地构建和培训模型。

主分支与 **思维大师**.

#### 主要功能

- **全面的数据处理** ：将几个经典的NLP数据集包装到友好的模块中，以便于使用，例如Multi330k，Squad，Conll等。
- **友好的NLP模型工具集** ：MindNLP提供各种可配置的组件。 使用MindNLP自定义模型很友好。
- **易于使用的引擎** ：MindNLP简化复杂的训练过程。 它支持培训师和评估器界面，以轻松培训和评估模型。


### 支持的模型

由于支持模型太多，请检查 [这里](https://mindnlp.cqu.ai/supported_models) 

<！ -##教程

- （更多教程列表...） - >

<！ -##注意 - >

### 执照

该项目在 [Apache 2.0许可证](LICENSE).

### 反馈和联系

动态版本仍在开发中，如果您发现任何问题或对新功能有任何想法，请随时通过 [Github问题](https://github.com/mindspore-lab/mindnlp/issues).

### 致谢

Mindspore是一个开源项目，欢迎任何贡献和反馈。
我们希望工具箱和基准可以为不断增长的研究服务
通过提供灵活的和标准化的工具包来重新实现现有方法，社区
并开发自己的新语义细分方法。

### 引用

如果您发现此项目在您的研究中有用，请考虑引用：

```latex
@misc{mindnlp2022,
    title={{MindNLP}: Easy-to-use and high-performance NLP and LLM framework based on MindSpore},
    author={MindNLP Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindnlp}},
    year={2022}
}
```