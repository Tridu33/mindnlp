# 使用Trainer

在[Quick Start](./quick_start.md) 教程中，我们学习了使用 `Traienr` API 来微调模型。

本教程将全面介绍如何配置 `Trainer` 以训练模型以获得最佳结果。

MindNLP 的 `TrainingArguments` 和 `Trainer` 类简化了训练机器学习模型的过程。 `TrainingArguments` 允许您轻松配置基本的训练参数。然后，`Trainer` 利用这些配置来有效地处理整个训练循环。这些工具共同消除了训练任务的大部分复杂性，使新手和专家都能够有效地优化他们的模型。

## 配置训练参数

通过创建 `TrainingArguments` 对象，您可以为训练过程指定所需的配置。

以下是实例化 `TrainingArugments` 对象的代码片段：

```python
from mindnlp.engine import TrainingArguments

training_args = TrainingArguments(
    output_dir="../../output",
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50
)
```

让我们分解代码以详细了解每个参数。

### 基本参数

* 代码_9_占位符
  
该参数指定保存模型检查点和训练输出的目录。

* 代码_10_PLACEHOLDER
  
此参数定义模型将经历的整个数据集的训练周期总数。
  
调整训练纪元的数量直接影响模型从数据集中学习的效果。较多的 epoch 允许模型从训练数据中学习更多信息并取得更好的结果。然而，设置太多的纪元可能会导致过度拟合，模型在训练数据上表现良好，但在新的、未见过的数据上表现不佳。

### 优化器参数

`TrainingArguments` 允许您指定优化器的参数，其中包括：

* 代码_12_PLACEHOLDER
  
该参数指定用于训练模型的优化器。到目前为止，MindNLP 支持 AdamW 和 SGD。您可以通过将此参数设置为 `"adamw"` 或 `"sgd"` 来选择优化器。默认情况下，`TrainingArguments` 选择 AdamW。

* 代码_16_PLACEHOLDER
  
该参数设置优化器的初始学习率，确定损失最小化过程中每次迭代的步长。
  
这是检查训练过程是否无法正确收敛的首要参数之一。
  
较高的学习率可以更快地收敛，但如果太高，则可能会超出最小值，从而因跳动或偏离最佳权重而导致不稳定。相反，学习率太低可能导致收敛缓慢，可能陷入局部极小值。

* 优化器的高级参数
  
使用以下高级参数的默认值足以满足大多数训练的需要。有兴趣的读者和专家可以对其进行调整，以获得更好的训练效果。

- `weight_decay`：此参数有助于通过惩罚大权重来防止过度拟合。权重衰减是损失函数中添加的正则化项，可有效降低模型中权重的大小。
    
- `adam_beta1` 和 `adam_beta2`：这些参数特定于 AdamW 优化器。 `adam_beta1` 控制一阶矩估计的指数衰减率（类似于动量项），而 `adam_beta2` 控制二阶矩估计的指数衰减率（与自适应学习率相关）。
    
- `adam_epsilon`：这是一个非常小的数字，可防止在 Adam 优化器的实现中被零除。它用于提高数值稳定性。
    
- `max_grad_norm`：用于梯度裁剪，这是一种防止深度神经网络中梯度爆炸的技术。将梯度剪切到指定的范数有助于稳定训练过程。

### 批量大小参数

与批量大小相关的参数允许您控制在训练和评估阶段一次处理多少个示例。以下是这些参数的摘要：

* 代码_24_PLACEHOLDER
  
该参数设置每个训练步骤的批量大小。
  
大批量可以加快训练速度并使更新更加一致，但它可能需要更多内存，并且可能会收敛到次优最小值。
  
另一方面，较小的批量大小需要较少的内存，并且可能有助于模型更好地学习，尽管它可能会减慢训练过程。

* 代码_25_PLACEHOLDER
  
该参数设置评估中每个步骤的批量大小。

注意：如果您已经预先对数据集进行了批处理，例如通过调用 `dataset.batch()`，您可能希望将 `TrainingArguments` 中的批处理大小设置为 1，因此 `Trainer` 不会在已批处理的数据集上进一步进行批处理。

### 评估、保存和记录策略

`TrainingArguments` 允许您定义训练过程中的评估、保存和记录策略。

#### 评估策略

`evaluation_strategy` 参数确定在训练过程中何时评估模型。评估对于监控验证数据集上的模型性能至关重要，验证数据集通常与训练数据集不同。

执行评估的策略可以是：

-“否”：不进行评估。
- “步骤”：按照训练步骤的指定间隔进行评估。
如果选择“steps”策略，则需要指定 `eval_steps` 来控制每次评估之间应发生多少训练步骤。
- “epoch”：评估发生在每个 epoch 结束时。

#### 保存策略

`save_strategy` 参数控制在训练过程中何时保存模型的状态。保存对于保留训练不同阶段的模型检查点至关重要，这对于恢复或进一步微调非常有用。

节省策略可以是：

- “否”：不执行保存。
- “步骤”：按照训练步骤的指定间隔进行保存。
如果选择“steps”策略，则需要指定 `save_steps` 来控制每个保存的检查点之间应发生多少步训练。
- “纪元”：保存发生在每个纪元结束时。

#### 记录策略

`logging_strategy` 参数确定在训练过程中何时应记录模型的训练指标。日志记录对于跟踪进度、理解模型行为和诊断训练期间的问题非常重要。

日志记录策略可以是：

- “否”：不执行任何记录。
- “步骤”：记录按照训练步骤的指定间隔进行。
如果选择“steps”策略，则需要指定logging_steps来控制每个日志记录事件之间应发生多少训练步骤。
- “纪元”：记录发生在每个纪元结束时。

## 创建训练器

MindNLP 中的 `Trainer` 接受来自 `TrainingArgument` 对象的配置并处理整个训练循环。

假设您已经定义了 `model`、`dataset_train`、`dataset_val` 和函数 `compute_metrics`，例如在 [Quick Start](./quick_start.md) 教程中，可以使用以下代码创建 `Trainer` 对象：

```python
from mindnlp.engine import Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    args=training_args,
)
```

以下是用于自定义训练行为的关键参数的解释：

* `model`：传递您计划训练的模型实例。这是将接受训练过程的主要对象。

* `args`：您的 `TrainingArgument` 对象，用于设置训练配置。

* `train_dataset`、`eval_dataset`：这些数据集分别用于训练或评估模型。请记住按照 [Data Preprocess](./data_preprocess.md) 教程中的方式预处理数据集。

* `compute_metrics`：根据模型的预测计算特定性能指标的函数。它采用 `mindnlp.engine.utils.EvalPrediction` 对象，其中包含预测和标签，并返回指标结果。
  
`compute_metrics` 函数的示例可以定义如下：
  
```python
导入评估
将 numpy 导入为 np
从 mindnlp.engine.utils 导入 EvalPrediction
  
指标=评估.负载（“准确度”）
  
defcompute_metrics(eval_pred: EvalPrediction):
logits，标签= eval_pred
预测 = np.argmax(logits, axis=-1)
返回 metric.compute(预测=预测，参考=标签)
```
  
请注意，目前我们仍然需要从HF评估模块加载准确性指标。

创建训练器后，运行 `trainer.train()` 开始训练过程。
