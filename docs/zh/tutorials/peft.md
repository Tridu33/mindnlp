# peft


PEFT（参数有效的微调）是一种具有最小参数更新的大型预训练模型，以降低计算成本并保留概括。 在peft中，洛拉（ [低级适应](https://arxiv.org/abs/2106.09685) ）使用低级矩阵来有效调整具有最小额外参数的神经网络的部分。 该技术使您能够训练通常在消费者设备上无法访问的大型型号。

在本教程中，我们将使用MindNLP探索这项技术。 例如，我们将使用MT0模型，该模型是在多语言任务上进行的MT5模型。 您将学习如何初始化，修改和培训模型，从而获得有效的微调实践经验。

## 加载模型并添加PEFT适配器
首先，我们通过向型号加载器提供型号来加载验证的型号 `AutoModelForSeq2SeqLM`。 然后使用PEFT适配器添加到模型 `get_peft_model`，这使模型可以维护其大部分预训练参数，同时有效地使用一组可训练的参数来适应新任务。


```python
from mindnlp.transformers import AutoModelForSeq2SeqLM
from mindnlp.peft import LoraConfig, TaskType, get_peft_model

# Load the pre-trained model
model_name_or_path = "bigscience/mt0-large" 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# Get the model with a PEFT adapter
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

# Print the trainable parameters of the model
model.print_trainable_parameters()
```

 `LoraConfig` 指定应如何配置PEFT适配器：

*`task_type` ：定义任务的类型，在这种情况下，taskType.seq_2_seq_lm用于序列到序列语言建模。
*`inference_mode` ：在训练时应设置为虚假的布尔值，以实现适配器的特定训练功能。
*`r` ：代表适配器一部分的低级矩阵的等级。 较低的等级意味着较小的复杂性，训练的参数较少。
*`lora_alpha` ：lora alpha是重量矩阵的缩放系数。 较高的alpha值为Lora激活分配了更大的权重。
*`lora_dropout` ：设置适配器层中的辍学率以防止过度拟合。

## 准备数据集

要微调模型，让我们使用 [Financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) 数据集。 Financial_phrasebank数据集是专门为金融部门内部的情感分析任务而设计的。 它包含从金融新闻文章中提取的句子，这些句子是根据所表达的情感分类的 - 消极，中立或积极。

尽管数据集是为情感分类任务设计的，但我们在此处将其用于序列到序列任务，以简单。

### 加载数据集
加载数据集 `load_dataset` 来自Mindnlp。

然后将数据改组和分割，分配90％进行培训，验证10％。


```python
from mindnlp.dataset import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree")
train_dataset, validation_dataset = dataset.shuffle(64).split([0.9, 0.1])
```

### 添加文本标签
由于我们正在训练序列到序列模型，因此该模型的输出需要是文本，在我们的情况下是 "negative"，，，，"neutral" 或者 "positive"。 因此，除了每个条目中的数字标签（0、1或2）之外，我们还需要将文本标签添加到。 这是通过 `add_text_label` 功能。 该功能通过培训和验证数据集中的每个条目映射到 `map` API。


```python
classes = dataset.source.ds.features["label"].names
def add_text_label(sentence, label):
    return sentence, label, classes[label.item()]

train_dataset = train_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])
validation_dataset = validation_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])
```

### 令牌化
然后，我们将文本与与MT0模型关联的令牌化。 首先，加载令牌：


```python
from mindnlp.transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

接下来，修改 `BaseMapFunction` 从MindNLP总结了令牌化步骤。

请注意，这两个 `sentence` 和 `text_label` 列需要被象征化。

此外，为了避免由于多个线程试图同时代币化数据而引起的意外行为，我们使用 `Lock` 来自 `threading` 模块以确保仅一个线程可以一次执行令牌化。


```python

import numpy as np
from mindnlp.dataset import BaseMapFunction
from threading import Lock
lock = Lock()

max_length = 128
class MapFunc(BaseMapFunction):
    def __call__(self, sentence, label, text_label):
        lock.acquire()
        model_inputs = tokenizer(sentence, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(text_label, max_length=3, padding="max_length", truncation=True)
        lock.release()
        labels = labels['input_ids']
        labels = np.where(np.equal(labels, tokenizer.pad_token_id), -100, labels)
        return model_inputs['input_ids'], model_inputs['attention_mask'], labels
```

接下来，我们应用地图功能，如有必要，将数据集洗牌，然后批量数据集：


```python

def get_dataset(dataset, tokenizer, batch_size=None, shuffle=True):
    input_colums=['sentence', 'label', 'text_label']
    output_columns=['input_ids', 'attention_mask', 'labels']
    dataset = dataset.map(MapFunc(input_colums, output_columns),
                          input_colums, output_columns)
    if shuffle:
        dataset = dataset.shuffle(64)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset

batch_size = 8
train_dataset = get_dataset(train_dataset, tokenizer, batch_size=batch_size)
eval_dataset = get_dataset(validation_dataset, tokenizer, batch_size=batch_size, shuffle=False)
```

## 训练模型

现在，我们准备好模型和数据集，让我们为培训做准备。

### 优化器和学习率调度程序

我们设置了用于更新模型参数的优化器，以及在整个培训过程中管理学习率的学习率调度程序。


```python
from mindnlp.modules.optimization import get_linear_schedule_with_warmup
import mindspore.experimental.optim as optim

# Setting up optimizer and learning rate scheduler
optimizer = optim.AdamW(model.trainable_params(), lr=1e-3)

num_epochs = 3 # Number of iterations over the entire training dataset
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=(len(train_dataset) * num_epochs))
```

### 训练步骤

接下来，定义控制每个训练步骤的功能。

定义 `forward_fn` 该执行模型的正向通过以计算损失。

然后通过 `forward_fn` 到 `mindspore.value_and_grad` 创建 `grad_fn` 这同时计算参数更新所需的损失和梯度。

定义 `train_step` 这会根据计算的梯度更新模型的参数，这将在每个训练的每个步骤中调用。


```python
import mindspore
from mindspore import ops

# Forward function to compute the loss
def forward_fn(**batch):
    outputs = model(**batch)
    loss = outputs.loss
    return loss

# Gradient function to compute gradients for optimization
grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())

# Define the training step function
def train_step(**batch):
    loss, grads = grad_fn(**batch)
    optimizer(grads)  # Apply gradients to optimizer for updating model parameters
    return loss
```

### 训练循环

现在一切都准备就绪，让我们实施培训和评估循环，并为培训过程提供。

此过程通过数据集（即多个时期）上的多个迭代来优化模型的参数，并评估其在评估数据集上的性能。


```python
from tqdm import tqdm

# Training loop across epochs
for epoch in range(num_epochs):
    model.set_train(True)
    total_loss = 0
    train_total_size = train_dataset.get_dataset_size()
    # Iterate over each entry in the training dataset
    for step, batch in enumerate(tqdm(train_dataset.create_dict_iterator(), total=train_total_size)):
        loss = train_step(**batch)
        total_loss += loss.float()  # Accumulate loss for monitoring
        lr_scheduler.step()  # Update learning rate based on scheduler

    model.set_train(False)
    eval_loss = 0
    eval_preds = []
    eval_total_size = eval_dataset.get_dataset_size()
    # Iterate over each entry in the evaluation dataset
    for step, batch in enumerate(tqdm(eval_dataset.create_dict_iterator(), total=eval_total_size)):
        with mindspore._no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.float()
        eval_preds.extend(
            tokenizer.batch_decode(ops.argmax(outputs.logits, -1).asnumpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataset)
    eval_ppl = ops.exp(eval_epoch_loss) # Perplexity
    train_epoch_loss = total_loss / len(train_dataset)
    train_ppl = ops.exp(train_epoch_loss) # Perplexity
    print(f "{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

让我们分解训练循环实施并了解关键组成部分：

*模型训练模式

在培训开始之前，该模型通过 `model.set_train(True)`。 在评估之前，模型的特定训练行为是由 `model.set_train(False)`.

*损失和困惑

 `total_loss = 0` 初始化和 `total_loss += loss.float()` 在一个时期内累积每批的总损失。 这种积累对于监视模型的性能至关重要。

印刷消息中报告了平均损失和困惑（PPL），这是语言模型的常见度量。

*学习率调度程序

 `lr_scheduler.step()` 根据预定义的时间表处理每批处理后，调整了学习率。 这对于有效学习至关重要，有助于更快地融合或逃脱当地的最小值。

*评估循环

在评估期间，除了 `model.set_train(False)`，，，，`mindspore._no_grad()` 确保在评估阶段未计算梯度，这可以保存记忆和计算。
这 `tokenizer.batch_decode()` 功能将输出逻辑从模型转换回可读文本。 这对于检查模型的预测和进一步的定性分析很有用。

## 训练后
现在我们已经完成了培训，我们可以评估其性能并保存训练有素的模型以供将来使用。

### 准确性汇总并检查有预测的结果

让我们记录验证数据集上预测的准确性。 准确性是模型预测匹配实际标签的频率的直接度量，从而提供了一个直接的度量标准以反映模型的有效性。


```python
# Initialize counters for correct predictions and total predictions
correct = 0
total = 0

# List to store actual labels for comparison
ground_truth = []

# Compare each predicted label with the true label
for pred, data in zip(eval_preds, validation_dataset.create_dict_iterator(output_numpy=True)):
    true = str(data['text_label'])
    ground_truth.append(true)
    if pred.strip() == true.strip():
        correct += 1
    total += 1

# Calculate the percentage of correct predictions
accuracy = correct / total * 100

# Output the accuracy and sample predictions for review
print(f "{accuracy=} % on the evaluation dataset")
print(f "{eval_preds[:10]=}")
print(f "{ground_truth[:10]=}")
```

### 保存模型
如果您对结果感到满意，则可以如下保存模型：


```python
# Save the model
peft_model_id = f "../../output/{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}" 
model.save_pretrained(peft_model_id)
```

## 使用该模型进行推理

现在，让我们加载保存的模型，并演示如何将其用于对新数据进行预测。

为了加载已通过PEFT训练的模型，我们首先将基本模型加载 `AutoModelForSeq2SeqLM.from_pretrained`。 最重要的是，我们将受过训练的PEFT适配器添加到模型中 `PeftModel.from_pretrained` ：


```python
from mindnlp.transformers import AutoModelForSeq2SeqLM
from mindnlp.peft import PeftModel, PeftConfig

peft_model_id = f "../../output/{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}" 

# Load the model configuration
config = PeftConfig.from_pretrained(peft_model_id)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)

# Load the pretrained adapter
model = PeftModel.from_pretrained(model, peft_model_id)
```

接下来，从验证数据集检索条目，或者自己创建条目。

我们象征着 `'sentence'` 在此条目中，并将其用作模型的输入。 对此表示敬意，并对模型的预测感到好奇。


```python
# Retrieve an entry from the validation dataset.
# example = next(validation_dataset.create_dict_iterator(output_numpy=True)) # Get an example entry from the validation dataset
# print(example['sentence'])
# print(example['text_label'])

# Alternatively, create your own text
example = {'sentence': 'Nvidia Tops $3 Trillion in Market Value, Leapfrogging Apple.'}

inputs = tokenizer(example['sentence'], return_tensors="ms") # Get the tokenized text label
print(inputs)

model.set_train(False)
with mindspore._no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10) # Predict the text label using the trained model
    print(outputs)
    print(tokenizer.batch_decode(outputs.asnumpy(), skip_special_tokens=True)) # Print decoded text label from the prediction
```