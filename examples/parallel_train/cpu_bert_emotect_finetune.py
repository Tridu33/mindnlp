#!/usr/bin/env python
# coding: utf-8



def main():
    import mindspore
    from mindspore.dataset import GeneratorDataset, transforms

    from mindnlp.engine import Trainer


    # prepare dataset
    class SentimentDataset:
        """Sentiment Dataset"""

        def __init__(self, path):
            self.path = path
            self._labels, self._text_a = [], []
            self._load()

        def _load(self):
            with open(self.path, "r", encoding="utf-8") as f:
                dataset = f.read()
            lines = dataset.split("\n")
            for line in lines[1:-1]:
                label, text_a = line.split("\t")
                self._labels.append(int(label))
                self._text_a.append(text_a)

        def __getitem__(self, index):
            return self._labels[index], self._text_a[index]

        def __len__(self):
            return len(self._labels)


    # download dataset
    # get_ipython().system('wget https://baidu-nlp.bj.bcebos.com/emotion_detection-dataset-1.0.0.tar.gz -O emotion_detection.tar.gz')
    # get_ipython().system('tar xvf emotion_detection.tar.gz')


    def process_dataset(source, tokenizer, max_seq_len=64, batch_size=32, shuffle=True):
        is_ascend = mindspore.get_context('device_target') == 'Ascend'

        column_names = ["label", "text_a"]
        
        dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
        # transforms
        type_cast_op = transforms.TypeCast(mindspore.int32)
        def tokenize_and_pad(text):
            if is_ascend:
                tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_len)
            else:
                tokenized = tokenizer(text)
            return tokenized['input_ids'], tokenized['attention_mask']
        # map dataset
        dataset = dataset.map(operations=tokenize_and_pad, input_columns="text_a", output_columns=['input_ids', 'attention_mask'])
        dataset = dataset.map(operations=[type_cast_op], input_columns="label", output_columns='labels')
        # # batch dataset
        if is_ascend:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                            'attention_mask': (None, 0)})

        return dataset


    # 昇腾NPU环境下暂不支持动态Shape，数据预处理部分采用静态Shape处理：

    from mindnlp.transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    tokenizer.pad_token_id

    dataset_train = process_dataset(SentimentDataset("data/train.tsv"), tokenizer)
    dataset_val = process_dataset(SentimentDataset("data/dev.tsv"), tokenizer)
    dataset_test = process_dataset(SentimentDataset("data/test.tsv"), tokenizer, shuffle=False)

    dataset_train.get_col_names()

    type(dataset_train)

    print(next(dataset_train.create_dict_iterator()))

    from mindnlp.transformers import BertForSequenceClassification, BertModel

    # set bert config and define parameters for training
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3)


    from mindnlp.engine import TrainingArguments

    training_args = TrainingArguments(
        output_dir="bert_emotect_finetune_cpu",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3.0
    )


    from mindnlp import evaluate
    import numpy as np

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics
    )
    print("开始 training")
    trainer.train()

    dataset_infer = SentimentDataset("data/infer.tsv")

    def predict(text, label=None):
        label_map = {0: "消极", 1: "中性", 2: "积极"}

        text_tokenized = Tensor([tokenizer(text).input_ids])
        logits = model(text_tokenized)
        predict_label = logits[0].asnumpy().argmax()
        info = f"inputs: '{text}', predict: '{label_map[predict_label]}'"
        if label is not None:
            info += f" , label: '{label_map[label]}'"
        print(info)

    from mindspore import Tensor
    print("开始 predict")

    for label, text in dataset_infer:
        predict(text, label)

    predict("家人们咱就是说一整个无语住了 绝绝子叠buff")

if __name__ == '__main__':
    import sys
    sys.path.append("/home/tridu33/workspace/githubSrc/mindnlp")
    sys.path.append("/home/usersshared/giteeSrc/mindformers")
    # import os
    # 设置环境变量    
    # LD_PRELOAD = "/home/tridu33/.conda/envs/openmind-ms/lib/python3.9/site-packages/faiss_cpu.libs/libgomp-d22c30c5.so.1.0.0"+\
    # ":"+"/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0"+\
    # ":"+"/home/tridu33/.conda/envs/openmind-pt/lib/python3.9/site-packages/torch.libs/libgomp-6e1a1d1b.so.1.0.0"
    # os.environ['LD_PRELOAD']=LD_PRELOAD
    # print(LD_PRELOAD)
    main()
