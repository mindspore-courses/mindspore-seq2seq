# MindSpore Seq2Seq

## 注意，本教程基于MindSpore 2.1.1 版本完成，请使用对应版本的MindSpore。


这个存储库包含使用 [MindSpore](https://www.mindspore.cn/) 2.2.10 和 [spaCy](https://spacy.io/) 3.0 以及 Python 3.7.5 实现序列到序列（seq2seq）模型的教程。

**如果您发现任何错误或对解释有异议，请随时 [提交问题](https://github.com/umeiko/mindspore-seq2seq/issues/new)。我欢迎任何反馈，无论是积极的还是消极的！**

## Getting Started

安装Mindspore请参考[MindSpore官网](https://www.mindspore.cn/)

我们还将使用 spaCy 对我们的数据进行标记化。要安装 spaCy，请按照[此处的说明](https://spacy.io/usage/)进行操作，确保安装英语和德语模型：


``` bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```
国内环境如无法安装，请尝试使用`direct`模式下载数据集。

```bash
python -m spacy download en_core_web_sm-3.7.0 --direct
python -m spacy download de_core_news_sm-3.7.0 --direct
```

## Tutorials

* 1 - [序列到序列学习与神经网络](https://github.com/umeiko/mindspore-seq2seq/blob/main/1%20-%20%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.ipynb) 

    第一个教程涵盖了使用 MindSpore  的 seq2seq 项目的工作流程。我们将介绍使用编码器-解码器模型的 seq2seq 网络的基础知识，如何在 MindSpore 中实现这些模型，以及如何使用 Vocab 处理文本方面的繁重工作。模型本身将基于[使用神经网络进行序列到序列学习](https://arxiv.org/abs/1409.3215)的实现，该实现使用多层 LSTM。


## References

这里是我在制作这些教程时参考的一些工作。但是请注意，其中一些可能已经过时。
- https://github.com/bentrevett/pytorch-seq2seq
- https://github.com/spro/practical-pytorch
- https://github.com/keon/seq2seq
- https://github.com/pengshuang/CNN-Seq2Seq
- https://github.com/pytorch/fairseq
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
