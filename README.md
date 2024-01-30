# MindSpore Seq2Seq

## 注意，本教程基于MindSpore 2.1.1 (1到4章节) 与 MindSpore 2.2.10 (5到6章节) 完成，请使用对应版本的MindSpore。


这个存储库包含使用 [MindSpore](https://www.mindspore.cn/) 2.2.10 和 [spaCy](https://spacy.io/) 3.0 以及 Python 3.7.5 实现序列到序列（seq2seq）模型的教程，该教程能够使您更好地利用hugging face社区中的数据集与评估方法与MindSpore结合使用。

**如果您发现任何错误或对解释有异议，请随时 [提交问题](https://github.com/umeiko/mindspore-seq2seq/issues/new)。我欢迎任何反馈，无论是积极的还是消极的！**

## Getting Started

安装Mindspore请参考[MindSpore官网](https://www.mindspore.cn/)

我们还将使用 spaCy 对我们的文本数据进行处理。要安装 spaCy，请按照[此处的说明](https://spacy.io/usage/)进行操作，确保安装英语和德语模型：


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

* 2 - [使用RNN编码器-解码器学习短语表示进行统计机器翻译](https://github.com/umeiko/mindspore-seq2seq/blob/main/2%20-%20%E4%BD%BF%E7%94%A8RNN%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9F%AD%E8%AF%AD%E8%A1%A8%E7%A4%BA%E8%BF%9B%E8%A1%8C%E7%BB%9F%E8%AE%A1%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91.ipynb)

    现在我们已经掌握了基本的工作流程，这个教程将专注于提升我们的结果。借助于从前一个教程中获得的有关MindSporet的知识，我们将介绍第二个模型，该模型有助于解决编码器-解码器模型面临的信息压缩问题。这个模型将基于[使用RNN编码器-解码器学习短语表示进行统计机器翻译](https://arxiv.org/abs/1406.1078)的实现，该实现使用GRU（门控循环单元）。

* 3 - [通过联合学习对齐和翻译实现神经机器翻译](https://github.com/umeiko/mindspore-seq2seq/blob/main/3%20-%20%E9%80%9A%E8%BF%87%E8%81%94%E5%90%88%E5%AD%A6%E4%B9%A0%E5%AF%B9%E9%BD%90%E5%92%8C%E7%BF%BB%E8%AF%91%E5%AE%9E%E7%8E%B0%E7%A5%9E%E7%BB%8F%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91.ipynb)

    接下来，我们将学习有关注意力机制的知识，通过实现[通过联合学习对齐和翻译进行神经机器翻译](https://arxiv.org/abs/1409.0473)。这进一步缓解了信息压缩问题，允许解码器通过创建上下文向量来“回顾”输入句子，这些上下文向量是编码器隐藏状态的加权和。这个加权和的权重是通过注意力机制计算的，其中解码器学会关注输入句子中最相关的单词。

* 4 - [打包填充序列，掩码，推断和 BLEU](https://github.com/umeiko/mindspore-seq2seq/blob/main/4%20-%20%E6%89%93%E5%8C%85%E5%A1%AB%E5%85%85%E5%BA%8F%E5%88%97-%E6%8E%A9%E7%A0%81-%E6%8E%A8%E6%96%AD%E5%92%8C%20BLEU.ipynb)

    在这个笔记本中，我们将通过添加*pack填充序列*和*掩码*来改进先前的模型架构。这两种方法在自然语言处理中常用。Pack填充序列允许我们仅使用我们的RNN处理输入句子的非填充元素。掩码用于强制模型忽略我们不希望它查看的特定元素，例如填充元素上的注意力。这两者共同为我们提供了一些性能提升。我们还介绍了一种非常基本的使用模型进行推理的方法，使我们能够获得对我们想要提供给模型的任何句子的翻译，并且可以查看这些翻译的源序列上的注意值。最后，我们展示了如何从我们的翻译中计算BLEU指标。


* 5 - [序列到序列卷积学习](https://github.com/umeiko/mindspore-seq2seq/blob/main/5%20-%20%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%E5%8D%B7%E7%A7%AF%E5%AD%A6%E4%B9%A0.ipynb)

    最后，我们将摆脱基于RNN的模型，实现一个完全卷积的模型。RNN的一个缺点是它们是顺序的。也就是说，在RNN处理单词之前，所有先前的单词也必须被处理。卷积模型可以完全并行化，这使得它们能够更快地训练。我们将实现[卷积序列到序列](https://arxiv.org/abs/1705.03122)模型，该模型在编码器和解码器中都使用多个卷积层，并在它们之间使用注意机制。


* 6 - [Attention is All you need](https://github.com/umeiko/mindspore-seq2seq/blob/main/6%20-%20Attention%20is%20All%20you%20need.ipynb)

    继续使用非RNN的模型，我们实现了来自[Attention Is All You Need](https://arxiv.org/abs/1706.03762)的Transformer模型。该模型完全基于注意机制，引入了多头注意力。编码器和解码器由多个层组成，每个层都包含多头注意力和位置前馈子层。该模型目前用于许多最先进的序列到序列和迁移学习任务。


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

## 已知问题
- 1~4章节 需要Mindspore-2.1.1以上版本才能够正确运行，这是由于截至编写时 (2024-1-20) 的2.2相关版本`Mindspore-2.2.10`中GRU相关算子无法正确使用，详见[这里](https://gitee.com/mindspore/mindspore/issues/I8VSVM)。 
- 截至编写时(2024-1-20), Mindspore中的`ops.clip_by_norm(grads, max_norm=clip)`函数因为某些原因运行存在问题，所以笔记中相应的部分进行了注释，您可以尝试解除注释观察该部分是否在未来的更新中得到了正确的修复。
- 第五章需要Mindspore-2.2.10以上版本才能够正确运行，这可能是由于Mindspore的某些版本内部view导致梯度丢失引起的，详见[这里](https://gitee.com/mindspore/mindspore/issues/I8WIB4)。 

## Citing

### BibTeX

    @misc{mindsporeseq2seq,
      title={MindSpore Seq2Seq},
      year={2024},
      author={umeiko},
      publisher={GitHub},
      journal={GitHub repository},
      howpublished ={\url{https://github.com/umeiko/mindspore-seq2seq}}
    }
