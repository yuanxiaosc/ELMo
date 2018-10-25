# ELMo

ELMo: Embeddings from Language Models. Using ELMo as a word embedding in the deep neural network model.



> NAACL 2018最佳论文 [Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)：艾伦人工智能研究所提出新型深度语境化词表征（研究者使用从双向 LSTM 中得到的向量，该 LSTM 是使用成对语言模型（LM）目标在大型文本语料库上训练得到的。因此，该表征叫作 ELMo（Embeddings from Language Models）表征。）。

# ELMo 使用方法

+ [官方版](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
+ [知乎简版](https://zhuanlan.zhihu.com/p/37915351)

## [ELMo TensorFlow Hub 的使用方法](https://tfhub.dev/google/elmo/2)

**Overview**
Computes contextualized word representations using character-based word representations and bidirectional LSTMs, as described in the paper "Deep contextualized word representations" [1].

This modules supports inputs both in the form of raw text strings or tokenized text strings.

The module outputs fixed embeddings at each LSTM layer, a learnable aggregation of the 3 layers, and a fixed mean-pooled vector representation of the input.

The complex architecture achieves state of the art results on several benchmarks. Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups. The use of an accelerator is recommended.

Trainable parameters
The module exposes 4 trainable scalar weights for layer aggregation.

**Example use**
```python
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
["the cat is on the mat", "dogs are in the fog"],
signature="default",
as_dict=True)["elmo"]
```
```python
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
tokens_input = [["the", "cat", "is", "on", "the", "mat"],
["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [6, 5]
embeddings = elmo(
inputs={
"tokens": tokens_input,
"sequence_len": tokens_length
},
signature="tokens",
as_dict=True)["elmo"]
```


**Input**
The module defines two signatures: default, and tokens.

With the default signature, the module takes untokenized sentences as input. The input tensor is a string tensor with shape [batch_size]. The module tokenizes each string by splitting on spaces.

With the tokens signature, the module takes tokenized sentences as input. The input tensor is a string tensor with shape [batch_size, max_length] and an int32 tensor with shape [batch_size] corresponding to the sentence length. The length input is necessary to exclude padding in the case of sentences with varying length.

**Output**
The output dictionary contains:

+ word_emb: the character-based word representations with shape [batch_size, max_length, 512].
+ lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
+ lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
+ elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
+ default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].


# [Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

> Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

> 在本论文中，我们介绍了一种新型深度语境化词表征，可对词使用的复杂特征（如句法和语义）和词使用在语言语境中的变化进行建模（即对多义词进行建模）。我们的词向量是深度双向语言模型（biLM）内部状态的函数，在一个大型文本语料库中预训练而成。本研究表明，这些表征能够被轻易地添加到现有的模型中，并在六个颇具挑战性的 NLP 问题（包括问答、文本蕴涵和情感分析）中显著提高当前最优性能。此外，我们的分析还表明，揭示预训练网络的深层内部状态至关重要，可以允许下游模型综合不同类型的半监督信号。

Comments:	NAACL 2018. Originally posted to openreview 27 Oct 2017. v2 updated for NAACL camera ready
Subjects:	Computation and Language (cs.CL)
Cite as:	arXiv:1802.05365 [cs.CL]
 	(or arXiv:1802.05365v2 [cs.CL] for this version)


# 更多资源
+ [望江人工智库](https://yuanxiaosc.github.io/tags/ELMO/)
+ [Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)
+ [TensorFlow Hub 实现](https://tfhub.dev/google/elmo/2)
+ [allennlp.org - elmo](https://allennlp.org/elmo)
+ [bilm-tf](https://github.com/allenai/bilm-tf)
+ [NAACL 2018最佳论文：艾伦人工智能研究所提出新型深度语境化词表征](https://www.jiqizhixin.com/articles/060704)
+ [把 ELMo 作为 keras 的一个嵌入层使用](https://github.com/strongio/keras-elmo)


