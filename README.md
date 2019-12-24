[TOC]



## cs224nSum

课程资料：

- Course page: https://web.stanford.edu/class/cs224n
- Video page: https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
- Video page (Chinese): 
  - 可选字幕版：https://www.bilibili.com/video/av61620135 
  - 纯中文字幕版：https://www.bilibili.com/video/av46216519

学习笔记参考：

[CS224n-2019 学习笔记](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)

[斯坦福CS224N深度学习自然语言处理2019冬学习笔记目录](https://zhuanlan.zhihu.com/p/59011576)

参考书：

- Dan Jurafsky and James H. Martin. [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
- Jacob Eisenstein. [Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
- Yoav Goldberg. [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf)
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. [Deep Learning](http://www.deeplearningbook.org/)

神经网络相关的基础:

- Michael A. Nielsen. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- Eugene Charniak. [Introduction to Deep Learning](https://mitpress.mit.edu/books/introduction-deep-learning)



### Lecture 01: Introduction and Word Vectors

1. The course (10 mins)
2. Human language and word meaning (15 mins)
3. Word2vec introduction (15 mins)
4. Word2vec objective function gradients (25 mins)
5. Optimization basics (5 mins)
6. Looking at word vectors (10 mins or less)

**Lecture notes**

- [x] cs224n-2019-lecture01-wordvecs1
  - WordNet, 一个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms** 的列表的辞典
  - 在传统的自然语言处理中，我们把词语看作离散的符号，单词通过one-hot向量表示
  - 在Distributional semantics中，一个单词的意思是由经常出现在该单词附近的词(上下文)给出的，单词通过一个向量表示，称为word embeddings或者word representations，它们是分布式表示(distributed representation)
  - Word2vec的思想
- [x] cs224n-2019-notes01-wordvecs1
  - Natural Language Processing. 
  - Word Vectors. 
  - Singular Value Decomposition(SVD). (对共现计数矩阵进行SVD分解，得到词向量)
  - Word2Vec.
  - Skip-gram. (根据中心词预测上下文)
  - Continuous Bag of Words(CBOW). (根据上下文预测中心词)
  - Negative Sampling. 
  - Hierarchical Softmax. 
- [x] Gensim word vector visualization

**Suggested Readings**

- [x] [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) (该博客分为2个部分，skipgram思想，以及改进训练方法：下采样和负采样)
- [x] [理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078)(上述文章的翻译)
- [x] [Applying word2vec to Recommenders and Advertising](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/) (word2vec用于推荐和广告)
- [x] [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)(没太看懂，之后再看一遍)
- [x] [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)
- [x] [[NLP] 秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)(推荐了一些很好的资料)
- [ ] word2vec Parameter Learning Explained
- [ ] 基于神经网络的词和文档语义向量表示方法研究
- [ ] word2vec中的数学原理详解
- [x] 网易有道word2vec(词向量相关模型，word2vec部分代码解析与tricks)

**Assignment 1：Exploring Word Vectors**

- [x] Count-Based Word Vectors(共现矩阵的搭建, SVD降维, 可视化展示)

- [x] Prediction-Based Word Vectors(Word2Vec, 与SVD的对比, 使用gensim, 同义词,反义词,类比,Bias)

**review**

word2vec的思想、算法步骤分解、代码



### Lecture 02: Word Vectors 2 and Word Senses

1. Finish looking at word vectors and word2vec (12 mins)
2. Optimization basics (8 mins)
3. Can we capture this essence more effectively by counting? (15m)
4. The GloVe model of word vectors (10 min)
5. Evaluating word vectors (15 mins)
6. Word senses (5 mins)

**Lecture notes**

- [x] Gensim word vector visualization

- [x] cs224n-2019-lecture02-wordvecs2
  - 复习word2vec(一个单词的向量是一行；得到的概率分布不区分上下文的相对位置；每个词和and, of等词共同出现的概率都很高)
  - optimization: 梯度下降，随机梯度下降SGD，mini-batch(32或64,减少噪声，提高计算速度)，每次只更新出现的词的向量(特定行)
  - 为什么需要两个向量？——数学上更简单(中心词和上下文词分开考虑),最终是把2个向量平均。也可以每个词只用一个向量。
  - word2vec的两个模型：Skip-grams(SG), Continuous Bag of Words(CBOW), 还有negative sampling技巧，抽样分布技巧(unigram分布的3/4次方)
  - 为什么不直接用共现计数矩阵？随着词语的变多会变得很大；维度很高，需要大量空间存储；后续的分类问题会遇到稀疏问题。解决方法：降维，只存储一些重要信息，固定维度。即做SVD。很少起作用，但在某些领域内被用的比较多，举例：Hacks to X(several used in Rohde et al. 2005)
  - Count based vs. direct prediction
  - Glove-结合两个流派的想法，在神经网络中使用计数矩阵，共现概率的比值可以编码成meaning component
  - 评估单词向量的方法（内在—同义词、类比等，外在—在真实任务中测试，eg命名实体识别）
  - 词语多义性问题-1.聚类该词的所有上下文，得到不同的簇，将该词分解为不同的场景下的词。2.直接加权平均各个场景下的向量，奇迹般地有很好的效果

- [ ] cs224n-2019-notes02-wordvecs2

**Suggested Readings**

1. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
2. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
3. [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)

Additional Readings:

1. [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
2. [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
3. [On the Dimensionality of Word Embedding.](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

Python review session
[[slides](https://web.stanford.edu/class/cs224n/readings/python-review.pdf)]



### Lecture 03: Word Window Classification, Neural Networks, and Matrix Calculus

**Lecture notes**

**Suggested Readings**



### Lecture 04: Backpropagation and Computation Graphs

**Lecture notes**

**Suggested Readings**



### Lecture 05: Linguistic Structure: Dependency Parsing

**Lecture notes**

**Suggested Readings**



### Lecture 06: The probability of a sentence? Recurrent Neural Networks and Language Models

**Lecture notes**

**Suggested Readings**



### Lecture 07: Vanishing Gradients and Fancy RNNs

**Lecture notes**

**Suggested Readings**



### Lecture 08: Machine Translation, Seq2Seq and Attention

**Lecture notes**

**Suggested Readings**



### Lecture 09: Practical Tips for Final Projects

**Lecture notes**

**Suggested Readings**



### Lecture 10: Question Answering and the Default Final Project

**Lecture notes**

**Suggested Readings**



### Lecture 11: ConvNets for NLP

**Lecture notes**

**Suggested Readings**



### Lecture 12: Information from parts of words: Subword Models

**Lecture notes**

**Suggested Readings**



### Lecture 13: Modeling contexts of use: Contextual Representations and Pretraining

**Lecture notes**

**Suggested Readings**



### Lecture 14: Transformers and Self-Attention For Generative Models(guest lecture by Ashish Vaswani and Anna Huang)

**Lecture notes**

**Suggested Readings**



### Lecture 15: Natural Language Generation

**Lecture notes**

**Suggested Readings**



### Lecture 16: Reference in Language and Coreference Resolution

**Lecture notes**

**Suggested Readings**



### Lecture 17: Multitask Learning: A general model for NLP? (guest lecture by Richard Socher)

**Lecture notes**

**Suggested Readings**



### Lecture 18: Constituency Parsing and Tree Recursive Neural Networks

**Lecture notes**

**Suggested Readings**



### Lecture 19: Safety, Bias, and Fairness (guest lecture by Margaret Mitchell)

**Lecture notes**

**Suggested Readings**



### Lecture 20: Future of NLP + Deep Learning

**Lecture notes**

**Suggested Readings**



