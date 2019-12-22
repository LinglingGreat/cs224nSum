##cs224nSum

课程资料：

- Course page: https://web.stanford.edu/class/cs224n
- Video page: https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
- Video page (Chinese): 
  - 可选字幕版：https://www.bilibili.com/video/av61620135 
  - 纯中文字幕版：https://www.bilibili.com/video/av46216519

学习笔记参考：

[CS224n-2019 学习笔记](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)



### Lecture 01: Introduction and Word Vectors

1. The course (10 mins)
2. Human language and word meaning (15 mins)
3. Word2vec introduction (15 mins)
4. Word2vec objective function gradients (25 mins)
5. Optimization basics (5 mins)
6. Looking at word vectors (10 mins or less)

**Lecture notes**

- [x] cs224n-2019-lecture01-wordvecs1.pdf
  - WordNet, 一个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms** 的列表的辞典
  - 在传统的自然语言处理中，我们把词语看作离散的符号，单词通过one-hot向量表示
  - 在Distributional semantics中，一个单词的意思是由经常出现在该单词附近的词(上下文)给出的，单词通过一个向量表示，称为word embeddings或者word representations，它们是分布式表示(distributed representation)
  - Word2vec的思想
- [x] cs224n-2019-notes01-wordvecs1.pdf
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

**Assignment 1：Exploring Word Vectors**

- [x] Count-Based Word Vectors(共现矩阵的搭建, SVD降维, 可视化展示)

- [x] Prediction-Based Word Vectors(Word2Vec, 与SVD的对比, 使用gensim, 同义词,反义词,类比,Bias)

review

word2vec的思想、算法步骤分解、代码



### Lecture 02: Word Vectors 2 and Word Senses

**Suggested Readings**



### Lecture 03: Word Window Classification, Neural Networks, and Matrix Calculus

**Suggested Readings**



### Lecture 04: Backpropagation and Computation Graphs

**Suggested Readings**



### Lecture 05: Linguistic Structure: Dependency Parsing

**Suggested Readings**



### Lecture 06: The probability of a sentence? Recurrent Neural Networks and Language Models

**Suggested Readings**



### Lecture 07: Vanishing Gradients and Fancy RNNs

**Suggested Readings**



### Lecture 08: Machine Translation, Seq2Seq and Attention

**Suggested Readings**



### Lecture 09: Practical Tips for Final Projects

**Suggested Readings**



### Lecture 10: Question Answering and the Default Final Project

**Suggested Readings**



### Lecture 11: ConvNets for NLP

**Suggested Readings**



### Lecture 12: Information from parts of words: Subword Models

**Suggested Readings**



### Lecture 13: Modeling contexts of use: Contextual Representations and Pretraining

**Suggested Readings**



### Lecture 14: Transformers and Self-Attention For Generative Models(guest lecture by Ashish Vaswani and Anna Huang)

**Suggested Readings**



### Lecture 15: Natural Language Generation

**Suggested Readings**



### Lecture 16: Reference in Language and Coreference Resolution

**Suggested Readings**



### Lecture 17: Multitask Learning: A general model for NLP? (guest lecture by Richard Socher)

**Suggested Readings**



### Lecture 18: Constituency Parsing and Tree Recursive Neural Networks

**Suggested Readings**



### Lecture 19: Safety, Bias, and Fairness (guest lecture by Margaret Mitchell)

**Suggested Readings**



### Lecture 20: Future of NLP + Deep Learning

**Suggested Readings**



