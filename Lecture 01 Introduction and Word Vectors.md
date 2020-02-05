### 重点笔记

#### 单词的表示

**WordNet**, 一个包含同义词集和上位词(“is a”关系) **synonym sets and hypernyms** 的列表的辞典

同义词：

```
from nltk.corpus import wordnet as wn
poses = { 'n':'noun', 'v':'verb', 's':'adj (s)', 'a':'adj', 'r':'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()],
                    ", ".join([l.name() for l in synset.lemmas()])))
```

![](https://looperxx.github.io/imgs/1560068762906.png)

上位词：

```
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
```

![](https://looperxx.github.io/imgs/1560068729196.png)

存在的问题：

- 作为一个资源很好，但忽略了细微差别，比如"proficient"只在某些上下文和"good"是同义词
- 难以持续更新单词的新含义
- 主观的
- 需要人类劳动来创造和调整
- 无法计算单词相似度



在传统的自然语言处理中，我们把词语看作离散的符号，单词通过one-hot向量表示。所有向量是正交的，没有相似性概念，向量维度过大。

在Distributional semantics中，一个单词的意思是由经常出现在该单词附近的词(上下文)给出的，单词通过一个向量表示，称为word embeddings或者word representations，它们是分布式表示(distributed representation)



#### Word2vec

中心思想：给定大量文本数据，训练每个单词的向量，使得给定中心词c时，上下文词o的概率最大，而这个概率的衡量方式是c和o两个词向量的相似性。

c和o相似性的计算方法是：

$P(o|c)=\frac{exp(u^T_ov_c)}{\sum_{w\in V}exp(u^T_wv_c)}$

每个词有两个向量：作为中心词的向量u和作为上下文词的v

这里用了$softmax(x_i)$函数，放大了最大的概率(max)，仍然为较小的xi赋予了一定概率(soft)

**两个算法**：

CBOW——根据中心词周围的上下文单词来预测该词的词向量

skip-gram——根据中心词预测周围上下文词的概率分布

**两个训练方法**：

negative sampling——通过抽取负样本来定义目标

hierarchical softmax——通过使用一个有效的树结构来计算所有词的概率来定义目标



#### 基于SVD的词嵌入方法

对共现矩阵X应用SVD分解方法得到$X=USV^T$，选择U前k行得到k维的词向量。

方法存在的问题：维度经常发生改变，矩阵稀疏，矩阵维度高，计算复杂度高等





