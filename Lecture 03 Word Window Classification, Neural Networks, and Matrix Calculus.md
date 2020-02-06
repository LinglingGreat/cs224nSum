### 分类问题

情感分类，命名实体识别，买卖决策等

softmax分类器，cross-entropy损失函数(线性分类器)

神经网络分类器，词向量分类的不同(同时学习权重矩阵和词向量，因此参数也更多)，神经网络简介



### 命名实体识别(NER)

找到文本中的"名字"并且进行分类

- 可能的用途
  - 跟踪文档中提到的特定实体（组织、个人、地点、歌曲名、电影名等）
  - 对于问题回答，答案通常是命名实体
  - 许多需要的信息实际上是命名实体之间的关联
  - 同样的技术可以扩展到其他 slot-filling 槽填充 分类
- 通常后面是命名实体链接/规范化到知识库

难点

- 很难计算出实体的边界
  - 第一个实体是 “First National Bank” 还是 “National Bank”
- 很难知道某物是否是一个实体
  - 是一所名为“Future School” 的学校，还是这是一所未来的学校？
- 很难知道未知/新奇实体的类别
  - “Zig Ziglar” ? 一个人
- 实体类是模糊的，依赖于上下文
  - 这里的“Charles Schwab” 是 PER 不是 ORG

在上下文语境中给单词分类，怎么用上下文？将词及其上下文词的向量连接起来

比如如果这个词在上下文中是表示位置，给高分，否则给低分



梯度

神经网络，最大边缘目标函数，反向传播

技巧：梯度检验，正则，Dropout，激活函数，数据预处理(减去均值，标准化，白化Whitening)，参数初始化，学习策略，优化策略(momentum, adaptive)



### 参考资料

[https://looperxx.github.io/CS224n-2019-03-Word%20Window%20Classification,Neural%20Networks,%20and%20Matrix%20Calculus/](https://looperxx.github.io/CS224n-2019-03-Word Window Classification,Neural Networks, and Matrix Calculus/)