# Stanford CS224W: ML with Graphs | 2021

### Lecture 2.1 - Traditional Feature-based Methods: Node
- 传统图机器学习就是基于特征的方法，提取特征，然后用经典机器学习的方法训练和推理

Node 的特征：
- degree
- node centrality (eigen vector, #shortest path)
- cluster coefficient
- graphlet degree vector (GDV)

### Lecture 2.2 - Traditional Feature-based Methods: Link
link prediction as a task:
- links missing at random
- links over time

link prediction via proximity:
- [相似地址发现](240906_find_similar_erc20_usdt_addr_v001b.ipynb) 用的是这个方案

The text in the image reads:

**Link-Level Features: Overview**
- Distance-based feature (shortest-path)
- Local neighborhood overlap (Jaccard's coefficient 这个可以在查找币安相关地址上试试)
- Global neighborhood overlap （Katz Index）


### Lecture 2.3 - Traditional Feature-based Methods: Graph
- kernel methods (度量两个图的相似度)
  - graph kernels: graphlet kernel, Weisfeiler-Lehman kernel
  - node key idea: graph feature vector, like BOW (bag of words), such as bag-of-node, bag-of-node-degree
  - graphlet key idea: graphlet count vector, count graphlet 计算量太大 -> Weisfeiler-Lehman kernel
  - Weisfeiler-Lehman: color refinement; The WL kernel value is computed by the inner product of the color count vectors

**Graphlet Kernel**
- Graph is represented as *Bag-of-graphlets*
- *Computationally expensive*

**Weisfeiler-Lehman Kernel**
- Apply *K-step color refinement algorithm* to enrich node colors
  - Different colors capture different *K-hop neighborhood* structures
- Graph is represented as *Bag-of-colors*
- *Computationally efficient*
- Closely related to Graph Neural Networks (as we will see!)




### Lecture 2 Summary
**Traditional ML Pipeline**
- Hand-crafted feature + ML model

**Hand-crafted features for graph data**

- **Node-level:**
  - Node degree, centrality, clustering coefficient, graphlets

- **Link-level:**
  - Distance-based feature
  - local/global neighborhood overlap

- **Graph-level:**
  - Graphlet kernel, WL kernel

### Lecture 3.1 - Node Embeddings
- graph representation learning (no need feature engineering) -> feature representing (embedding)

embedding 可以用来
- Node classification
- Link prediction
- Graph classification
- Anomalous node detection


Encoder, Decoder

embedding similarity -> dot product of 2 vectors

How to choose node similarity measure? (by random walk (unsupervised/self-supervised))

### Lecture 3.2-Random Walk Approaches for Node Embeddings
预测 u 到达 v 的概率。是否能到达要看 random walk (u->v) 能不能走到

short fixed-length random walks -> multiset of visited nodes -> maximum likelihood of neighbors





## reference
- [video](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)
- [pyg](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
- [pyg colabs](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)