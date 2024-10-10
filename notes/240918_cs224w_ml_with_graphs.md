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

tips:
- negative sampling k, prop to its degree, k = 5-20
- SGD

node2vec:
- biased random walk
- bfs(local view) or dfs (global view)
- p (return), q (in-out)

**Summary so far**

- **Core idea:** Embed nodes so that distances in embedding space reflect node similarities in the original network.
- **Different notions of node similarity:**
  - Naïve: similar if 2 nodes are connected
  - Neighborhood overlap (covered in Lecture 2)
  - Random walk approaches (covered today)
- **So what method should I use..?**
- No one method wins in all cases....
  - E.g., node2vec performs better on node classification while alternative methods perform better on link prediction (Goyal and Ferrara, 2017 survey)
- Random walk approaches are generally more efficient
- **In general:** Must choose definition of node similarity that matches your application!


### Lecture 3.3 - Embedding Entire Graphs

graph embedding:
- sum of all node embedding
- virtual node
- anonymous walk embedding: l-step anonymous walk count as graph vector, l = 7 need 122500 random walks
- learn embedding of anonymous walk: next walk can be predicted (like seq2seq predict next token)
  - output: walk embedding (z_i) and graph embedding (z_G)
  - z_G: 1. z_G1^T*z_G2 as kernel; 2. as input of downstream task
- hierarchical embedding: lecture 8

**How to Use Embeddings**

- **How to use embeddings $$z_i$$ of nodes**

  - **Clustering/community detection**: Cluster nodes
  - **Node classification**: Predict label of nodes
  - **Link prediction**: Predict edge $$(i, j)$$ based on:
    - Where we can: concatenate, avg, product, or distance between the embeddings:
      - Concatenate: $$f(z_i, z_j) = g([z_i, z_j])$$
      - Hadamard: $$f(z_i, z_j) = g(z_i * z_j)$$ (per coordinate product)
      - Sum/Avg: $$f(z_i, z_j) = g(z_i + z_j)$$
      - Distance: $$f(z_i, z_j) = g(||z_i - z_j||_2)$$
    - hadamard, sum/avg good for undirected graph
  - **Graph classification**: graph embeddings

Today's Summary

We discussed **graph representation learning**, a way to learn **node and graph embeddings** for downstream tasks, **without feature engineering**.

- **Encoder-decoder framework:**
  - Encoder: embedding lookup
  - Decoder: predict score based on embedding to match node similarity

- **Node similarity measure:** (biased) random walk
  - Examples: DeepWalk, Node2Vec

- **Extension to Graph embedding:** Node embedding aggregation and Anonymous Walk Embeddings


## Graph as Matrix: PageRank, Random Walks and Embeddings
### Lecture 4.1 - PageRank
graph as matrix

- Treating a graph as a matrix allows us to:
  - Determine node importance via random walk (PageRank)
  - Obtain node embeddings via matrix factorization (MF)
  - View other node embeddings (e.g. Node2Vec) as MF

- Random walk, matrix factorization and node embeddings are closely related!

PageRank
- web as graph

We will cover the following **Link Analysis approaches** to compute the **importance** of nodes in a graph:
- PageRank
- Personalized PageRank (PPR)
- Random Walk with Restarts

PageRank: The "Flow" Model
- page with more in-links is more important, in-link from important page count more (RECURSIVE)
- stochastic adjacency matrix M
  - *M* is a **column stochastic matrix**
  - Columns sum to 1
- rank vector r: r_i is importance score of page_i
- the flow equation: r = M * r

pagerank -> MF <- random walk

r is stationary distribution for the random walk,
like eigenvector of a matrix

flow equation <-> eigenvector formulation

PageRank = r is principal eigenvector of M with eigenvalue 1
Power iteration

**PageRank: Summary**

- **PageRank:**
  - Measures importance of nodes in a graph using the link structure of the web
  - Models a random web surfer using the stochastic adjacency matrix $$M$$
  - PageRank solves $$r = Mr$$ where $$r$$ can be viewed as both the principal eigenvector of $$M$$ and as the stationary distribution of a random walk over the graph


### Lecture 4.2 - PageRank: How to Solve?
initial -> repeat until convergence (power iteration, 50 iteration is enough)

2 problems:
- dead ends: 1.0 teleport prob
- spider traps: 1 - b to jump to random page b = 0.8, 0.9


PageRank equation [Brin-Page, 98]

$$r_j = \sum_{i \rightarrow j} \beta \frac{r_i}{d_i} + (1 - \beta) \frac{1}{N}$$



- The Google Matrix G:


  $$P = \beta M + (1 - \beta) \left[ \frac{1}{N} \right]_{N \times N}
  $$

  [1/N]_{N \times N}\) ... \(N\) by \(N\) matrix where all entries are \(1/N\)

- We have a recursive problem: $$ r = G \cdot r $$



**Solving PageRank: Summary**

- **PageRank** solves for \( r = Gr \) and can be efficiently computed by **power iteration of the stochastic adjacency matrix (G)**
- Adding random uniform teleportation solves issues of **dead-ends and spider-traps**



### Lecture 4.3 - Random Walk with Restarts
recommend problem

**PageRank:**
- Ranks nodes by "importance"
- Teleports with uniform probability to any node in the network

**Personalized PageRank:**
- Ranks proximity of nodes to the teleport nodes $$S$$

**Proximity on graphs:**
- **Q:** What is most related item to **Item Q?**
- **Random Walks with Restarts**
  - Teleport back to the starting node: $$S = \{Q\}$$

PR, PPR, random walk with restart, only diff of teleport:
- any
- a set of nodes
- a node

some node in query nodes get more importance if it was walked more times.


### Lecture 4.4 - Matrix Factorization and Node Embeddings

02-nodeemb.pdf: p61-p70

min A - Z^T * Z
factor matrix A to Z
Inner product decoder with node similarity defined by edge connectivity is equivalent to matrix factorization of 𝐴.

DeepWalk can also be explained as MF

limitation 1-3 of DeepWalk/node2vec

**Summary**

- **PageRank**
  - Measures importance of nodes in graph
  - Can be efficiently computed by power iteration of adjacency matrix

- **Personalized PageRank (PPR)**
  - Measures importance of nodes with respect to a particular node or set of nodes
  - Can be efficiently computed by random walk

- **Node embeddings** based on random walks can be expressed as matrix factorization

- **Viewing graphs as matrices plays a key role in all above algorithms!**

### Lecture 5.1 - Message passing and Node Classification

three old techniques which are helpful to new GNN:
- relational classification
- iterative classification
- belief propagation

[homophily](https://youtu.be/6g9vtxUmfwM?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&t=342): char -> conn
[influence](): conn -> char

collective classification:
- local
- relational
- collective inference



### Lecture 5.2 - Relational and Iterative Classification

- probabilistic relational classifier: a little similar to KNN
  - only label, not use node feature
- iterative
  - both of label and feature





## reference
- [video](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)
- [pyg](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
- [pyg colabs](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)