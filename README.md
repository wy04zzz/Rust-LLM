# Rust-LLM

实现大模型的关键算子，Feed-Forward神经网络，Self-Attention层，KV-Cache以及大模型的参数加载，实现了故事续写功能。

项目使用Felladrin/Minueza-32M-UltraChat模型，数据类型为FP32。

### 主要算子

#### SiLU函数

SiLU算子公式如下：
$$
y=silu(x) × y
$$
其中

$$
silu(x) = sigmoid(x) × x
$$

$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

#### RMS Normalization

$$
y_i=\frac{w×x_i}{\sqrt{ \frac{1}{n} \sum_{j} x_{ij}^2 +\epsilon}}
$$

#### 支持广播的矩阵乘

实现BLAS矩阵乘的一个简化版本，公式为：

$$
C=\alpha AB^T + \beta C
$$

### 关于KV-Cache的实现

![cachetree](C:\Users\wy04\Desktop\Rust-LLM\doc\figure\cachetree.png)

当使用前缀树来管理全局缓存时，会话级别的本地KVCacheBlock中的所有token关联的KV在不再发生变化后，会尝试将它们合并到全局KVCache中。如果会话本地的最后一个缓存块与全局前缀树中的KVCache同步操作未能成功，将采取以下措施：

- 将全局KVCache中相同序列的最新缓存同步到会话本地的稳定块的末尾；
- 回收那些同步失败的本地末块中可以重复利用的计算结果。

### 故事续写：执行与效果

```bash
cd story-teller
cargo r --release -- -m ../models/story/
```

![story-telling](C:\Users\wy04\Desktop\Rust-LLM\doc\figure\story-telling.png)