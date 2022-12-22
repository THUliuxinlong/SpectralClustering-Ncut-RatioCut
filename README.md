# 矩阵大作业

[TOC]

## 最小化 Ncut 和 RatioCut 相比于最小化 cut 的优势

mincut相对容易，但是在许多情况下，只是将⼀个单独的顶点与图中的其余部分分开。Ncut和Ratiocut这两个目标函数试图实现的是集群是“平衡的”， 分别通过顶点数或边权重来衡量。但是引入平衡条件会使之前的问题变成NP难的。  

## 理论推导 

分别将最小化 Ncut 和 RatioCut 的优化目标等价转化为和矩阵 W、 D 等有关的形式，并进行适当的松弛以得到近似解。  

[(7条消息) 谱聚类（Spectral Clustering）1——算法原理_大笨牛@的博客-CSDN博客_谱聚类伪代码](https://blog.csdn.net/Graduate2015/article/details/116738776?ops_request_misc=&request_id=&biz_id=102&utm_term=谱聚类&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-116738776.142^v67^wechat_v2,201^v3^control_1,213^v2^t3_esquery_v3&spm=1018.2226.3001.4187)

[(7条消息) 小白入门谱聚类算法原理与实现_Drone_xjw的博客-CSDN博客_谱聚类算法原理与实现](https://blog.csdn.net/xjw9602/article/details/103489808?ops_request_misc=&request_id=&biz_id=102&utm_term=相似度谱聚类图像分割&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-103489808.142^v67^wechat_v2,201^v3^control_1,213^v2^t3_esquery_v3&spm=1018.2226.3001.4187)

[谱聚类（spectral clustering）原理总结 - 刘建平Pinard - 博客园 (cnblogs.com)](https://www.cnblogs.com/pinard/p/6221564.html)

[(5条消息) 谱聚类（spectral clustering)及其实现详解_杨铖的博客-CSDN博客_spectral clustering](https://blog.csdn.net/yc_1993/article/details/52997074?ops_request_misc=&request_id=&biz_id=102&utm_term=谱聚类&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-52997074.142^v67^wechat_v2,201^v3^control_1,213^v2^t3_esquery_v3&spm=1018.2226.3001.4187)

## 图像分割

1.基于Ncut的结果（两种Normalized spectral clustering）：

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-22-231632.png" alt="image-20221203112151348" style="zoom:80%;" />

``` python
silhouette_score: 0.516969    CH_score: 52118.063906    DBI: 0.541529
time_Ncut_1: 116.464843 s
```

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-22-231650.png" alt="image-20221203112231075" style="zoom:80%;" />

```python
silhouette_score: 0.505002    CH_score: 47887.103704    DBI: 0.542372
time_Ncut_2: 114.181966 s
```

2.基于Ratio的结果（Unnormalized spectral clustering）：

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-22-231634.png" alt="image-20221203112247378" style="zoom:80%;" />

```python
silhouette_score: 0.572918    CH_score: 43084.603199    DBI: 0.527152
time_RatioCut: 131.537393 s
```

3.Kmeans的结果：

<img src="https://github.com/THUliuxinlong/PicGo/raw/main/img/2022-12-22-231636.png" alt="image-20221203112301899" style="zoom:80%;" />

```python
silhouette_score: 0.604372    CH_score: 55283.346491    DBI: 0.481775
time_Kmeans: 1.935717 s
```

实际上，谱聚类要做的事情其实就是将高维度的数据，以特征向量的形式简洁表达，属于一种降维的过程。本来高维度用k-means不好分的点，在经过线性变换以及降维之后，十分容易求解。
