# HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation

Source code for paper "HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation"

To run HP-MA
```
python HP-MA.py
```
5 times performance on LastFM:

Runtimes|HR@3  | NDCG@3| HR@5|NDCG@5|HR@10 |NDCG@10
-|-|-|-|-|-|-
1|0.8626|0.7663|0.9239|0.7905|0.9683|0.8050
2|0.8589|0.7674|0.9175|0.7919|0.9641|0.8068
3|0.8631|0.7741|0.9186|0.7963|0.9667|0.8126
4|0.8605|0.7578|0.9149|0.7795|0.9508|0.7924
5|0.8589|0.7647|0.9091|0.7859|0.9503|0.7993

As we can see, the all of the results meet up with the performance in our paper.

# Requirements

* numpy

* scipy

* Tensorflow (1.2.1)

* Keras (2.1.0)

# Reference

@inproceedings{

> author = {Xuesi Li, Wei Liu.},
 
> title = {HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation},
 
> keywords = {Recommender System, High-quality Paths, Multi-view Attention},
 
}
