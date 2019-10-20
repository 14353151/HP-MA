# HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation

Source code for paper "HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation"

To run HP-MA
```
python HP-MA.py
```
5 times performance on LastFM:

HR@3  | NDCG@3| HR@5|NDCG@5|HR@10 |NDCG@10
-|-|-|-|-|-
0.8626|0.7663|0.9239|0.7905|0.9683|0.8050
0.8589|0.7674|0.9175|0.7919|0.9641|0.8068
0.8631|0.7741|0.9186|0.7963|0.9667|0.8126
0.8605|0.7578|0.9149|0.7795|0.9508|0.7924
0.8626|0.7663|0.9239|0.7905|0.9683|0.8050

# Requirements

* numpy

* scipy

* Tensorflow (1.2.1)

* Keras (2.1.1)

# Reference

@inproceedings{

> author = {Xuesi Li, Wei Liu.},
 
> title = {HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation},
 
> keywords = {Recommender System, High-quality Paths, Multi-view Attention},
 
}
