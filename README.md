# HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation

Source code for paper "HP-MA: Multi-view Attention Neural Network with High-quality Paths for Recommendation"

To run HP-MA
```
python HP-MA.py
```
performance on MovieLens:

method|HR@3  | NDCG@3| HR@5|NDCG@5|HR@10 |NDCG@10
-|-|-|-|-|-|-
MF|0.2071|0.1465|0.2787|0.1878|0.3923|0.2013
NeuMF|0.2452|0.1949|0.3191|0.2252|0.4225|0.2587
MCRec|0.2685|0.1998|0.3446|0.2313|0.4317|0.2597
NeuACF|0.2412|0.1863|0.3379|0.2259|0.4767|0.2705
HP-MA|0.2801|0.2240|0.3614|0.2539|0.4341|0.2762
HP-MA-mix|0.2948|0.2288|0.3637|0.2573|0.4454|0.2810

1) HP-MA outperforms all the other methods in most cases.
2) HP-MA-mix is additional model that derived from HP-MA. Unlike HP-MA which only use 3-hops path, the HP-MA-mix leverages 2-hops, 3-hops and 4-hops path to improve the recommendation performance. As we can see, it peforms slightly better than the orginal method.

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
