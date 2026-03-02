SURE-Session-Recommendation
Dataset: MovieLens 100K
We replace Apriori with FP-Growth to improve scalability and eliminate candidate generation overhead. FP-Growth builds a compact FP-tree structure, reducing multiple database scans and improving rule mining efficiency.

Apriori repeatedly generates and prunes candidate itemsets, leading to combinatorial explosion. FP-Growth compresses transactions into an FP-tree structure and avoids candidate generation, reducing time complexity.

We implemented the core SURE framework for uninteresting item removal using association rule mining. Due to time constraints, instead of the transformer-based SASRec model, we used a lightweight bigram sequential model to evaluate the impact of filtering. This allows us to isolate the effect of association rule filtering on recommendation performance.