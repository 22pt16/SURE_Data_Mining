SURE-Session-Recommendation
Dataset: MovieLens 100K
We replace Apriori with FP-Growth to improve scalability and eliminate candidate generation overhead. FP-Growth builds a compact FP-tree structure, reducing multiple database scans and improving rule mining efficiency.

Apriori repeatedly generates and prunes candidate itemsets, leading to combinatorial explosion. FP-Growth compresses transactions into an FP-tree structure and avoids candidate generation, reducing time complexity.