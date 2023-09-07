> Note: The comment might not be correct, please refer to the source code.

# Data sturcture

+ `xcor` and `ycor` are temp storage to store the coordinate of the non-overlap pins; `dcor` is the index in `TreeNode* nodes;`

## StTree (Steiner Tree)
+ `TreeNode* nodes;` first 'deg' nodes are pins.
## treenode
+ `stackAlias`: 
    + when the steiner pin has the same gcell coordinate with some pin(s), its stackAlias will be the index of the first pin(s) in `TreeNode* nodes;`
    + otherwise, the stackAlias will be just the index 
+ `eID`: the edge IDs for all overlapped pins (only applicable for the first index, which is accessedby stackAlias) 
+ `edge[3]`: the edge IDs for the single pin
+ `conCNT`: connection count for all overlapped pins

# Layer assignment
## Net order
+ First sort based on minx of each net
+ Then sort based on total edge wirelength of each net

## Layer assignment
+ 
