This is to help understand the code structure of CUGR

# Data structure

## GrNet: *Nets object storing all nets*
+ vector<vector<GrPoint>> pinAccessBoxes # *pinAccessBoxes[i][j] is the j_th gcell coordinates of i_th pin
+ std::unordered_set<GrPoint> ovlpPoints # *ovlpPoints is the set storing all gcells that contain more than one pin from the same net*

## InitRoute: The Route Class for one single net
+ routeNodes: vector of RouteNode(gcell)

## RouteNode (in `initRoute.h`): *Gcell object storing all gcells*
+ pinIdxs store all pin index within the node (not steiner point)
+ toConnect store all gcells that are connected to the node (might be steiner point!)
```
One application example, when we execute edge shift, we need to pick two connected gcells where neither contains a pin (therefore, steiner point), but whose degree is 3:
```
> fromNode.pinIdxs.size() == 0 && toNode.pinIdxs.size() == 0 && fromNode.degree() == 3 && toNode.degree() == 3

## RTrees data structure that stores all boxes (like blockage, i.e., `fixedMetals`)
+ `fixedMetals[i]` stores all boxes of layer i
+ `fixedMetals[i][j]` stores the j_th box of layer i with tuple format (boost::box, index)
# How CUGR runs FLUTE

The functions are hierarchically organized as follows:
+ runs() # *main function*
    + routeApprx() # *main iterative algorithm without post-processing*
        + fluteAllAndRoute()
            + plan_fluteOnly()
                + Here, they use `router` to run FLUTE, which works on a SINGLE net 
                + **runFlute()**
                    ```
                    1. get net center by avg all pin for each net, and then avg all nets
                    2. get pin center: 3 kinds of accessibility because we have TWO edges (not consider z-dim) for each gcell (0) totally vio-free because we have TWO edges for each gcell (1) one side vio-free (2) no side vio-free 
                    3.  because we have TWO edges for each gcell
                    ```
                    + pinCenters: 
                + For each gcell (RouteNode in the repository), update the edge usage
            + edge_shift2d() 
        + route()
            + For each gcell (RouteNode in the repository), update the edge usage

# How CUGR works

## InitRoute
+ First Run FLUTE
+ Then, fine tune routing tree by edge shift multiple times. 
    + Each time, pick two steiner points and check whether the edge can be shifted
    + The cost here is calculated by `getCost2D`, see Sec How CUGR calculates costs(2D)
+ In initial route, each L-shape candidate has 1/2 usage
### How CUGR calculates costs(2D)
+ fixed_usage = fixedMetalMap2D[dir (direction)][gridline (row/col)][cp (col/row)];
+ wire_usage = wireUsageMap2D[dir][gridline][cp]
+ cap = capacityMap2D[dir][gridline][cp];
+ cost += 1 / (1.0 + exp(-1 * grDatabase.getLogisticSlope() * (demand - cap)));
+ for logisticSlope, update each iteration: grDatabase.setLogisticSlope(db::setting.initLogisticSlope * pow(2, iter));

## 3D pattern routing
+ `getRoutingOrder`: for each net, starts from the pin whose degree is 1
+ First schedule and divide nets into batches
+ Run DP-based 3D pattern routing, ONLY L-shape is considered
    + The cost is calculated by `fromNode.exitCosts[from_layer] + from_edge_cost + bend_via_cost + to_edge_cost;`
        + `from_edge_cost -> getBufferedWireCost -> getWireCost -> getWireDistCost(edge) + getWireShortCost(edge);`
            + `getWireDistCost`: length in real unit 
            + `getWireShortCost`: avg length per track / (1 + exp(-logisticSlope * (demand - capacity))); demand = fixed tracks + used tracks + 1 + sqrt((u_via * v_via)/2)

## Maze routing iteratively
+ How to determine whether the net needs to be maze routed?: `getNetsToRoute::getNumVio > 0`
    + getNumVio(edge) = wire_usage + $\sqrt{(u_{via} + v_{via})/2}$ - capacity
        + via cost sums top via and bottom via, both of them can be obtained by accessing `routedViaMap`
    + maze routing nets are those with `getNumVio > 0`
+ The order is determined by half-perimeter of the net bounding box

# How CUGR2.0 works

## Data structure
+ Each pin has a series of `AccessPoints` based on its size and location; Then, the best access point `selectedAccessPoints` is selected based on the accessibity (capacity >= 1 for connected edges, q: why not capacity - demand >= 1) and distacne to the net center. When multiple pin is in the same 2-D gcell, the access point's layer Intervel will be updated.

`selectedAccessPoints.key` = selected pin location x* ymax + y
`selectedAccessPoints.value`.first = Point object
`selectedAccessPoints.value`.second = layer Intervel

## How to calculate costs?

### Pattern Routing
+ Via cost for **one** layer is (`GridGraph::getViaCost`): `weight_via_number = 4` + wire_cost of two edges  * (layerMinLengths/(lowerEdgeLength + higherEdgeLength) * parameters.via_multiplier = 2) (One gcell node for a via has two egdes, named lowerEdge and higherEdge); Wire cost has the same calculation methods in Maze routing
### Maze Routing
+ Each turning point has a via cost by `weight_via_number = 4`
+ Wire cost (`wireCostView`) = edge physical length * (`unit_weight_wire_length = 0.5 / m2_pitch (pitch width at m2 layer)` + `unitLengthShortCost` * k), 
    +  k = 1 if capacity < 1; k = logistic(cap - demand, `maze_logistic_slope` = 0.5)
    + `unitLengthShortCost` = min(`unit_area_short_cost` * layer width (by `getWidth`)) among all layers, where `unit_area_short_cost` = (`weight_short_area = 500`)/(m2_pitch^2)


## Pattern Routing
1. `constructSteinerTree`
2. `constructRoutingDAG`
    + Each node is a `std::shared_ptr<PatternRoutingNode>`, and the topology is stored by `.children` vector.

## Maze Routing
    // edges[u][0] is the upper node (x+1, or y+1) connecting u 
    // edges[u][1] is the lower node (x-1, or y-1) connecting u
    // edges[u][2] is the diff-layer node connecting u
    // costs[u][i] is corresponding cost of edges[u][i]
In maze routing, each node is divided into two sub-node (ranging from (0,x*y) to (x*y+1, 2*x*y)). Each sub-node can only connects either horizontal or vertical edges

## How to do patching (for CUGR2.0)
1. Pin access patch, for pins at the bottom layer. add 3*3 (3 is defined by `pin_patch_padding`) gcells around the pin with height 3.
2. Wire segment patches, if the wire segment is not sparse, i.e., aviable tracks < 2 ( 2 is defined by `wire_patch_threshold`), also patch above and bottom layer unless the above/bottom layer is not sparse.
If the prev cell is not patched. the `wire_patch_threshold` will be increased by 1.2 per cell (1.2 is defined by `wire_patch_inflation_rate`)

## Parameters

## What we changed CUGR2 so far
+ (_updateAP) In SelectAccessPoints. We change the accessbility from capacity to capacity - demand
    + This seems to be trivial, it improves GR results, but very trivial, probably most pins only occupy one gcell (so no need to select)

+ (_2Viacost) In Maze routing, double via cost: this improves GR results for all cases

+ (_NewSort) With the two techniques AND with new sort: It has significant influence, but necessarily always better than original sort

## How via influences demand in CUGR2
via has two gcells (for two layers) each gcell has two edge, one is lowerEdge, the other is higherEdge. 
The demand = layerMinLengths / (low edge length + high edge length) * (via_multiplier = 2)
> I guess layerMinLength is the minimum allowed length in this layer due to fabirication constraint. So, via_multiplier is a mathematical estimate of how long the via will occupy the edge.

In Our 2D case. We need to consider two types of "vias":
1. turning points in 2-pin net. The number of layers could be estimated by \sqrt{total layer number}. but we could explore a congested-based estimate using statistics
2. Pins and steiner points. The number of layers could be directly high_layer - min_layer + 1, (this might be 3 for most cases, I need to print them for verification, if it is 3 for most case, maybe \sqrt{total layer number} is a good estimate)

Then, the demand could be calculated by layer_number*(via_multiplier = 2) * layerMinLengths / (low edge length + high edge length)

## How CUGR2 handles 1. Single Pin which covers multiple layers 2. Multiple pins which are in the same gcell node

When translate from DP results (`PatternRoutingNode`) to real routing results (`GRTreeNode`).
Even the node only occupies one gcell, it (`getRoutingTree`) would create child nodes to connects from low layer to high layer.
And then, in `commitVia`, these influence are committed.