# RepAlloc
Code for the optimization of quantum repeater placement with the use of existing fiber infrastructure.
Our paper with background information can be found [here](https://arxiv.org/abs/2005.14715).

The linear programming solver used is [CPLEX](https://www.ibm.com/analytics/cplex-optimizer), which need to be installed in order to use the Python API and run the code. Free academic research licenses are available. Other requirements are NetworkX and Numpy.

The `Colt.gml` and `Surfnet.gml` files are both retreived from the [Topology Zoo](http://www.topology-zoo.org/), while `SurfnetFiberdata.gml` was provided to us by [Surfnet](https://www.surf.nl/) and contains real fiber data of a part of the internet infrastructure of the Netherlands.
