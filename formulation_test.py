import cplex
import networkx as nx
import matplotlib.pyplot as plt

def create_graph():
    G = nx.petersen_graph()
    plt.subplots()
    nx.draw(G)
    plt.show()

if __name__ == "__main__":
    create_graph()


