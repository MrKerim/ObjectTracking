import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes for input and output
G.add_node('Input')
G.add_node('Output')

# Add edges between nodes
G.add_edge('Input', 'Convolutional Layer')
G.add_edge('Convolutional Layer', 'Batch Normalization')
G.add_edge('Batch Normalization', 'Activation Function')
G.add_edge('Activation Function', 'Output')

# Draw the graph
pos = nx.spring_layout(G)  # Positions nodes using the Fruchterman-Reingold force-directed algorithm
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
plt.title('Graph Visualization of Convolutional Layer')
plt.show()
