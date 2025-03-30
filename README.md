# DHT-based Federated Learning System
## Jun-Ting Hsu (hsu00191@umn.edu), Thomas Knickerbocker (knick073@umn.edu)

## Description
A federated learning system with a network of nodes capable of performing a machine learning tasks using a decentralized peer-to-peer Chord network. T compute nodes form a Chord DHT network. Each compute node is be assigned a fixed set of training data files by a client through the DHT network, and carries out local training on its own data. The client will then collect the results of the local training from each node (without any centralized parameter server). Compute nodes in the system initially connect with the supernode to receive a unique node ID and connection point to an existing node in the Chord network. Using this connection point, new nodes initialize or connect to the network following the Chord protocol. After the network has been populated, ML training data is be distributed across all nodes, locally trained, and aggregated by the client to validate a given dataset.


## Assumptions:
- Supernode running on port 9090 of csel-kh1250-06.cselabs.umn.edu (can be manually set via 'main' in client, compute_node, and supernode.py)
- Compute nodes belong to one of the por/address pairs listed in compute_nodes.txt (such as ports 8091-8094 of csel-kh1250-06.cselabs.umn.edu)

### Running:
**First time only:**
```bash
thrift --gen py service.thrift
```
1. Start supernode:
`python3 supernode.py`
2. Start compute nodes:
`python3 compute_node.py <por_no>`
repeat for each node, i.e.:
```bash
python3 compute_node.py 8091
python3 compute_node.py 8092
python3 compute_node.py 8093
python3 compute_node.py 8094 
```
3. Start client:
```bash
python3 client.py ./letters validate_letters.txt
```

