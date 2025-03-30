# DHT-based Federated Learning System
### Jun-Ting Hsu (hsu00191@umn.edu), Thomas Knickerbocker (knick073@umn.edu)

## Description
A federated learning system with a network of nodes capable of performing a machine learning tasks using a decentralized peer-to-peer Chord network. T compute nodes form a Chord DHT network. Each compute node is to be assigned a fixed set of training data files by a client through the DHT network, and carries out local training on its own data. The client will then collect the results of the local training from each node (without any centralized parameter server). Compute nodes in the system initially connect with the supernode to receive a unique node ID and connection point to an existing node in the Chord network. Using this connection point, new nodes initialize or connect to the network following the Chord protocol. After the network has been populated, ML training data is be distributed across all nodes, locally trained, and aggregated by the client to validate a given dataset.


## Assumptions:
- Supernode should be activate first before executing any compute_node
- Compute nodes belong to one of the ip/port pairs listed in compute_nodes.txt (make sure that ip is registered and can access to compute_nodes.txt)
- For now, let's rely on compute_nodes.txt to share IP/port info between the coordinator and compute nodes, assuming the directory it's in is backed by a shared file system like NFS. If it turns out NFS (or any shared FS) isn't available, we’ll explore alternate communication protocols (e.g., sockets, Redis, etc.)

### Running:
**First time only:**
```bash
thrift --gen py service.thrift
```
1. Start supernode:
run `python3 register_ip.py` to register for supernode
`python3 supernode.py`
2. Start compute nodes:
`python3 register_ip.py`
`python3 compute_node.py <por_no>`
repeat for each node, i.e.:
```bash
python3 compute_node.py 8001
python3 compute_node.py 8002
python3 compute_node.py 8003
python3 compute_node.py 8004 
```
3. Start client:
```bash
python3 client.py ./letters validate_letters.txt
```

