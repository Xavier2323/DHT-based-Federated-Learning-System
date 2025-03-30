import glob
import sys
import os
import shutil
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

import time
import random
import socket
import threading
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from service import SupernodeService
from service.ttypes import NodeAddress, ResponseCode, NodeInfo

class SupernodeHandler:
    def __init__(self, max_nodes=10):
        self.max_nodes = max_nodes
        self.nodes = {}  # Dictionary to store active nodes
        self.node_joining = False  # Flag to prevent multiple concurrent joins
        self.joining_node_id = None
        self.node_addresses = {}  # Dictionary to map port to ip:port pairs
        self.id_to_port = {}  # Dictionary to map node IDs to ports
        
        # Load node addresses from compute_nodes.txt
        self.load_node_addresses()
        
        print("Supernode initialized with max_nodes =", max_nodes)
    
    def load_node_addresses(self):
        try:
            with open('compute_nodes.txt', 'r') as f:
                for line in f:
                    if line.strip():
                        ip, port = line.strip().split(',')
                        self.node_addresses[int(port)] = ip
            print(f"Loaded {len(self.node_addresses)} node addresses")
        except Exception as e:
            print(f"Error loading node addresses: {e}")
            sys.exit(1)
    
    def request_join(self, port):
        # Check if another node is currently joining
        if self.node_joining:
            print(f"NACK: Node with port {port} attempted to join while another node is joining")
            return ResponseCode.NACK
        
        # Check if port is in the pre-defined addresses
        if port not in self.node_addresses:
            print(f"NACK: Node with port {port} not found in compute_nodes.txt")
            return ResponseCode.NACK
        
        # Check if we've reached the maximum number of nodes
        if len(self.nodes) >= self.max_nodes:
            print(f"NACK: Maximum number of nodes ({self.max_nodes}) reached")
            return ResponseCode.NACK
        
        # Generate a new unique ID
        while True:
            new_id = random.randint(0, self.max_nodes - 1)
            if new_id not in self.nodes:
                break
        
        # Set joining flag and ID
        self.node_joining = True
        self.joining_node_id = new_id
        self.id_to_port[new_id] = port

        print(f"Node with ip:port {self.node_addresses[port]}:{port} is requesting to join with ID {new_id}")
        return new_id
    
    def confirm_join(self, id):
        # Check if this is the node we're expecting
        if not self.node_joining or id != self.joining_node_id:
            print(f"NACK: Unexpected confirm_join from node {id}")
            return ResponseCode.NACK
        
        # Check if the ID is already in use
        if id in self.nodes:
            print(f"NACK: Node with ID {id} already exists")
            return ResponseCode.NACK
        
        # Add the node to the network
        self.nodes[id] = NodeAddress(ip=self.node_addresses[self.id_to_port[id]], port=self.id_to_port[id], id=id)
        
        # Reset joining flag
        self.node_joining = False
        self.joining_node_id = None
        
        print(f"Node with ID {id} confirmed join")
        print(f"Current nodes: {self.nodes}")
        return ResponseCode.SUCCESS
    
    def get_node(self):
        # Return a random node from the network
        if not self.nodes:
            print("Empty network, no nodes available")
            return NodeInfo(NodeAddress(ip="", port=0, id=-1), 0)
        
        random_id = random.choice(list(self.nodes.keys()))
        print(f"Returning random node with ID {random_id}")
        # Return the node address and the current nework size
        return NodeInfo(self.nodes[random_id], len(self.nodes))

if __name__ == '__main__':
    routing_dir = "./routing"
    if os.path.exists(routing_dir):
        shutil.rmtree(routing_dir)
        print(f"Cleaned {routing_dir}")

    # Create a new directory
    os.makedirs(routing_dir)
    print(f"Created {routing_dir}")


    handler = SupernodeHandler()
    processor = SupernodeService.Processor(handler)
    transport = TSocket.TServerSocket(host='0.0.0.0', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    
    print('Starting the supernode server...')
    server.serve()