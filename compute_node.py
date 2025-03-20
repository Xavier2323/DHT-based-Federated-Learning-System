import glob
import sys
import os
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

import time
import socket
import math
import hashlib
import threading
import numpy as np
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from service import ComputeNodeService, SupernodeService
from service.ttypes import NodeAddress, ModelWeights, ResponseCode

from ML.ML import mlp

class ComputeNodeHandler:
    def __init__(self, port, supernode_host='0.0.0.0', supernode_port=9090):
        self.port = port
        # Create a socket to determine the external IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            self.ip = sock.getsockname()[0]
        finally:
            sock.close()

        self.id = None
        self.predecessor = None
        self.successor = None
        self.finger_table = []
        self.max_nodes = 10  # Assuming max 10 nodes as per requirements
        self.supernode_host = supernode_host
        self.supernode_port = supernode_port
        self.models = {}  # Dictionary to store trained models
        self.training_files = {}  # Dictionary to track training status
        self.finger_table_lock = threading.Lock()
        self.training_lock = threading.Lock()

        # Join the network
        self.join_network()
        
        # print node info
        print(self.print_info())
    
    def join_network(self):
        # Connect to the supernode
        transport = TSocket.TSocket(self.supernode_host, self.supernode_port)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        supernode = SupernodeService.Client(protocol)
        
        try:
            transport.open()
            
            # Request to join the network
            response = supernode.request_join(self.port)
            if isinstance(response, int):
                self.id = response
            else:
                print(f"Failed to join network: {response}")
                transport.close()
                sys.exit(1)
            
            # Get a node in the network
            node_address = supernode.get_node()
            print(f"Node address: {node_address}")
            
            # If this is the first node in the network
            if node_address.id == -1:
                print("First node in the network")
                self.predecessor = NodeAddress(ip=self.ip, port=self.port, id=self.id)
                self.successor = NodeAddress(ip=self.ip, port=self.port, id=self.id)
                self.init_finger_table()
                
                # Confirm join with the supernode
                supernode.confirm_join(self.id)
            else:
                # Connect to the existing node
                node_transport = TSocket.TSocket(node_address.ip, node_address.port)
                node_transport = TTransport.TBufferedTransport(node_transport)
                node_protocol = TBinaryProtocol.TBinaryProtocol(node_transport)
                node = ComputeNodeService.Client(node_protocol)
                
                try:
                    node_transport.open()
                    
                    # Find successor for this node
                    self.successor = node.find_successor(self.id)
                    
                    # Initialize finger table
                    self.init_finger_table()
                    
                    # Update others' finger tables
                    self.update_others()
                    
                    # Notify successor about this node
                    succ_transport = TSocket.TSocket(self.successor.ip, self.successor.port)
                    succ_transport = TTransport.TBufferedTransport(succ_transport)
                    succ_protocol = TBinaryProtocol.TBinaryProtocol(succ_transport)
                    succ_node = ComputeNodeService.Client(succ_protocol)
                    
                    try:
                        succ_transport.open()
                        self.predecessor = succ_node.get_predecessor()
                        my_node = NodeAddress(ip=self.ip, port=self.port, id=self.id)
                        succ_node.notify(my_node)
                        succ_transport.close()
                    except Exception as e:
                        print(f"Error notifying successor: {e}")
                    
                    # Confirm join with the supernode
                    supernode.confirm_join(self.id)
                    
                    node_transport.close()
                except Exception as e:
                    print(f"Error connecting to node: {e}")
            
            transport.close()
            
            # Fix fingers on all nodes
            self.fix_fingers_on_all_nodes()
            
        except Exception as e:
            print(f"Error joining network: {e}")
            if transport.isOpen():
                transport.close()
            sys.exit(1)
    
    def init_finger_table(self):
        self.finger_table = []
        for i in range(int(math.log2(self.max_nodes)) + 1):
            start = (self.id + 2**i) % self.max_nodes
            if self.successor is None:
                self.finger_table.append(NodeAddress(ip=self.ip, port=self.port, id=self.id))
            else:
                self.finger_table.append(self.successor)
    
    def update_others(self):
        for i in range(int(math.log2(self.max_nodes)) + 1):
            # Find the node whose ith finger might be this node
            p = (self.id - 2**i + self.max_nodes) % self.max_nodes
            
            # Find the node preceding p
            p_node = self.find_predecessor(p)
            
            # Connect to the node and update its finger table
            p_transport = TSocket.TSocket(p_node.ip, p_node.port)
            p_transport = TTransport.TBufferedTransport(p_transport)
            p_protocol = TBinaryProtocol.TBinaryProtocol(p_transport)
            p_client = ComputeNodeService.Client(p_protocol)
            
            try:
                p_transport.open()
                p_client.update_finger_table(self.id, i)
                p_transport.close()
            except Exception as e:
                print(f"Error updating finger table of node {p_node.id}: {e}")

    def find_successor(self, id):
        if self.successor is None:
            return NodeAddress(ip=self.ip, port=self.port, id=self.id)
        
        if self.id < id <= self.successor.id or (self.successor.id < self.id and (id > self.id or id <= self.successor.id)):
            return self.successor
        
        n0 = self.closest_preceding_node(id)
        if n0.id == self.id:
            return self.successor
        
        n0_transport = TSocket.TSocket(n0.ip, n0.port)
        n0_transport = TTransport.TBufferedTransport(n0_transport)
        n0_protocol = TBinaryProtocol.TBinaryProtocol(n0_transport)
        n0_client = ComputeNodeService.Client(n0_protocol)
        
        try:
            n0_transport.open()
            result = n0_client.find_successor(id)
            n0_transport.close()
            return result
        except Exception as e:
            print(f"Error finding successor through node {n0.id}: {e}")
            return self.successor

    def find_predecessor(self, id):
        n = self
        while not (n.id < id <= n.successor.id or (n.successor.id < n.id and (id > n.id or id <= n.successor.id))):
            n = n.closest_preceding_node(id)
            n_transport = TSocket.TSocket(n.ip, n.port)
            n_transport = TTransport.TBufferedTransport(n_transport)
            n_protocol = TBinaryProtocol.TBinaryProtocol(n_transport)
            n_client = ComputeNodeService.Client(n_protocol)
            
            try:
                n_transport.open()
                n = n_client.closest_preceding_node(id)
                n_transport.close()
            except Exception as e:
                print(f"Error finding predecessor through node {n.id}: {e}")
                break
        return n

    def closest_preceding_node(self, id):
        for i in range(len(self.finger_table) - 1, -1, -1):
            finger = self.finger_table[i]
            if (self.id < finger.id < id) or (id < self.id and (finger.id > self.id or finger.id < id)):
                return finger
        return NodeAddress(ip=self.ip, port=self.port, id=self.id)

    def fix_fingers(self):
        with self.finger_table_lock:
            for i in range(len(self.finger_table)):
                start = (self.id + 2**i) % self.max_nodes
                self.finger_table[i] = self.find_successor(start)
    
    def fix_fingers_on_all_nodes(self):
        # Start the recursive fix_fingers call from the successor
        self.fix_fingers_recursive(self.successor)
    
    def fix_fingers_recursive(self, node):
        if node.id == self.id:
            return
        
        transport = TSocket.TSocket(node.ip, node.port)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        client = ComputeNodeService.Client(protocol)
        
        try:
            transport.open()
            client.fix_fingers()
            next_node = client.get_successor()
            transport.close()
            
            self.fix_fingers_recursive(next_node)
        except Exception as e:
            print(f"Error fixing fingers on node {node.id}: {e}")

    def put_data(self, filename):
        key = self.hash_filename(filename)
        if self.is_responsible_for_key(key):
            print(f"Node {self.id} is responsible for file {filename} with key {key}")
            threading.Thread(target=self.train_model, args=(filename,)).start()
        else:
            next_node = self.find_successor(key)
            transport = TSocket.TSocket(next_node.ip, next_node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            node = ComputeNodeService.Client(protocol)
            
            try:
                transport.open()
                node.put_data(filename)
                transport.close()
            except Exception as e:
                print(f"Error forwarding data to node {next_node.id}: {e}")
    
    def get_model(self, filename):
        key = self.hash_filename(filename)
        if self.is_responsible_for_key(key):
            print(f"Node {self.id} is responsible for file {filename} with key {key}")
            with self.training_lock:
                if filename in self.models:
                    if self.training_files[filename]:
                        V, W = self.models[filename]
                        return ModelWeights(V=V.tolist(), W=W.tolist(), training_complete=True)
                    else:
                        return ModelWeights(V=[], W=[], training_complete=False)
                else:
                    return ModelWeights(V=[], W=[], training_complete=False)
        else:
            next_node = self.find_successor(key)
            transport = TSocket.TSocket(next_node.ip, next_node.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            node = ComputeNodeService.Client(protocol)
            
            try:
                transport.open()
                result = node.get_model(filename)
                transport.close()
                return result
            except Exception as e:
                print(f"Error getting model from node {next_node.id}: {e}")
                return ModelWeights(V=[], W=[], training_complete=False)
    
    def print_info(self):
        info = f"Node ID: {self.id}\n"
        info += f"IP: {self.ip}, Port: {self.port}\n"
        info += f"Predecessor: {self.predecessor.id if self.predecessor else 'None'}\n"
        info += f"Successor: {self.successor.id if self.successor else 'None'}\n"
        
        info += "Finger Table:\n"
        for i, finger in enumerate(self.finger_table):
            info += f"  [{i}]: {finger.id} (IP: {finger.ip}, Port: {finger.port})\n"
        
        info += "Key Range: "
        if self.predecessor:
            info += f"({self.predecessor.id}, {self.id}]\n"
        else:
            info += f"(-, {self.id}]\n"
        
        info += "Stored Files:\n"
        for filename in self.models.keys():
            info += f"  {filename} (Key: {self.hash_filename(filename)})\n"
        
        return info
    
    def get_predecessor(self):
        if self.predecessor:
            return self.predecessor
        return NodeAddress(ip="", port=0, id=-1)
    
    def set_predecessor(self, pred):
        self.predecessor = pred
    
    def get_successor(self):
        return self.successor
    
    def notify(self, node):
        if (self.predecessor is None or 
            (node.id > self.predecessor.id and node.id < self.id) or 
            (self.predecessor.id > self.id and (node.id > self.predecessor.id or node.id < self.id))):
            self.predecessor = node
    
    def is_responsible_for_key(self, key):
        if self.predecessor is None:
            return True
        
        if self.predecessor.id < self.id:
            return self.predecessor.id < key <= self.id
        else:
            return key > self.predecessor.id or key <= self.id
    
    def hash_filename(self, filename):
        h = hashlib.sha1(filename.encode()).hexdigest()
        return int(h, 16) % self.max_nodes
    
    def train_model(self, filename):
        with self.training_lock:
            self.training_files[filename] = False
        
        print(f"Node {self.id} starting training for file {filename}")
        
        model = mlp()
        
        try:
            file_path = os.path.join('letters', filename)
            model.init_training_random(file_path, 26, 20)
            model.set_momentum(0.7)
            train_error = model.train(0.0001, 250)
            
            V, W = model.get_weights()
            
            print(f"V = {V[:5]}")
            print(f"W = {W[:5]}")
            print(f"Training error: {train_error}")

            with self.training_lock:
                self.models[filename] = (V, W)
                self.training_files[filename] = True
            
            print(f"Node {self.id} finished training for file {filename} with error {train_error}")
        except Exception as e:
            print(f"Error training model for file {filename}: {e}")
            with self.training_lock:
                if filename in self.training_files:
                    del self.training_files[filename]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python compute_node.py <port>")
        sys.exit(1)
    
    port = int(sys.argv[1])

    supernode_host = 'csel-kh1250-10.cselabs.umn.edu'
    supernode_port = 9090
    handler = ComputeNodeHandler(port, supernode_host, supernode_port)
    processor = ComputeNodeService.Processor(handler)
    transport = TSocket.TServerSocket(host='0.0.0.0', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    
    print(f'Starting the compute node server on port {port}...')
    server.serve()