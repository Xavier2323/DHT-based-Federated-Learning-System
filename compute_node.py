import glob
import sys
import os
import time
import threading
import hashlib
import socket
import numpy as np
from collections import defaultdict
import logging
logging.basicConfig(filename='compute_node.log', level=logging.DEBUG)

sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TCompactProtocol
from thrift.server import TServer

from service import ComputeNodeService, SupernodeService
from service.ttypes import NodeAddress, ModelWeights, ResponseCode, FingerTableEntry

from ML.ML import mlp

def is_port_open(ip, port, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            if s.connect_ex((ip, port)) == 0:
                print(f"Successor {ip}:{port} is reachable!")
                return True
        print(f"Waiting for {ip}:{port} to open...")
        time.sleep(2)  # Retry every 2 seconds
    return False

class ComputeNodeHandler:
    def __init__(self, port, supernode_host='0.0.0.0', supernode_port=9090, max_nodes=10):
        self.port = port
        self.supernode_host = supernode_host
        self.supernode_port = supernode_port
        self.max_nodes = max_nodes
        # Create a socket to determine the external IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            self.ip = sock.getsockname()[0]
        finally:
            sock.close()

        
        # Node identity and network state
        self.node_id = None
        self.predecessor = None
        self.successor = None
        self.finger_table = []
        self.entry_node = None  # Store the entry node when joining
        
        # Data storage
        self.stored_data = {}  # key: filename, value: True if training completed
        self.models = {}  # key: filename, value: (V, W) weights
        self.training_lock = threading.Lock()  # Lock for training operations
        
        # ML parameters
        self.k = 26  # Number of output classes (letters)
        self.h = 20  # Number of hidden units
        self.eta = 0.0001  # Learning rate
        self.epochs = 250  # Number of training epochs
        
        # Join the network
        self.join_network()
        
    def join_network(self):
        """Join the Chord DHT network through the supernode"""
        try:
            # Connect to supernode
            transport = TSocket.TSocket(self.supernode_host, self.supernode_port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            supernode = SupernodeService.Client(protocol)
            
            transport.open()
            
            # Request node ID from supernode
            self.node_id = supernode.request_join(self.port)
            
            if self.node_id == ResponseCode.NACK:
                print("Failed to join network: NACK received from supernode")
                transport.close()
                sys.exit(1)
            
            print(f"Received node ID: {self.node_id}")
            
            # Get a node address to join the network
            node_address_info = supernode.get_node()
            node_address = node_address_info.nodeAddress
            current_size = node_address_info.currentSize
            
            print(f"Current network size: {current_size}")
            
            if node_address.id == -1:
                # First node in the network
                print("This is the first node in the network")
                
                self.predecessor = NodeAddress(ip=self.ip, port=self.port, id=self.node_id)
                self.successor = NodeAddress(ip=self.ip, port=self.port, id=self.node_id)
                
                # Initialize finger table (pointing to self)
                self.init_finger_table()
                
                # Confirm join to supernode
                supernode.confirm_join(self.node_id)
            else:
                # Join existing network through the received node
                print(f"Joining network through node {node_address.id} at {node_address.ip}:{node_address.port}")
                self.join_existing_network(node_address, current_size)
                
                # Confirm join to supernode
                supernode.confirm_join(self.node_id)
            
            transport.close()
            
        except Exception as e:
            print(f"Error joining network: {e}")
            sys.exit(1)
    
    def join_existing_network(self, entry_node, current_size):
        """Join an existing Chord network through entry_node with improved error handling"""
        # Store the entry node for later use
        self.entry_node = entry_node
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Connect to entry node
                transport = TSocket.TSocket(entry_node.ip, entry_node.port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                node = ComputeNodeService.Client(protocol)
                
                transport.open()
                print(f"Connected to entry node {entry_node.id} at {entry_node.ip}:{entry_node.port}")
                
                # Find successor for this node's ID
                self_address = NodeAddress(ip=self.ip, port=self.port, id=self.node_id)
                successor_address = node.find_successor(self.node_id)
                
                # Check if successor is valid
                if not successor_address or successor_address.id == -1:
                    print(f"Invalid successor received. Retry {attempt + 1}")
                    transport.close()
                    time.sleep(2)
                    continue
                
                print(f"Closing connection to entry node {entry_node.id}")
                transport.close()
                
                # Set successor
                self.successor = successor_address
                
                # Get predecessor from successor with robust connection
                transport2 = TSocket.TSocket(successor_address.ip, successor_address.port)
                transport2 = TTransport.TBufferedTransport(transport2)
                protocol2 = TBinaryProtocol.TBinaryProtocol(transport2)
                succ_node = ComputeNodeService.Client(protocol2)
                
                transport2.open()
                print(f"Connected to successor {self.successor.id} at {self.successor.ip}:{self.successor.port}")
                
                # Get predecessor from successor and update successor's predecessor to this node
                self.predecessor = succ_node.get_predecessor()
                succ_node.set_predecessor(self_address)
                
                # Update predecessor's successor to this node
                transport3 = TSocket.TSocket(self.predecessor.ip, self.predecessor.port)
                transport3 = TTransport.TBufferedTransport(transport3)
                protocol3 = TBinaryProtocol.TBinaryProtocol(transport3)
                pred_node = ComputeNodeService.Client(protocol3)
                
                transport3.open()
                print(f"Connected to predecessor {self.predecessor.id} at {self.predecessor.ip}:{self.predecessor.port}")
                pred_node.set_successor(self_address)
                
                # Close the connections
                transport3.close()
                transport2.close()
                
                # Initialize finger table
                self.init_finger_table()
                
                # Fix fingers with improved error handling
                self.fix_fingers(self.node_id, hop_count=0, max_hops=current_size + 1)
                
                return  # Successful join
            
            except Exception as e:
                print(f"Error joining existing network (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Close any open transports
                for t in [transport, transport2, transport3]:
                    try:
                        if t and t.isOpen():
                            t.close()
                    except:
                        pass
                
                # Wait before retry
                time.sleep(2)
        
        # If all attempts fail
        print("Failed to join network after multiple attempts")
        sys.exit(1)

    def init_finger_table(self):
        """Initialize finger table with appropriate entries"""
        self.finger_table = []
        
        # Calculate number of fingers needed (log base 2 of max_nodes)
        num_fingers = 0
        temp = self.max_nodes
        while temp > 0:
            num_fingers += 1
            temp = temp // 2
        
        # First finger is always the successor
        self.finger_table.append(FingerTableEntry(
            node=self.successor,
            start=(self.node_id + 2**0) % self.max_nodes,
            end=(self.node_id + 2**1) % self.max_nodes
        ))
        
        # Additional fingers
        for i in range(1, num_fingers):
            start = (self.node_id + 2**i) % self.max_nodes
            end = (self.node_id + 2**(i+1)) % self.max_nodes
            
            # Find successor for this finger
            successor = self.find_successor(start)
            
            self.finger_table.append(FingerTableEntry(
                node=successor,
                start=start,
                end=end
            ))
        
        print(f"Initialized finger table with {len(self.finger_table)} entries")
        
    def fix_fingers(self, initiator_id=None, hop_count=0, max_hops=10):
        """Updated fix_fingers method with hop count to prevent infinite recursion"""
        print(f"Node {self.node_id}: Fixing fingers (hop count: {hop_count})")
        
        # Check hop count to prevent infinite propagation
        MAX_HOPS = max_hops
        if hop_count >= MAX_HOPS:
            print(f"Node {self.node_id}: Max hop count reached. Stopping fix_fingers.")
            return ResponseCode.SUCCESS
        
        # Re-initialize finger table
        try:
            self.init_finger_table()
            print(f"Node {self.node_id}: Finger table successfully initialized")
            self.print_info()  # Print detailed node information after fixing fingers
        except Exception as e:
            print(f"Error initializing finger table: {e}")
            return ResponseCode.ERROR
        
        # If this is the first call, set initiator_id to this node's ID
        if initiator_id is None:
            initiator_id = self.node_id
        
        # Propagate to successor if not self
        if self.successor.id != self.node_id:
            try:
                time.sleep(1)  # Small delay to prevent immediate retry
                
                # Check successor reachability
                if not is_port_open(self.successor.ip, self.successor.port, timeout=5):
                    print(f"Node {self.node_id}: Successor {self.successor.id} is not reachable")
                    return ResponseCode.ERROR
                
                # Establish connection to successor
                transport = TSocket.TSocket(self.successor.ip, self.successor.port)
                transport.setTimeout(1000)  # 1 second timeout
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                node = ComputeNodeService.Client(protocol)
                
                # Open transport and call fix_fingers
                transport.open()
                print(f"Node {self.node_id}: Fixing fingers on successor {self.successor.id}")
                
                # Increment hop count when propagating
                node.fix_fingers(initiator_id, hop_count + 1, max_hops)
                
                # Close transport
                transport.close()
            
            except Exception as e:
                print(f"Error propagating fix_fingers: {e}")
                return ResponseCode.ERROR
        
        # Check if we've completed a full circle
        if self.node_id == initiator_id:
            print(f"Node {self.node_id}: Finger table update completed full circle")
            return ResponseCode.SUCCESS
        
        # If we reach here, the fix_fingers call was successful
        print(f"Node {self.node_id}: Fix_fingers completed successfully")
        return ResponseCode.SUCCESS
    
    def find_successor(self, id):
        """Find the successor node for a given ID"""
        # If ID is between this node and its successor, return successor
        if self.is_between(id, self.node_id, self.successor.id, True):
            return self.successor
        
        # Otherwise, forward the query to the closest preceding node
        closest_preceding = self.closest_preceding_node(id)
        
        logging.debug(f"Data distribution: Node {self.node_id} forwarding to {closest_preceding.id}")
        print(f"Node {self.node_id}: Closest preceding node for {id} is {closest_preceding.id}")
        
        # If closest preceding node is this node, return successor
        if closest_preceding.id == self.node_id:
            return self.successor
        
        try:
            transport = TSocket.TSocket(closest_preceding.ip, closest_preceding.port)
            transport = TTransport.TBufferedTransport(transport)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            node = ComputeNodeService.Client(protocol)
            
            
            transport.open()
            print(f"Node {self.node_id}: Forwarding find_successor to node {closest_preceding.id}")
            result = node.find_successor(id)
            transport.close()
            
            return result
        except Exception as e:
            print(f"Error in find_successor: {e}")
            return self.successor
    
    def find_successor_local(self, id):
        """Local version of find_successor to avoid circular imports during initialization"""
        # If ID is between this node and its successor, return successor
        if self.is_between(id, self.node_id, self.successor.id, True):
            return self.successor
        
        # Otherwise, return the successor (simplified for initialization)
        return self.successor
    
    def closest_preceding_node(self, id):
        """Find the closest preceding node for a given ID in the finger table"""
        # Search backwards through finger table
        for i in range(len(self.finger_table) - 1, -1, -1):
            finger_id = self.finger_table[i].node.id
            
            # Check if finger is between current node and target id
            if self.is_between(finger_id, self.node_id, id, False):
                logging.debug(f"Node {self.node_id}: Closest preceding node for {id} is {finger_id} on finger {i}")
                return self.finger_table[i].node
        
        return NodeAddress(ip=self.ip, port=self.port, id=self.node_id)
    
    def is_between(self, id, start, end, inclusive=False):
        """Check if id is between start and end in the circular ID space"""
        # Handle wrap-around in the circular ID space
        if start < end:
            if inclusive:
                return start < id <= end
            else:
                return start < id < end
        else:  # Wrapping around
            if inclusive:
                return start < id or id <= end
            else:
                return start < id or id < end
    
    def get_predecessor(self):
        """Get this node's predecessor"""
        return self.predecessor
    
    def set_predecessor(self, node):
        """Set this node's predecessor"""
        self.predecessor = node
        return ResponseCode.SUCCESS
    
    def set_successor(self, node):
        """Set this node's successor"""
        self.successor = node
        
        # Update first finger entry
        if self.finger_table:
            self.finger_table[0].node = node
        
        return ResponseCode.SUCCESS
    
    # Data distribution and retrieval methods
    
    def hash_filename(self, filename):
        """Hash a filename to a node ID in the Chord network"""
        # Create a hash of the filename
        hash_obj = hashlib.md5(filename.encode())
        # Convert to int and take modulo max_nodes
        return int(hash_obj.hexdigest(), 16) % self.max_nodes
    
    def put_data(self, filename):
        """Store data in the DHT or forward to the appropriate node"""
        # Hash the filename to get its key
        key = self.hash_filename(filename)
        
        print(f"Node {self.node_id}: Received put_data for file {filename} with key {key}")
        
        with open(f'routing/{filename}_routing.txt', 'a') as file:
            file.write(f"Node ID: {self.node_id}, Filename: {filename}, Key: {key}, Time: {time.time()}\n")

        
        # Check if this node is responsible for this key
        if self.is_between(key, self.predecessor.id, self.node_id, True):
            print(f"Node {self.node_id}: Storing file {filename}")
            
            # Mark as not trained yet
            self.stored_data[filename] = False
            
            # Start training in a separate thread
            training_thread = threading.Thread(target=self.train_data, args=(filename,))
            training_thread.daemon = True
            training_thread.start()
            
            return ResponseCode.SUCCESS
        else:
            # Forward to the closest preceding node
            next_node = self.closest_preceding_node(key)
            
            if next_node.id == self.node_id:
                next_node = self.successor
            
            try:
                print(f"Node {self.node_id}: Forwarding file {filename} to node {next_node.id}")
                
                transport = TSocket.TSocket(next_node.ip, next_node.port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                node = ComputeNodeService.Client(protocol)
                
                transport.open()
                node.put_data(filename)
                transport.close()
                
                return ResponseCode.SUCCESS
            except Exception as e:
                print(f"Error forwarding put_data: {e}")
                return ResponseCode.ERROR
    
    def get_model(self, filename):
        """Retrieve a model from the DHT or forward to the appropriate node"""
        # Hash the filename to get its key
        key = self.hash_filename(filename)
        
        print(f"Node {self.node_id}: Received get_model for file {filename} with key {key}")
        
        # Check if this node is responsible for this key
        if self.is_between(key, self.predecessor.id, self.node_id, True):
            print(f"Node {self.node_id}: Checking local model for file {filename}")
            
            # Check if file exists locally
            if filename not in self.stored_data:
                print(f"Node {self.node_id}: File {filename} not found")
                return ModelWeights(V=[], W=[], training_complete=False)
            
            # Check if training is complete
            if not self.stored_data[filename]:
                print(f"Node {self.node_id}: File {filename} is still training")
                return ModelWeights(V=[], W=[], training_complete=False)
            
            # Return the trained model
            V, W = self.models[filename]
            
            # Convert numpy arrays to lists for thrift
            V_list = V.tolist()
            W_list = W.tolist()
            
            print(f"Node {self.node_id}: Returning model for file {filename}")
            return ModelWeights(V=V_list, W=W_list, training_complete=True)
        else:
            # Forward to the closest preceding node
            next_node = self.closest_preceding_node(key)
            
            if next_node.id == self.node_id:
                next_node = self.successor
            
            try:
                print(f"Node {self.node_id}: Forwarding get_model for file {filename} to node {next_node.id}")
                
                transport = TSocket.TSocket(next_node.ip, next_node.port)
                transport = TTransport.TBufferedTransport(transport)
                protocol = TBinaryProtocol.TBinaryProtocol(transport)
                node = ComputeNodeService.Client(protocol)
                
                transport.open()
                result = node.get_model(filename)
                transport.close()
                
                return result
            except Exception as e:
                print(f"Error forwarding get_model: {e}")
                return ModelWeights(V=[], W=[], training_complete=False)
    
    def train_data(self, filename):
        """Train a model on the given file"""
        with self.training_lock:
            print(f"Node {self.node_id}: Starting training for file {filename}")
            
            try:
                # Initialize MLP
                model = mlp()
                
                # Get full path to the file
                file_path = os.path.join('letters', filename)
                
                # Initialize with random weights
                success = model.init_training_random(file_path, self.k, self.h)
                
                if not success:
                    print(f"Node {self.node_id}: Failed to initialize model for file {filename}")
                    self.stored_data[filename] = False
                    return
                
                # Train the model
                training_error = model.train(self.eta, self.epochs)
                
                print(f"Node {self.node_id}: Training complete for file {filename} with error {training_error}")
                
                # Store the trained weights
                V, W = model.get_weights()
                self.models[filename] = (V, W)
                
                # Mark as trained
                self.stored_data[filename] = True
            except Exception as e:
                print(f"Node {self.node_id}: Error training model for file {filename}: {e}")
                self.stored_data[filename] = False
    
    def print_info(self):
        """Print node information for debugging"""
        print("\n---- Node Information ----")
        print(f"Node ID: {self.node_id}")
        print(f"IP:Port: {self.ip}:{self.port}")
        print(f"Predecessor: ID={self.predecessor.id}, IP:Port={self.predecessor.ip}:{self.predecessor.port}")
        print(f"Successor: ID={self.successor.id}, IP:Port={self.successor.ip}:{self.successor.port}")
        if self.entry_node:
            print(f"Entry Node: ID={self.entry_node.id}, IP:Port={self.entry_node.ip}:{self.entry_node.port}")
        
        print("\n---- Finger Table ----")
        for i, finger in enumerate(self.finger_table):
            print(f"Finger {i+1}: Node={finger.node.id}, Range=[{finger.start}, {finger.end})")
        
        print("\n---- Stored Data ----")
        accepted_keys = []
        if self.predecessor:
            for i in range((self.predecessor.id + 1) % self.max_nodes, (self.node_id + 1) % self.max_nodes):
                accepted_keys.append(i % self.max_nodes)
            if self.predecessor.id >= self.node_id:  # Handle wrap-around
                for i in range(0, (self.node_id + 1) % self.max_nodes):
                    accepted_keys.append(i)
        
        print(f"Accepted key range: {accepted_keys}")
        print(f"Stored files: {list(self.stored_data.keys())}")
        trained_files = [f for f, trained in self.stored_data.items() if trained]
        print(f"Trained files: {trained_files}")
        
        return ResponseCode.SUCCESS

def run_server(port, supernode_host='0.0.0.0', supernode_port=9090):
    print(f'Starting compute node server on port {port}...')
    transport = TSocket.TServerSocket(host='0.0.0.0', port=port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    
    handler = ComputeNodeHandler(port, supernode_host, supernode_port)
    processor = ComputeNodeService.Processor(handler)
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    
    handler.print_info()  # Print initial node info
    server.serve()
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python compute_node.py <port>")
        sys.exit(1)
    
    port = int(sys.argv[1])
    supernode_host = 'csel-kh1250-06.cselabs.umn.edu'
    supernode_port = 9090
    run_server(port, supernode_host, supernode_port)