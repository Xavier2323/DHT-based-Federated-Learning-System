import glob
import sys
import os
sys.path.append('gen-py')
sys.path.insert(0, glob.glob('../thrift-0.19.0/lib/py/build/lib*')[0])

import time
import numpy as np
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from service import ComputeNodeService, SupernodeService
from service.ttypes import NodeAddress, ModelWeights, ResponseCode

from ML.ML import mlp

class Client:
    def __init__(self, supernode_host='0.0.0.0', supernode_port=9090):
        self.supernode_host = supernode_host
        self.supernode_port = supernode_port
        self.node_connection = None
        self.connect_to_network()
    
    def connect_to_network(self):
        # Connect to the supernode
        transport = TSocket.TSocket(self.supernode_host, self.supernode_port)
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        supernode = SupernodeService.Client(protocol)
        
        try:
            transport.open()
            
            # Get a node in the network
            node_address_info = supernode.get_node()
            node_address = node_address_info.nodeAddress
            current_size = node_address_info.currentSize
            
            print(f"Current size of the network: {current_size}")
            
            if node_address.id == -1:
                print("No nodes in the network")
                transport.close()
                sys.exit(1)
            
            # Connect to the node
            self.node_connection = {
                'host': node_address.ip,
                'port': node_address.port,
                'id': node_address.id
            }
            
            print(f"Connected to node {node_address.id} at {node_address.ip}:{node_address.port}")
            
            transport.close()
        except Exception as e:
            print(f"Error connecting to network: {e}")
            if transport.isOpen():
                transport.close()
            sys.exit(1)
    
    def put_data(self, directory):
        # Connect to the node
        transport = TSocket.TSocket(self.node_connection['host'], self.node_connection['port'])
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        node = ComputeNodeService.Client(protocol)
        
        try:
            transport.open()
            
            # Get list of files in the directory
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            
            print(f"Distributing {len(files)} files to node {self.node_connection['id']}")
            
            # Send each file to the network
            for filename in files:
                print(f"Sending file {filename}")
                node.put_data(filename)
            
            transport.close()
            return files
        except Exception as e:
            print(f"Error putting data to network: {e}")
            if transport.isOpen():
                transport.close()
            return []
    
    def get_models(self, files):
        # Connect to the node
        transport = TSocket.TSocket(self.node_connection['host'], self.node_connection['port'])
        transport = TTransport.TBufferedTransport(transport)
        protocol = TBinaryProtocol.TBinaryProtocol(transport)
        node = ComputeNodeService.Client(protocol)
        
        try:
            transport.open()
            
            models = {}
            all_trained = False
            
            print("Waiting for all models to be trained...")
            
            # Wait for all models to be trained
            while not all_trained:
                all_trained = True
                for filename in files:
                    if filename not in models:
                        print(f"Checking model for file {filename}")
                        result = node.get_model(filename)
                        
                        if result.training_complete:
                            print(f"Model for file {filename} is trained")
                            models[filename] = (np.array(result.V), np.array(result.W))
                        else:
                            print(f"Model for file {filename} is not trained yet")
                            all_trained = False
                
                if not all_trained:
                    print("Waiting for training to complete...")
                    time.sleep(2)
            
            transport.close()
            return models
        except Exception as e:
            print(f"Error getting models from network: {e}")
            if transport.isOpen():
                transport.close()
            return {}
    
    def aggregate_models(self, models):
        # Initialize average V and W
        first_model = list(models.values())[0]
        avg_V = np.zeros_like(first_model[0])
        avg_W = np.zeros_like(first_model[1])
        
        # Sum all models
        for V, W in models.values():
            avg_V += V
            avg_W += W
        
        # Calculate average
        avg_V /= len(models)
        avg_W /= len(models)
        
        # print(f"Aggregated model V: {avg_V}, W: {avg_W}")
        print(len(models), "models aggregated")
        return avg_V, avg_W
    
    def validate_model(self, V, W, validation_file):
        # Initialize MLP
        model = mlp()
        
        # Initialize with the aggregated weights
        model.init_training_model(validation_file, V, W)
        
        # print(f"model Weights: {model.W}, {model.V}")

        # Validate the model
        validation_error = model.validate(validation_file)
        
        return validation_error

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python client.py <data_directory> <validation_file>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    validation_file = sys.argv[2]
    
    supernode_host = '0.0.0.0'
    supernode_port = 9090
    
    client = Client(supernode_host, supernode_port)
    
    # Put data to the network
    files = client.put_data(data_dir)
    
    if not files:
        print("No files to distribute")
        sys.exit(1)
    
    start_time = time.time()
    # Get trained models
    models = client.get_models(files)
    
    if not models:
        print("No models received")
        sys.exit(1)
    
    # Aggregate models
    avg_V, avg_W = client.aggregate_models(models)
    
    # Validate the aggregated model
    validation_error = client.validate_model(avg_V, avg_W, validation_file)
    print(f"Validation error: {validation_error}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")