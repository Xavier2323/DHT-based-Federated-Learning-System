// Define the model weights structure
struct ModelWeights {
  1: list<list<double>> V;
  2: list<list<double>> W;
  3: bool training_complete;
}

// Define the address structure
struct NodeAddress {
  1: string ip;
  2: i32 port;
  3: i32 id;
}

// Define the response codes
enum ResponseCode {
  SUCCESS = 0,
  NACK = 1,
  WAIT = 2,
  NOT_FOUND = 3,
  EMPTY_NETWORK = 4
}

// Supernode Service
service SupernodeService {
  // Request to join the network
  ResponseCode request_join(1: i32 port);
  
  // Confirm joining the network
  ResponseCode confirm_join(1: i32 id);
  
  // Get a random node in the network
  NodeAddress get_node();
}

// Compute Node Service
service ComputeNodeService {
  // Put data into the network
  oneway void put_data(1: string filename);
  
  // Get model from the network
  ModelWeights get_model(1: string filename);
  
  // Fix finger tables
  void fix_fingers();
  
  // Print node information
  string print_info();
  
  // Additional functions for node join and maintenance
  NodeAddress find_successor(1: i32 id);
  NodeAddress find_predecessor(1: i32 id);
  NodeAddress get_predecessor();
  void set_predecessor(1: NodeAddress pred);
  NodeAddress get_successor();
  void set_successor(1: NodeAddress succ);
  void notify(1: NodeAddress node);
}