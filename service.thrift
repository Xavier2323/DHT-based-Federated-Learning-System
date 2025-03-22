enum ResponseCode {
    SUCCESS = 200,
    ERROR = 500,
    NACK = 400
}

struct NodeAddress {
    1: string ip,
    2: i32 port, 
    3: i32 id
}

struct FingerTableEntry {
    1: NodeAddress node,
    2: i32 start,
    3: i32 end
}

struct ModelWeights {
    1: list<list<double>> V,
    2: list<list<double>> W,
    3: bool training_complete
}

service SupernodeService {
    i32 request_join(1: i32 port),
    ResponseCode confirm_join(1: i32 id),
    NodeAddress get_node()
}

service ComputeNodeService {
    // Chord protocol methods
    NodeAddress find_successor(1: i32 id),
    NodeAddress closest_preceding_node(1: i32 id),
    NodeAddress get_predecessor(),
    ResponseCode set_predecessor(1: NodeAddress node),
    ResponseCode set_successor(1: NodeAddress node),
    ResponseCode fix_fingers(1: optional i32 initiator_id),
    
    // Data distribution and model methods
    oneway void put_data(1: string filename),
    ModelWeights get_model(1: string filename),
    
    // Debugging
    ResponseCode print_info()
}