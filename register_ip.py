# register_ip.py
import socket
import os

FILENAME = "compute_nodes.txt"
BASE_PORT = 8000 # whatever port you want to start from

def get_next_port(ip, filename=FILENAME, base_port=BASE_PORT):
    existing_ports = []

    # if os.path.exists(filename):
    #     with open(filename, "r") as file:
    #         for line in file:
    #             line = line.strip()
    #             if not line: continue
    #             node_ip, port = line.split(",")
    #             if node_ip == ip:
    #                 existing_ports.append(int(port))

    # if existing_ports:
    #     next_port = max(existing_ports) + 1
    # else:
    #     next_port = base_port

    if os.path.exists(filename):
        with open(filename, "r") as file:
            next_port = base_port + len(file.readlines())
    else:
        next_port = base_port
        
    return next_port

def register_compute_node(ip, port, filename=FILENAME):
    entry = f"{ip},{port}\n"

    if os.path.exists(filename):
        with open(filename, "r") as file:
            if entry in file.readlines():
                print(f"[INFO] Node {ip}:{port} already registered.")
                return port
    
    with open(filename, "a") as file:
        file.write(entry)
        print(f"[INFO] Registered compute node {ip}:{port}")
        print(f"[INFO] You should run python3 compute_node.py {port}")
    
    return port

if __name__ == "__main__":
    ip = socket.gethostname()
    next_port = get_next_port(ip)
    register_compute_node(ip, next_port)

    print(f"[OUTPUT] Assigned port: {next_port}")