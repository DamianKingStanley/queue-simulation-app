import simpy
import random
import statistics
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
import os

app = Flask(__name__)

class Bookshop:
    def __init__(self, env, num_servers, service_time):
        self.env = env
        self.server = simpy.Resource(env, num_servers)
        self.service_time = service_time
        self.customers = []
        
        # Tracking server utilization
        self.num_servers = num_servers
        self.server_busy_time = [0] * num_servers  # Total busy time for each server
        self.server_last_busy_start = [None] * num_servers  # Last busy start time for each server

        # Tracking queue length over time
        self.queue_length_time = []  # List of tuples (time, queue_length)
        self.env.process(self.track_queue_length())

    def track_queue_length(self):
        """Track queue length at regular intervals."""
        while True:
            current_queue_length = len(self.server.queue)
            self.queue_length_time.append((self.env.now, current_queue_length))
            yield self.env.timeout(1)  # Record every 1 minute

    def serve_customer(self, customer):
        """Serve a customer."""
        arrival_time = self.env.now

        with self.server.request() as request:
            yield request
            start_service = self.env.now

            # Assign the customer to a server
            server_id = self.get_available_server()
            if server_id is not None:
                self.server_last_busy_start[server_id] = start_service

            # Service time
            service_duration = random.expovariate(1.0 / self.service_time)
            yield self.env.timeout(service_duration)
            end_service = self.env.now

            # Update server busy time
            if server_id is not None and self.server_last_busy_start[server_id] is not None:
                self.server_busy_time[server_id] += end_service - self.server_last_busy_start[server_id]
                self.server_last_busy_start[server_id] = None

            waiting_time = start_service - arrival_time
            service_time = end_service - start_service

            self.customers.append({
                'customer': customer,
                'arrival_time': arrival_time,
                'start_service': start_service,
                'end_service': end_service,
                'waiting_time': waiting_time,
                'service_time': service_time
            })

    def get_available_server(self):
        """Find an available server. Returns server_id or None."""
        for i in range(self.num_servers):
            if self.server_last_busy_start[i] is None:
                return i
        return None

def setup(env, num_customers, inter_arrival_time, bookshop):
    """Generate new customers at random intervals."""
    for i in range(num_customers):
        yield env.timeout(random.expovariate(1.0 / inter_arrival_time))
        env.process(bookshop.serve_customer(f'Customer {i+1}'))

def calculate_server_utilization(bookshop, simulation_time):
    utilizations = []
    for busy_time in bookshop.server_busy_time:
        utilization = (busy_time / simulation_time) * 100 if simulation_time > 0 else 0  # Percentage
        utilizations.append(utilization)
    average_utilization = statistics.mean(utilizations) if utilizations else 0
    return utilizations, average_utilization

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.form
    RANDOM_SEED = int(data.get('random_seed', 42))
    NEW_CUSTOMERS = int(data.get('new_customers', 100))
    INTER_ARRIVAL_TIME = float(data.get('inter_arrival_time', 5))
    SERVICE_TIME = float(data.get('service_time', 8))
    NUM_SERVERS = int(data.get('num_servers', 3))
    SIMULATION_TIME = float(data.get('simulation_time', 480))  # e.g., 8 hours

    # Set the random seed to ensure reproducible results
    random.seed(RANDOM_SEED)

    env = simpy.Environment()
    bookshop = Bookshop(env, NUM_SERVERS, SERVICE_TIME)
    env.process(setup(env, NEW_CUSTOMERS, INTER_ARRIVAL_TIME, bookshop))
    env.run(until=SIMULATION_TIME)

    simulation_time = env.now
    average_wait_time = statistics.mean([c['waiting_time'] for c in bookshop.customers]) if bookshop.customers else 0
    max_wait_time = max([c['waiting_time'] for c in bookshop.customers], default=0)
    utilizations, average_utilization = calculate_server_utilization(bookshop, simulation_time)

    # Generate and save the wait and service times graph
    plt.figure(figsize=(10, 6))
    wait_times = [c['waiting_time'] for c in bookshop.customers]
    service_times = [c['service_time'] for c in bookshop.customers]
    plt.plot(wait_times, label='Wait Time', marker='o')
    plt.plot(service_times, label='Service Time', marker='x')
    plt.xlabel('Customer')
    plt.ylabel('Time (minutes)')
    plt.title('Customer Wait and Service Times')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/simulation_graph.png')
    plt.close()

    # Generate and save the queue length over time graph
    plt.figure(figsize=(10, 6))
    if bookshop.queue_length_time:
        times, queue_lengths = zip(*bookshop.queue_length_time)
        plt.step(times, queue_lengths, where='post', label='Queue Length')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('static/queue_length_graph.png')
        plt.close()
    else:
        # If no data, create an empty graph
        plt.plot([], [])
        plt.xlabel('Time (minutes)')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Time')
        plt.grid(True)
        plt.savefig('static/queue_length_graph.png')
        plt.close()

    # Calculate Total Cost
    service_cost = float(data.get('service_cost', 10.0))
    wait_time_cost = float(data.get('wait_time_cost', 0.5))
    total_service_cost = service_cost * NUM_SERVERS * (SIMULATION_TIME / 60)  # Assuming service_cost is per hour
    total_wait_time = sum([c['waiting_time'] for c in bookshop.customers])
    total_wait_cost = wait_time_cost * total_wait_time
    total_cost = total_service_cost + total_wait_cost

    # Prepare the JSON response
    response = {
        'average_wait_time': round(average_wait_time, 2),
        'max_wait_time': round(max_wait_time, 2),
        'average_server_utilization': round(average_utilization, 2),
        'server_utilizations': [round(util, 2) for util in utilizations],
        'simulation_time': simulation_time,
        'customers_served': len(bookshop.customers),
        'total_cost': round(total_cost, 2),
        'simulation_graph': 'simulation_graph.png',
        'queue_length_graph': 'queue_length_graph.png'
    }

    return jsonify(response)

@app.route('/static/')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
