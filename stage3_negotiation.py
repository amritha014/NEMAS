import queue
import random
import threading
import time


class Coordinator:
    def __init__(self):
        self.shared_queue = queue.Queue()
        self.csm_id = None 
        # self.selected_csm_id = None

    def run(self, select_from_csms):
        agent1 = Agent1(self)  # Pass the Coordinator instance to Agent1
        agent2 = Agent2()

        assert self.shared_queue.empty(), "\nShared queue is not empty before negotiation."

        agent1_thread = agent1.initiate_negotiation(self.shared_queue, select_from_csms)
        time.sleep(1)  # Introduce a delay between starting threads
        agent2_thread = agent2.respond_to_negotiation(self.shared_queue)

        agent1_thread.join()
        agent2_thread.join()

        print("\nCoordinator: Process completed.")
        
    def get_selected_csm_id(self):
        return self.selected_csm_id

    
class Agent1:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def initiate_negotiation(self, shared_queue, select_from_csms):
        thread = threading.Thread(target=self.run_initiation, args=(shared_queue, select_from_csms))
        thread.start()
        return thread

    def run_initiation(self, shared_queue, select_from_csms):
        shared_queue.put(('select_from_csms', select_from_csms))
        time.sleep(3)  # Introduce a delay before passing control back
        print("\nAgent1: Initiating negotiation by sending select_from_csms to Agent2.")
        self.receive_csm_id_from_agent2(shared_queue)  # Call method to receive csm number from Agent2
        print("\nAgent1: Passing control back to Coordinator.")

    def receive_csm_id_from_agent2(self, shared_queue):
        message_type, csm_id = shared_queue.get()
        if message_type == 'csm_id':
            time.sleep(3)  # Introduce a delay before processing the csm number
            self.coordinator.csm = csm_id
            self.coordinator.selected_csm_id = csm_id  # Add this line
            print(f"\nAgent1: Received csm_id from Agent2: {csm_id}")
        time.sleep(3)  # Introduce a delay before passing control back
        print("\nAgent1: Passing control back to Coordinator.")

class Agent2:
    def respond_to_negotiation(self, shared_queue):
        thread = threading.Thread(target=self.run_response, args=(shared_queue,))
        thread.start()
        return thread

    def run_response(self, shared_queue):
        message_type, select_from_csms = shared_queue.get()
        if message_type == 'select_from_csms':
            time.sleep(3)  # Introduce a delay before processing the dictionaries
            print(f"\nAgent2: Received select_from_csms from Agent1.")
            csm_id = self.select_csm_id(select_from_csms)
            shared_queue.put(('csm_id', csm_id))  # Send the selected csm number to Agent1
        time.sleep(3)  # Introduce a delay before passing control back
        print("\nAgent2: Passing control back to Coordinator.")

    def select_csm_id(self, select_from_csms):
        csm_id = random.choice(list(select_from_csms.keys()))
        print(f"\nAgent2: Selected csm id: {csm_id}")
        return csm_id

if __name__ == "__main__":
    coordinator = Coordinator()
    coordinator.run(select_from_csms)


