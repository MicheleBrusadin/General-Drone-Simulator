import argparse
import threading
import time
from src.display import Display
from src.drone import Drone
from src.human import Human
from src.agent import Agent
from src.monitor import Monitor
from src.utils import read_config

def initialize():
    config = read_config("config.yaml")
    monitor = Monitor(config)
    monitor.update_plot()
    display = Display(
        config=config["display"],
        update_frequency=config["display"]["update_frequency"],
        title="Drone Simulation"
    )
    drones = []
    for i in range(config["drone"]["n_drones"]):
        drones.append(Drone(
            config=config["drone"],
            display=config["display"],
            update_frequency=config["display"]["update_frequency"],
            startx=config["display"]["width"] // 2,
            starty=config["display"]["height"] // 2,
        ))

    human = Human(
        input_length=len(config["drone"]["motors"]),
    )

    agent = Agent(
        config=config
    )

    return display, monitor, drones, human, agent


# Thread management
RUNNING = True

def run_drone_simulation(drone, agent, monitor, target):
    global RUNNING
    total_reward = 0
    state, normalized_state = drone.get_state(), drone.get_normalized_state(target)

    while RUNNING:
        previous_state = normalized_state
        action = agent.get_action(state=drone.get_normalized_state(target))
        
        # Update drone state
        done = drone.update_state(inputs=action)
        state, normalized_state = drone.get_state(), drone.get_normalized_state(target)
        
        # Train agent
        reward = agent.get_reward(state=state, target=target, done=done)
        agent.train_short_memory(previous_state, action, reward, normalized_state, done)
        agent.remember(previous_state, action, reward, normalized_state, done)
        total_reward += reward
        
        # Handle death
        if done:
            monitor.log_data(total_reward, drone.survive_duration)
            drone.reset_state()
            state, normalized_state = drone.get_state(), drone.get_normalized_state(target)
            agent.n_games += 1
            agent.train_long_memory()
            print(f"Game {agent.n_games} done, epsilon: {round(agent.epsilon*100,1)}, total reward: {round(total_reward)}")
            total_reward = 0

def main(args: argparse.Namespace):
    global RUNNING
    display, monitor, drones, human, agent = initialize()  # adjust this function accordingly
    target = {
        "x": display.width // 2,
        "y": display.height // 2,
        "distance": 150
    }
    threads = []
    for drone in drones:
        t = threading.Thread(target=run_drone_simulation, args=(drone, agent, monitor, target))
        threads.append(t)
        t.start()

    try:
        while True:
            weights = agent.get_model_weights()
            # print('###### WEIGHTS START ######')
            # print(weights)
            # print('###### WEIGHTS END ######')
            display.update(target, drones, agent)
            monitor.update_plot()

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation interrupted. Exiting...")
        RUNNING = False

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("All threads have been stopped. Exiting...")

import cProfile
import pstats
if __name__ == "__main__":
    # Parse --human to manually control the drone
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", help="Human contrl", action="store_true")
    args = parser.parse_args()
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        main(args)
    except KeyboardInterrupt:
        print("Simulation interrupted. Exiting...")
    finally:
        profiler.disable()

        # Create Stats object
        stats = pstats.Stats(profiler)

        # Sort the statistics by cumulative time taken and print the top 20 lines
        stats.sort_stats('tottime').print_stats(20)