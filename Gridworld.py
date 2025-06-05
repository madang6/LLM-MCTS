import os
import yaml
import random
import matplotlib.pyplot as plt

from MCTS import Node, MCTS
from MCTS import RRT, principal_component_analysis

class GridWorld():
    def __init__(self, grid_width, grid_height, start, goal, sub_objectives=None, obstacles=0, max_rollout_steps=50, obstacle_cost=1):
        
        self.GRID_WIDTH = grid_width
        self.GRID_HEIGHT = grid_height
        self.START = start
        self.GOAL  = goal
        self.SUB_OBJECTIVES = sub_objectives or []

        self.OBSTACLES = obstacles or []
        self.OBSTACLE_COST = obstacle_cost

        self.MAX_ROLLOUT_STEPS = max_rollout_steps

        # Define available moves: action -> (dx, dy)
        self.MOVES = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
    
    def valid_moves(self, state):
        """
        Return list of valid actions from the given state.
        The state is a tuple: (position, objectives_remaining).
        """
        (pos,batt), _ = state
        x, y = pos
        moves = []
        for action, (dx, dy) in self.MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                moves.append(action)
        return moves

    def is_terminal(self, state):
        """
        Check if the state is terminal.
        Terminal if agent is at GOAL and, if using sub-objectives, they are all completed.
        """
        (pos, batt),objectives_remaining = state
        # If no sub-objectives are used, terminal when at GOAL.
        return pos == self.GOAL and (not objectives_remaining or len(objectives_remaining) == 0)
    
    def is_dead(self, state):
        """
        Check if the battery is zero.
        """

        (pos, batt), _ = state
        return batt <= 0
    
    def is_subobj(self, state):
        """
        Check if the state is a sub-objective.
        """
        (pos, batt), robj = state
        return pos in robj

    def is_obstacle(self, state):
        """
        Check if the state encounters an obstacle.
        """
        pos, _ = state
        return pos in self.OBSTACLES
    
    def next_state(self, state, action):
        """
        Return the next state given a state and an action.
        If the new position is one of the remaining objectives, remove it.
        """
        (pos, batt),objectives_remaining = state
        x, y = pos
        dx, dy = self.MOVES[action]
        new_pos = (x + dx, y + dy)
        new_batt = batt - 1
        
        # If there are sub-objectives, remove the one reached.
        if objectives_remaining and new_pos in objectives_remaining:
            new_objectives = tuple(obj for obj in objectives_remaining if obj != new_pos)
        else:
            new_objectives = objectives_remaining

        if new_pos in self.OBSTACLES:
            new_batt -= self.OBSTACLE_COST

        return ((new_pos, new_batt),new_objectives)
    
    def rollout(self, node):
        """Run a random simulation (rollout) from the given state."""
        current_state = node.state
        # current_reward = node.reward
        current_reward = 0
        for _ in range(self.MAX_ROLLOUT_STEPS):
            if current_state[0][1] > 0:
                if self.is_dead(current_state):
                    return current_reward
                elif self.is_subobj(current_state):
                    current_reward -= 1
                elif self.is_terminal(current_state):
                    return current_reward - 5  # Terminal state reached
                actions = self.valid_moves(current_state)
                if not actions:
                    break
                action = random.choice(actions)
                current_state = self.next_state(current_state, action)
            else:
                return 0
        return 0  # Did not reach terminal state

    def plot_episode(self, path, save_path=None):
        """Plot the grid world and the episode path."""
        # Extract positions from state tuples for plotting.
        xs = [state[0][0][0] + 0.5 for state in path]
        ys = [state[0][0][1] + 0.5 for state in path]
        
        plt.figure(figsize=(6, 6))
        
        # Draw grid lines.
        for x in range(self.GRID_WIDTH + 1):
            plt.plot([x, x], [0, self.GRID_HEIGHT], color='gray', lw=1)
        for y in range(self.GRID_HEIGHT + 1):
            plt.plot([0, self.GRID_WIDTH], [y, y], color='gray', lw=1)

        # Plot the episode path.
        # plt.plot(xs, ys, marker='o', color='red', linestyle='-', linewidth=2, markersize=8)
        # Create a color gradient based on battery remaining.
        battery_levels = [state[0][1] for state in path]
        norm = plt.Normalize(min(battery_levels), max(battery_levels))
        cmap = plt.get_cmap('viridis')
        colors = [cmap(norm(batt)) for batt in battery_levels]

        for i in range(len(xs) - 1):
            plt.plot(xs[i:i+2], ys[i:i+2], color=colors[i], linewidth=2)
        
        # Mark the start and goal positions.
        # import pdb; pdb.set_trace()
        plt.scatter(self.START[0][0] + 0.5, self.START[0][1] + 0.5, marker='s', s=100, color='green', label='Start')
        plt.scatter(self.GOAL[0] + 0.5, self.GOAL[1] + 0.5, marker='*', s=150, color='blue', label='Goal')
        
        # Mark the sub-objectives if any.
        if self.SUB_OBJECTIVES:
            for obj in self.SUB_OBJECTIVES:
                plt.scatter(obj[0] + 0.5, obj[1] + 0.5, marker='D', s=100, color='purple', label='Sub-objective')
        
        if self.OBSTACLES:
            for obs in self.OBSTACLES:
                plt.scatter(obs[0] + 0.5, obs[1] + 0.5, marker='X', s=100, color='black', label='Obstacle')

        plt.xlim(0, self.GRID_WIDTH)
        plt.ylim(0, self.GRID_HEIGHT)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Episode Path in Grid World")
        # Remove duplicate legend entries.
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path="config.yml"):
    project_root = os.path.dirname(os.path.abspath(__file__))
    decision_tree_path = os.path.join(project_root, "plots", "decision_tree.png")
    trajectory_path = os.path.join(project_root, "plots", "episode.png")

    config = load_config(config_path)

    visualize = config.get("visualize", True)
    verbose = config.get("verbose", True)
    save = config.get("save", True)

    random.seed(config.get("random_seed", 56))  # For reproducibility.

    GRID_WIDTH = config.get("grid_width", 5)
    GRID_HEIGHT = config.get("grid_height", 5)
    start_pos = tuple(config.get("start_position", [0, 0]))
    start_battery = config.get("start_battery", 20)
    START = (start_pos, start_battery)
    GOAL = tuple(config.get("goal", [2, 2]))
    SUB_OBJECTIVES = [tuple(obj) for obj in config.get("sub_objectives", [])]
    OBSTACLES = [tuple(obs) for obs in config.get("obstacles", [])]

    OBSTACLE_COST = config.get("obstacle_cost", 10)
    MAX_EPISODE_STEPS = config.get("max_episode_steps", 10) # Number of real agent actions possible in an episode
    MAX_ROLLOUT_STEPS = config.get("max_rollout_steps", 20) # Rollout depth


    #RRT parameters
    # MAX_EPISODE_STEPS = 1 # Number of real agent actions possible in an episode
    # MAX_ROLLOUT_STEPS = 200 # Rollout depth

    iterations_per_move = config.get("iterations_per_move", 100) # Number of MCTS iterations per move.

    grid = GridWorld(GRID_WIDTH, GRID_HEIGHT, START, GOAL, SUB_OBJECTIVES, OBSTACLES, MAX_ROLLOUT_STEPS,
                     OBSTACLE_COST)
    
    algorithm_name = config.get("algorithm", "mcts").upper()
    if algorithm_name == "RRT":
        robot = RRT(grid, MAX_EPISODE_STEPS)
    elif algorithm_name == "MCTS":
        robot = MCTS(grid, MAX_EPISODE_STEPS)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    random.seed(config.get("random_seed_episode", 62))  # For reproducibility.

    # Simulate an episode.
    if algorithm_name == "MCTS":
        mcts_txt_path = os.path.join(project_root, "formatted_trees.txt")
        mcts_list_path = os.path.join(project_root, "mcts_trees.pkl")
        episode_path, tree_list = robot.simulate_episode(iterations_per_move)
    elif algorithm_name == "RRT":
        rrt_txt_path = os.path.join(project_root, "formatted_trees.txt")
        rrt_list_path = os.path.join(project_root, "rrt_trees.pkl")
        episode_path, trajectories, episode_tree = robot.simulate_episode()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    if verbose:
        print("Episode path (each state shows (position, objectives_remaining)):")
        for state in episode_path:
            print(state)

    # for idx,trajectory in enumerate(trajectories):
    #     print(f"Trajectory {idx}:")
    #     for state in trajectory:
    #         print(state)

    # Save the tree_list
    # robot.save_tree(tree_list, mcts_list_path)
    
    # Save the list of MCTS trees from the episode for evaluation.
    if algorithm_name == "MCTS":
        if save:
            robot.save_formatted_trees_to_file(mcts_list_path, mcts_txt_path)
            print("MCTS trees saved to mcts_trees.pkl")
        if visualize and os.path.exists(mcts_txt_path):
            grid.plot_episode(episode_path, save_path=trajectory_path)
            robot.load_and_plot_tree(mcts_txt_path, 6, save_path=decision_tree_path)
            # principal_component_analysis(trajectories)
    elif algorithm_name == "RRT":
        if save:
            robot.save_tree(trajectories, rrt_txt_path)
            print("RRT trajectories saved to rrt_tree.pkl")
        if visualize and os.path.exists(rrt_txt_path):
            grid.plot_episode(episode_path, save_path=trajectory_path)
            robot.visualize_tree(episode_tree, 6, save_path=decision_tree_path)
            # principal_component_analysis(trajectories)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GridWorld simulation")
    parser.add_argument("--config", default="config.yml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)