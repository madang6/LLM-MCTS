import os
import math
import random
import pickle

import re
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import numpy as np

class Node:
    def __init__(self, state, parent=None, environment=None):
        self.state = state
        self.parent = parent
        self.environment = environment
        self.battery = state[0][1]
        self.children = {}             # action -> child Node
        self.visits = 0
        self.reward = 0.0
        self.untried_actions = environment.valid_moves(state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
class RRT:
    def __init__(self, environment, max_episode_steps=100):
        self.environment       = environment
        print(f"Environment attributes: {vars(environment)}")
        self.MAX_EPISODE_STEPS = max_episode_steps

    def simulate_episode(self):
        """
        Simulate an episode using RRT to make decisions
        """
        # If SUB_OBJECTIVES is not set or empty, use an empty tuple.
        objectives = tuple(self.environment.SUB_OBJECTIVES) if self.environment.SUB_OBJECTIVES else tuple()
        current_state = (self.environment.START, objectives)
        # path = [current_state]
        root = Node(current_state, environment=self.environment)
        # tree = [root]
        steps = 0
        
        successes = []
        failures = []
        while steps < self.MAX_EPISODE_STEPS: 
            episode_tree = self.rrt(root)
            #TODO Implement some system to find successes and failures across MAX_EPISODE_STEPS

            # Build and visualize the tree graph.
            # import pdb; pdb.set_trace()
            # graph = self.build_graph_from_tree(episode_tree)
            steps += 1
            # self.visualize_tree(episode_tree)

            # Suppose 'episode_tree' is the list of nodes returned by your rrt() function.
            # end_node = None
            # for node in episode_tree:
            #     if self.environment.is_terminal(node.state):
            #         end_node = node
            #         break

            # if end_node is not None:
            #     path = self.reconstruct_path(end_node)
            #     print("Success! Path from root to terminal node:")
            #     for state in path:
            #         print(state)
            # else:
            #     print("No terminal (successful) node found in the tree.")

            # Find the candidate node that is closest to terminal conditions.
            end_node = self.find_best_candidate(episode_tree, self.environment.GOAL)

            trajectories = []
            for idx, node in enumerate(episode_tree):
                if node != end_node and not node.children and node.battery == 0:
                    # import pdb; pdb.set_trace()
                    print(f"Node {idx}: {node.state}, Children: {list(node.children.keys())}")
                    trajectories.append(self.reconstruct_path(node))

            if end_node is not None:
                path = self.reconstruct_path(end_node)
                # print("Best candidate path from root to near-terminal node:")
                # for state in path:
                    # print(state)
                # self.visualize_tree(episode_tree, fontsize=8, save_path=f"/home/admin/workspaces/explainability/Python-LLM/plots/rrt_tree_{steps}.png")
                return path, trajectories, episode_tree
            else:
                print("No candidate node found.")
                return None, trajectories

    def terminal_distance(self, state, goal):
        """
        Compute a distance metric from state to terminal.
        state: (((x, y), battery), remaining_objectives)
        goal: (x, y)
        The metric is the Manhattan distance from the agent's position to the goal,
        plus the sum of Manhattan distances from the agent's position to each remaining objective.
        """
        (pos, battery), objectives = state
        # Manhattan distance from position to goal.
        d_goal = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        # Penalty for each remaining sub-objective.
        d_obj = 0
        for obj in objectives:
            d_obj += abs(pos[0] - obj[0]) + abs(pos[1] - obj[1])
        return d_goal + d_obj

    def find_best_candidate(self, tree, goal):
        """
        Given a list of nodes in the tree, find the node with the smallest terminal distance.
        """
        best_node = None
        best_distance = float('inf')
        for node in tree:
            d = self.terminal_distance(node.state, goal)
            if d < best_distance:
                best_distance = d
                best_node = node
        return best_node

    def reconstruct_path(self,end_node):
        """
        Reconstruct the path from the given end node to the root using parent pointers.
        Returns a list of states from the root to the end node.
        """
        path = []
        current_node = end_node
        while current_node is not None:
            path.append(current_node.state)
            current_node = current_node.parent
        return list(reversed(path))
    
    def hierarchy_pos(self, G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        If the graph is a tree, this function returns a dictionary of positions 
        for a hierarchical layout.
        
        Parameters:
        G : the graph (should be a tree)
        root : the root node of the tree
        width : horizontal space allocated for this branch of the tree
        vert_gap : gap between levels of the hierarchy
        vert_loc : vertical location of root
        xcenter : horizontal location of root
        
        Returns:
        pos : a dict mapping each node to a (x, y) position.
        """
        pos = {root: (xcenter, vert_loc)}
        children = list(G.neighbors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos.update(self.hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                        vert_loc=vert_loc - vert_gap, xcenter=nextx))
        return pos
    
    def build_graph_from_tree(self, tree):
        G = nx.DiGraph()
        # import pdb; pdb.set_trace()
        for node in tree:
            node_id = id(node)
            # import pdb; pdb.set_trace()
            G.add_node(node_id, label=str(node.state))
            if node.parent:
                G.add_edge(id(node.parent), node_id)
        return G

    def visualize_tree(self, tree, fontsize=8, save_path=None):
        # Build graph from tree
        graph = self.build_graph_from_tree(tree)
        root_id = id(tree[0])
        pos = self.hierarchy_pos(graph, root=root_id)  # use our custom function
        labels = nx.get_node_attributes(graph, 'label')
        plt.figure(figsize=(25, 25))
        nx.draw(graph, pos, labels=labels, node_color='lightblue', arrowstyle='->', arrowsize=10, font_size=fontsize)
        plt.title("Hierarchical Layout of RRT Tree")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def sample_state(self):
        x = random.randint(0, self.environment.GRID_WIDTH - 1)
        y = random.randint(0, self.environment.GRID_HEIGHT - 1)
        return (x, y)
    
    # def find_nearest(self, tree, sample):
    #     # tree is a list of nodes, each with a state where state[0] is the position.
    #     nearest_node = None
    #     min_dist = float('inf')
    #     for node in tree:
    #         # import pdb; pdb.set_trace()
    #         pos = node.state[0]  # position component of the state
    #         dist = abs(pos[0] - sample[0]) + abs(pos[1] - sample[1])  # Manhattan distance
    #         if dist < min_dist:
    #             min_dist = dist
    #             nearest_node = node
    #     return nearest_node
    def find_nearest(self, tree, sample, initial_k=5, max_k=50):
        """
        Find the nearest valid node to the sample using a KDTree on positions,
        then post-filter the candidate nodes by checking that they are not terminal
        and not dead (using self.environment.is_terminal and self.environment.is_dead).
        
        If none of the first k candidates are valid, increase k (up to max_k) as a fallback.
        
        Parameters:
        tree: list of Node objects; each Node has a state where state[0] is the (x,y) position.
        sample: a tuple (x, y) representing the sample position.
        initial_k: the starting number of neighbors to query.
        max_k: the maximum number of neighbors to query as fallback.
        
        Returns:
        The first valid node found, or None if no valid candidate exists within max_k neighbors.
        """
        # Build a list of positions from the nodes.
        positions = [node.state[0][0] for node in tree]
        # Build the KDTree on these positions.
        kd_tree = cKDTree(positions)
        
        k = initial_k
        while k <= max_k:
            # Query the k nearest neighbors using Manhattan distance (p=1).
            dists, indices = kd_tree.query(sample, k=k, p=1)
            
            # Ensure indices is iterable.
            if k == 1:
                indices = [indices]
            
            # Check each candidate for validity.
            for idx in indices:
                candidate = tree[idx]
                if not self.environment.is_terminal(candidate.state) and not self.environment.is_dead(candidate.state):
                    return candidate
            
            # Increase k and try again.
            k *= 2  # Alternatively, k += initial_k
            
        # If no valid node is found within max_k neighbors, return None.
        return None
    
    def steer(self, from_state, to_state, step_size=1):
        from_x, from_y = from_state[0][0][0],from_state[0][0][1]  # current position from the tree
        to_x, to_y = to_state           # sampled position
        # Compute direction vector (using simple sign function)
        # import pdb; pdb.set_trace()
        dx = (to_x - from_x)
        dy = (to_y - from_y)
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        # Take a step in the direction (here step_size is usually 1 in a grid)
        actions = self.environment.valid_moves(from_state)
        best_next_state = None
        min_dist = float('inf')
        for action in actions:
            next_state = self.environment.next_state(from_state, action)
            # import pdb; pdb.set_trace()
            dist = abs(next_state[0][0][0] - to_state[0]) + abs(next_state[0][0][1] - to_state[1])
            if dist < min_dist:
                min_dist = dist
                best_next_state = next_state
                best_action = action
        return best_next_state, best_action

    def rrt(self, root):
        tree = [root]
        for _ in range(self.environment.MAX_ROLLOUT_STEPS):
            if all(self.environment.is_terminal(node.state) or self.environment.is_dead(node.state) for node in tree):
                break
            # print("here1")
            # import pdb; pdb.set_trace()
            random_state = self.sample_state()
            # print("here2")

            # Find the nearest node in the tree to the random state.
            valid_nearest_node = self.find_nearest(tree, random_state)
            # print("here3")

            # Steer from the nearest node toward the random state.
            new_state,action = self.steer(valid_nearest_node.state, random_state)
            # print("here4")

            # Create a new node and add to the tree.
            new_node = Node(state=new_state, parent=valid_nearest_node, environment=self.environment)
            valid_nearest_node.children[action] = new_node
            tree.append(new_node)
            # print("here5")

        return tree

    def save_tree(self, tree, filename):
        """Serialize the given object (tree or list of trees) to the specified filename."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree(self, filename):
        """Load a previously saved tree (or list of trees) from the specified filename."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class MCTS:
    def __init__(self, environment, max_episode_steps=100):
        self.environment       = environment
        print(f"Environment attributes: {vars(environment)}")
        self.MAX_EPISODE_STEPS = max_episode_steps

    def backpropagate(self, node, reward):
        """Propagate the simulation result up to the root node."""
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def simulate_episode(self, iterations_per_move=1000):
        """
        Simulate an episode using MCTS at each decision point.
        Returns both the path taken (list of states) and the list of MCTS trees (one per decision).
        """
        # If SUB_OBJECTIVES is not set or empty, use an empty tuple.
        objectives = tuple(self.environment.SUB_OBJECTIVES) if self.environment.SUB_OBJECTIVES else tuple()
        current_state = (self.environment.START, objectives)
        # import pdb; pdb.set_trace()
        path = [current_state]
        tree_list = []  # Save the root of the MCTS tree from each decision point.
        steps = 0

        while not self.environment.is_terminal(current_state) and steps < self.MAX_EPISODE_STEPS and not self.environment.is_dead(current_state):
            root = Node(current_state, environment=self.environment)
            best_action = self.mcts(root, iterations_per_move)
            tree_list.append(root)  # Save the current MCTS tree.
            if best_action is None:
                break  # No valid moves available.
            current_state = self.environment.next_state(current_state, best_action)
            path.append(current_state)
            steps += 1

        return path, tree_list
    
    def mcts(self, root, iterations):
        """Perform MCTS iterations starting from the root node and return the best move."""
        for _ in range(iterations):
            node = root

            # SELECTION: Traverse the tree until a node with untried actions is reached
            # print(f"environment vars: {vars(self.environment)}")
            while not self.environment.is_terminal(node.state) and node.is_fully_expanded() and not self.environment.is_dead(node.state):
                node = uct_select_child(node)

            # EXPANSION: Expand a new child if possible (and if not terminal)
            if not self.environment.is_terminal(node.state) and node.untried_actions and not self.environment.is_dead(node.state):
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_state = self.environment.next_state(node.state, action)
                child = Node(new_state, parent=node, environment=self.environment)
                node.children[action] = child
                node = child

            # SIMULATION: Rollout from the new node
            reward = self.environment.rollout(node)

            # BACKPROPAGATION: Update all nodes along the path
            # print(f"node: {node}, reward: {reward}")
            self.backpropagate(node, reward)

        # Choose the action with the highest visit count from the root
        if not root.children:
            return None  # No moves available
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action
    
    def save_tree(self, tree, filename):
        """Serialize the given object (tree or list of trees) to the specified filename."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree(self, filename):
        """Load a previously saved tree (or list of trees) from the specified filename."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def tree_to_str(self, node, indent=0):
        """
        Recursively convert a tree node and its children into a formatted string.
        Each node is explicitly labeled with its state, visits, reward, and untried actions.
        """
        spacer = ' ' * indent
        # Explicitly label the node
        node_str = (f"{spacer}Node: [State: {node.state}] | Visits: {node.visits} | "
                    f" Reward: {node.reward:.2f} | "
                    f" Untried: {node.untried_actions} | "
                    f" Battery: {node.battery}\n")
        for action, child in node.children.items():
            node_str += f"{spacer}  Action: {action}\n"
            node_str += self.tree_to_str(child, indent + 4)
        return node_str

    def format_trees(self, filename):
        """Load trees from the pickle file and format all trees into a nicely formatted text output."""
        tree_list = self.load_tree(filename)
        formatted_output = ""
        for i, tree in enumerate(tree_list):
            formatted_output += f"--- MCTS Tree for Decision Step {i+1} ---\n"
            formatted_output += self.tree_to_str(tree)
            formatted_output += "\n"
        return formatted_output
    
    def save_formatted_trees_to_file(self, pickle_filename, output_txt_filename):
        """Load the trees from the pickle file, format them, and save to a text file."""
        formatted_trees = self.format_trees(pickle_filename)
        with open(output_txt_filename, "w") as f:
            f.write(formatted_trees)
        print(f"Formatted trees saved to {output_txt_filename}")

    # def load_plot_and_split_tree_text(self, filename):
    #     """Load the formatted tree text from the specified file and split into sections."""
    #     with open(filename, "r") as f:
    #         tree_text = f.read()

    #     # Split the text into sections by decision step.
    #     sections = []
    #     current_section = []
    #     for line in tree_text.splitlines():
    #         if line.startswith('--- MCTS Tree for Decision Step 6 ---'):
    #             if current_section:
    #                 sections.append(current_section)
    #                 current_section = []
    #         if line.strip():
    #             current_section.append(line)
    #     if current_section:
    #         sections.append(current_section)

    #     # Now parse each section (each tree)
    #     parsed_trees = []
    #     for sec in sections:
    #         # We only want the lines starting with "Node:" (ignore "Action:" lines)
    #         node_lines = [line for line in sec if 'Node:' in line]
    #         nodes, edges = self.parse_tree_section(node_lines)
    #         parsed_trees.append((nodes, edges))

    #     # Create subplots (one per decision step)
    #     n_trees = len(parsed_trees)
    #     fig, axes = plt.subplots(n_trees, 1, figsize=(12, 4*n_trees))
    #     if n_trees == 1:
    #         axes = [axes]

    #     for idx, (nodes, edges) in enumerate(parsed_trees):
    #         G = nx.DiGraph()
    #         for node_id, label, indent in nodes:
    #             G.add_node(node_id, label=label)
    #         G.add_edges_from(edges)
    #         ax = axes[idx]
    #         pos = graphviz_layout(G, prog="dot")
    #         node_labels = nx.get_node_attributes(G, 'label')
    #         nx.draw(G, pos, with_labels=False, arrows=True, ax=ax)
    #         nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    #         ax.set_title(f"Decision Step {idx+1}")
    #         ax.axis('off')

    #     plt.tight_layout()
    #     plt.show()
    def load_and_plot_tree(self, filename, decision_step, save_path=None):
        """Load the tree text from file and plot only the tree for the given decision step."""
        with open(filename, "r") as f:
            tree_text = f.read()

        # Build the marker for the desired decision step.
        step_marker = f"--- MCTS Tree for Decision Step {decision_step} ---"
        target_section = []
        capture = False

        for line in tree_text.splitlines():
            if line.startswith(step_marker):
                capture = True
                continue  # Skip the marker line itself.
            # If a new decision step marker is encountered while capturing, break out.
            if capture and line.startswith('--- MCTS Tree for Decision Step'):
                break
            if capture and line.strip():
                target_section.append(line)

        if not target_section:
            print(f"No tree found for decision step {decision_step}")
            return

        # Extract only the lines relevant to the tree (e.g., only lines containing "Node:")
        node_lines = [line for line in target_section if 'Node:' in line]
        nodes, edges = self.parse_tree_section(node_lines)

        # Build the tree graph.
        G = nx.DiGraph()
        for node_id, label, indent in nodes:
            G.add_node(node_id, label=label)
        G.add_edges_from(edges)

        # Plot the tree.
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = graphviz_layout(G, prog="dot")
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=False, arrows=True, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
        ax.set_title(f"Decision Step {decision_step}")
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def parse_tree_section(self, lines):
        # Regex to match a node line and extract the indent, state, visits, reward.
        node_regex = re.compile(r'^( *)(Node: )\[(?:State:\s*(.+?))\]\s*\|\s*Visits:\s*([\d\.]+)\s*\|\s*Reward:\s*([\d\.]+)')

        nodes = []
        edges = []
        stack = []  # list of tuples: (indent, node_id)
        node_id = 0
        
        for line in lines:
            match = node_regex.match(line)
            if match:
                indent = len(match.group(1))
                state = match.group(3).strip()
                visits = match.group(4)
                reward = match.group(5)
                label = f"{state}\nV: {visits}\nR: {reward}"
                current_id = f"n{node_id}"
                nodes.append((current_id, label, indent))
                node_id += 1
                
                # Find parent (last node with indent < current indent)
                while stack and stack[-1][1] >= indent:
                    stack.pop()
                if stack:
                    parent_id = stack[-1][0]
                    edges.append((parent_id, current_id))
                stack.append((current_id, indent))
        return nodes, edges

def uct_select_child(node):
    """Select a child node using the UCT (Upper Confidence Bound) formula."""
    c = 1.0  # exploration constant
    best_score = -float('inf')
    best_child = None
    for child in node.children.values():
        exploit = child.reward / child.visits
        explore = c * math.sqrt(math.log(node.visits) / child.visits)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

# def flatten_state(state):
#     """
#     Flatten a state of the form:
#       (((agent_x, agent_y), battery), sub_objectives)
#     into a list:
#       [agent_x, agent_y, battery, sub_obj_y, sub_obj_x]
      
#     In this example, if a sub-objective is provided, we ignore its x-coordinate
#     and use only its y-coordinate (placing 0 in the final position). If no sub-objective
#     exists, we default both to 0.
    
#     Examples:
#       Input: (((0, 0), 10), ((3, 1),)) -> Output: [0, 0, 10, 1, 0]
#       Input: (((0, 0), 10), ())       -> Output: [0, 0, 10, 0, 0]
#     """
#     # Unpack the state
#     ((agent_x, agent_y), battery), subobjs = state

#     # If there is at least one sub-objective, take the first one.
#     # Here we assume that each sub-objective is given as (x, y),
#     # and we choose to use only its y value.
#     if subobjs and len(subobjs) > 0:
#         _, sub_obj_y = subobjs[0]
#     else:
#         sub_obj_y = 0

#     # For this example, we set the final value to 0.
#     return [agent_x, agent_y, battery, sub_obj_y, 0]
def flatten_state(state, num_subobjs):
    """
    Flatten a state of the form:
        (((agent_x, agent_y), battery), sub_objectives)
    into a list:
        [agent_x, agent_y, battery, flag1, flag2, ..., flag_num_subobjs]

    Here, the flags are defined such that if the state's sub_objectives tuple
    has length k, then the first k flags are 1 and the remaining (num_subobjs - k) flags are 0.
    """
    ((agent_x, agent_y), battery), subobjs = state
    # Determine the number of sub-objectives actually collected in this state.
    k = len(subobjs) if subobjs else 0
    # Create a flag vector of length num_subobjs: first k are 1, rest 0.
    flags = [1 if i < k else 0 for i in range(num_subobjs)]
    return [agent_x, agent_y, battery] + flags

def flatten_states(states):
    """
    Given a list of states, automatically determine the total number of sub-objectives
    from the first state's sub_objectives. If the first state's sub_objectives is non-empty,
    we set num_subobjs = len(first_state[1]) + 1; otherwise, we default to 2.
    
    Returns a list of flattened states.
    """
    if not states:
        return []
    first_state = states[0]
    # If the first state's sub_objectives is non-empty, we assume the total number is one more than its length.
    # otherwise, there are no sub-objectives
    num_subobjs = len(first_state[1]) if first_state[1] else 0
    return [flatten_state(state, num_subobjs) for state in states]

def principal_component_analysis(trajectories, save_path=None):

    stacked_trajectories = []
    for trajectory in trajectories:
        traj_vec = flatten_states(trajectory)
        # import pdb; pdb.set_trace()
        stacked_trajectories.append(traj_vec)
    
    all_states = [state for traj in stacked_trajectories for state in traj]
    # all_states = [state for traj in trajectories for state in traj]
    # import pdb; pdb.set_trace()
    traj_mat = np.vstack(all_states)
    # print(f"Trajectory matrix shape: {traj_mat.shape}")

    # Normalize the trajectory matrix.
    traj_mat = (traj_mat - np.mean(traj_mat, axis=0)) / np.std(traj_mat, axis=0)

    # Perform PCA on the trajectory matrix.
    pca = PCA(n_components=2)
    # pca.fit(traj_mat)
    transformed = pca.fit_transform(traj_mat)
    # print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    # print(f"PCA transformed shape: {transformed.shape}")
    # Plot the PCA results using matplotlib.
    plt.figure(figsize=(10, 8))
    plt.scatter(transformed[:, 0], transformed[:, 1], c='blue', marker='o', alpha=0.5)
    plt.title('PCA of Trajectories')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    