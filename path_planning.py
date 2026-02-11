"""
Path Planning Module for Lunar Rover Navigation
Implements A* and RRT* algorithms using hazard detection maps
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
import random
import time


class Node:
    """Node class for A* pathfinding"""
    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance from start
        self.h = 0  # Heuristic distance to goal
        self.f = 0  # Total cost
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash(self.position)


class RRTNode:
    """Node class for RRT* pathfinding"""
    def __init__(self, position: Tuple[float, float]):
        self.position = position
        self.parent = None
        self.cost = 0.0


class PathPlanner:
    """
    Advanced path planning for lunar rover navigation
    Supports A* and RRT* algorithms with hazard-aware cost functions
    """
    
    def find_safe_position(self, preferred_pos: Tuple[int, int], search_radius: int = 30) -> Tuple[int, int]:
        """Find nearest safe position if preferred position is unsafe"""
        row, col = preferred_pos
        
        # Check if preferred position is already safe
        if self.is_valid(preferred_pos) and self.hazard_map[row, col] == 1:
            return preferred_pos
        
        # Search in expanding circles
        for radius in range(1, search_radius):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    new_pos = (row + dr, col + dc)
                    if self.is_valid(new_pos) and self.hazard_map[new_pos[0], new_pos[1]] == 1:
                        return new_pos
        
        # If no safe position found, return any valid position
        for radius in range(1, search_radius):
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    new_pos = (row + dr, col + dc)
                    if self.is_valid(new_pos) and self.hazard_map[new_pos[0], new_pos[1]] != 3:
                        return new_pos
        
        # Last resort: return preferred position even if unsafe
        return preferred_pos if self.is_valid(preferred_pos) else (self.height//2, self.width//2)
    
    def __init__(self, hazard_map: np.ndarray):
        """
        Initialize path planner with hazard map
        
        Args:
            hazard_map: 2D array where:
                - 0 = Background (black) - high cost
                - 1 = Safe (RED) - low cost, preferred
                - 2 = Rocks (GREEN) - medium cost, highly textured
                - 3 = Crater (YELLOW) - high cost, avoid
        """
        self.hazard_map = hazard_map
        self.height, self.width = hazard_map.shape
        
        # Cost mapping: Higher values = more dangerous
        # Class 1 (RED) = Safe, Class 2 (GREEN) = Rocks, Class 3 (YELLOW) = Craters
        self.terrain_costs = {
            0: 10,    # Background (black) - moderate cost
            1: 1,     # Safe (RED) - preferred, lowest cost
            2: 5,     # Rocks (GREEN) - traversable with caution
            3: 20,    # Crater (YELLOW) - avoid, highest cost
        }
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_terrain_cost(self, position: Tuple[int, int]) -> float:
        """Get traversal cost for a position"""
        if not self.is_valid(position):
            return float('inf')
        terrain_type = self.hazard_map[position[0], position[1]]
        return self.terrain_costs.get(terrain_type, 100)
    
    def is_valid(self, position: Tuple[int, int]) -> bool:
        """Check if position is within bounds"""
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_safe_traversable(self, position: Tuple[int, int], safety_threshold: int = 3) -> bool:
        """Check if position is safe for traversal
        
        Class 1 (RED) = Safe, Class 2 (GREEN) = Rocks, Class 3 (YELLOW) = Craters
        """
        if not self.is_valid(position):
            return False
        terrain = self.hazard_map[position[0], position[1]]
        # Relaxed mode: avoid only yellow craters (class 3)
        if safety_threshold >= 3:  # Relaxed mode
            return terrain != 3  # Avoid yellow craters only
        elif safety_threshold >= 2:  # Moderate mode
            return terrain in [1, 2]  # RED (safe) and GREEN (rocks) OK
        else:  # Strict mode
            return terrain == 1  # Only RED (safe) zones
    
    def get_neighbors(self, position: Tuple[int, int], diagonal: bool = True) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        row, col = position
        neighbors = []
        
        # 4-directional movement
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Add diagonal movements
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def a_star(self, 
               start: Tuple[int, int], 
               goal: Tuple[int, int],
               safety_mode: bool = True) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
        """
        A* pathfinding algorithm with terrain-aware cost
        
        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            safety_mode: If True, avoid craters; if False, allow riskier paths
        
        Returns:
            path: List of positions from start to goal (or None if no path)
            metrics: Dictionary with planning metrics
        """
        start_time = time.time()
        
        if not self.is_valid(start) or not self.is_valid(goal):
            return None, {"error": "Invalid start or goal position"}
        
        open_set = []
        start_node = Node(start)
        start_node.g = 0
        start_node.h = self.heuristic(start, goal)
        start_node.f = start_node.g + start_node.h
        
        heapq.heappush(open_set, start_node)
        
        closed_set = set()
        open_dict = {start: start_node}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current.position in closed_set:
                continue
            
            # Goal reached
            if current.position == goal:
                path = []
                node = current
                while node:
                    path.append(node.position)
                    node = node.parent
                path.reverse()
                
                planning_time = time.time() - start_time
                total_cost = current.g
                
                metrics = {
                    "algorithm": "A*",
                    "nodes_explored": nodes_explored,
                    "path_length": len(path),
                    "total_cost": total_cost,
                    "planning_time_ms": planning_time * 1000,
                    "success": True
                }
                
                return path, metrics
            
            closed_set.add(current.position)
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                
                # Skip unsafe terrain in safety mode (more lenient now)
                if safety_mode and not self.is_safe_traversable(neighbor_pos, safety_threshold=3):
                    continue
                
                # Calculate cost with terrain consideration
                terrain_cost = self.get_terrain_cost(neighbor_pos)
                move_cost = self.heuristic(current.position, neighbor_pos)
                tentative_g = current.g + move_cost * terrain_cost
                
                neighbor_node = open_dict.get(neighbor_pos)
                
                if neighbor_node is None:
                    neighbor_node = Node(neighbor_pos, current)
                    neighbor_node.g = tentative_g
                    neighbor_node.h = self.heuristic(neighbor_pos, goal)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    
                    heapq.heappush(open_set, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node
                
                elif tentative_g < neighbor_node.g:
                    neighbor_node.parent = current
                    neighbor_node.g = tentative_g
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    heapq.heappush(open_set, neighbor_node)
        
        # No path found
        planning_time = time.time() - start_time
        metrics = {
            "algorithm": "A*",
            "nodes_explored": nodes_explored,
            "planning_time_ms": planning_time * 1000,
            "success": False,
            "error": "No valid path found"
        }
        
        return None, metrics
    
    def rrt_star(self,
                 start: Tuple[int, int],
                 goal: Tuple[int, int],
                 max_iterations: int = 2000,
                 step_size: float = 5.0,
                 goal_radius: float = 10.0,
                 rewire_radius: float = 15.0,
                 safety_mode: bool = True) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
        """
        RRT* pathfinding algorithm with terrain-aware sampling
        
        Args:
            start: Starting position
            goal: Goal position
            max_iterations: Maximum sampling iterations
            step_size: Maximum distance to extend tree
            goal_radius: Distance threshold to consider goal reached
            rewire_radius: Radius for rewiring optimization
            safety_mode: Avoid dangerous terrain
        
        Returns:
            path: List of positions from start to goal (or None)
            metrics: Planning metrics
        """
        start_time = time.time()
        
        if not self.is_valid(start) or not self.is_valid(goal):
            return None, {"error": "Invalid start or goal position"}
        
        # Initialize tree with start node
        tree = [RRTNode(start)]
        goal_node = None
        
        nodes_explored = 0
        
        for iteration in range(max_iterations):
            # Bias sampling toward goal (10% of the time)
            if random.random() < 0.1:
                sample_pos = goal
            else:
                # Random sampling in map bounds
                sample_pos = (
                    random.randint(0, self.height - 1),
                    random.randint(0, self.width - 1)
                )
            
            # Skip if unsafe terrain (more lenient)
            if safety_mode and not self.is_safe_traversable(sample_pos, safety_threshold=3):
                continue
            
            # Find nearest node in tree
            nearest_node = min(tree, key=lambda n: self.heuristic(n.position, sample_pos))
            
            # Steer toward sample
            direction = np.array(sample_pos) - np.array(nearest_node.position)
            distance = np.linalg.norm(direction)
            
            if distance == 0:
                continue
            
            direction = direction / distance
            step = min(step_size, distance)
            
            new_pos = (
                int(nearest_node.position[0] + direction[0] * step),
                int(nearest_node.position[1] + direction[1] * step)
            )
            
            # Validate new position
            if not self.is_valid(new_pos):
                continue
            
            if safety_mode and not self.is_safe_traversable(new_pos, safety_threshold=3):
                continue
            
            # Check collision-free path
            if not self._is_collision_free(nearest_node.position, new_pos, safety_mode):
                continue
            
            nodes_explored += 1
            
            # Create new node
            new_node = RRTNode(new_pos)
            terrain_cost = self.get_terrain_cost(new_pos)
            new_node.cost = nearest_node.cost + self.heuristic(nearest_node.position, new_pos) * terrain_cost
            new_node.parent = nearest_node
            
            # RRT* rewiring: find better parent within radius
            for node in tree:
                if self.heuristic(node.position, new_pos) <= rewire_radius:
                    potential_cost = node.cost + self.heuristic(node.position, new_pos) * terrain_cost
                    if potential_cost < new_node.cost and self._is_collision_free(node.position, new_pos, safety_mode):
                        new_node.parent = node
                        new_node.cost = potential_cost
            
            tree.append(new_node)
            
            # Rewire nearby nodes
            for node in tree:
                if node == new_node:
                    continue
                if self.heuristic(node.position, new_pos) <= rewire_radius:
                    potential_cost = new_node.cost + self.heuristic(new_pos, node.position) * self.get_terrain_cost(node.position)
                    if potential_cost < node.cost and self._is_collision_free(new_pos, node.position, safety_mode):
                        node.parent = new_node
                        node.cost = potential_cost
            
            # Check if goal reached
            if self.heuristic(new_pos, goal) <= goal_radius:
                goal_node = new_node
                # Continue searching for better paths (RRT* optimization)
        
        # Extract path if goal found
        if goal_node:
            path = []
            node = goal_node
            while node:
                path.append(node.position)
                node = node.parent
            path.reverse()
            
            planning_time = time.time() - start_time
            
            metrics = {
                "algorithm": "RRT*",
                "nodes_explored": nodes_explored,
                "path_length": len(path),
                "total_cost": goal_node.cost,
                "planning_time_ms": planning_time * 1000,
                "iterations": max_iterations,
                "success": True
            }
            
            return path, metrics
        
        # No path found
        planning_time = time.time() - start_time
        metrics = {
            "algorithm": "RRT*",
            "nodes_explored": nodes_explored,
            "planning_time_ms": planning_time * 1000,
            "iterations": max_iterations,
            "success": False,
            "error": "No valid path found"
        }
        
        return None, metrics
    
    def _is_collision_free(self, pos1: Tuple[int, int], pos2: Tuple[int, int], safety_mode: bool) -> bool:
        """Check if straight-line path between two positions is collision-free"""
        # Bresenham's line algorithm
        x0, y0 = pos1
        x1, y1 = pos2
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if not self.is_valid((x0, y0)):
                return False
            
            if safety_mode and not self.is_safe_traversable((x0, y0), safety_threshold=2):
                return False
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def smooth_path(self, path: List[Tuple[int, int]], iterations: int = 5) -> List[Tuple[int, int]]:
        """Smooth path using iterative shortcutting"""
        if not path or len(path) <= 2:
            return path
        
        smoothed = path.copy()
        
        for _ in range(iterations):
            i = 0
            while i < len(smoothed) - 2:
                # Try to shortcut from i to i+2 or beyond
                for j in range(len(smoothed) - 1, i + 1, -1):
                    if self._is_collision_free(smoothed[i], smoothed[j], safety_mode=True):
                        # Remove intermediate points
                        smoothed = smoothed[:i+1] + smoothed[j:]
                        break
                i += 1
        
        return smoothed
    
    def calculate_path_risk(self, path: List[Tuple[int, int]]) -> dict:
        """Calculate risk metrics for a planned path.

        Samples every cell along each segment (Bresenham) so crossings over
        rocks/craters are counted even when waypoints are on safe pixels.
        """
        if not path:
            return {"error": "No path provided"}

        terrain_types = {0: 0, 1: 0, 2: 0, 3: 0}
        total_cost = 0.0
        sampled_cells = 0

        def _cells_on_line(a: Tuple[int, int], b: Tuple[int, int]):
            """Yield all grid cells along the line from a to b (inclusive)."""
            x0, y0 = a
            x1, y1 = b
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while True:
                yield (x0, y0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

        # Sample along each segment
        for i in range(len(path) - 1):
            for cell in _cells_on_line(path[i][0:2], path[i + 1][0:2]):
                if not self.is_valid(cell):
                    continue
                terrain = self.hazard_map[cell[0], cell[1]]
                terrain_types[terrain] += 1
                total_cost += self.get_terrain_cost(cell)
                sampled_cells += 1

        # If path has a single node, still account for it
        if len(path) == 1:
            cell = path[0]
            terrain = self.hazard_map[cell[0], cell[1]]
            terrain_types[terrain] += 1
            total_cost += self.get_terrain_cost(cell)
            sampled_cells += 1

        total_pixels = max(sampled_cells, 1)

        risk_score = (
            (terrain_types[3] * 200 +  # Craters
             terrain_types[2] * 50 +   # Rocks
             terrain_types[0] * 100)   # Background
            / total_pixels
        )

        return {
            "total_cost": total_cost,
            "risk_score": risk_score,
            "safe_pixels": terrain_types[1],
            "rock_pixels": terrain_types[2],
            "crater_pixels": terrain_types[3],
            "background_pixels": terrain_types[0],
            "safe_percentage": (terrain_types[1] / total_pixels) * 100,
            "hazard_percentage": ((terrain_types[2] + terrain_types[3]) / total_pixels) * 100
        }