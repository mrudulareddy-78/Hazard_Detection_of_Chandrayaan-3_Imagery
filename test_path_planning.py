"""
Quick test script for path planning functionality
"""

import numpy as np
from path_planning import PathPlanner

# Create a simple test hazard map
# Class 0 = Background, Class 1 = Safe (RED), Class 2 = Rocks (GREEN), Class 3 = Crater (YELLOW)
hazard_map = np.ones((100, 100), dtype=int)  # Start with all safe (RED zones)

# Add some obstacles
hazard_map[30:40, 20:80] = 3  # Crater band (YELLOW)
hazard_map[60:70, 30:50] = 2  # Rocky area (GREEN)

print("Testing Path Planning System")
print("=" * 50)
print(f"Map size: {hazard_map.shape}")
print(f"Safe zones (RED): {np.sum(hazard_map == 1)} pixels")
print(f"Rocks (GREEN): {np.sum(hazard_map == 2)} pixels")
print(f"Craters (YELLOW): {np.sum(hazard_map == 3)} pixels")
print()

# Initialize planner
planner = PathPlanner(hazard_map)

# Test positions
start_pos = (10, 10)
goal_pos = (90, 90)

print(f"Start position: {start_pos}")
print(f"Goal position: {goal_pos}")
print()

# Test A* with safety mode
print("Testing A* with safety mode ON...")
path_a_safe, metrics_a_safe = planner.a_star(start_pos, goal_pos, safety_mode=True)

if path_a_safe:
    print(f"✅ Path found!")
    print(f"   - Path length: {len(path_a_safe)} nodes")
    print(f"   - Planning time: {metrics_a_safe['planning_time_ms']:.2f} ms")
    print(f"   - Nodes explored: {metrics_a_safe['nodes_explored']}")
    
    # Calculate risk
    risk_data = planner.calculate_path_risk(path_a_safe)
    print(f"   - Safe percentage: {risk_data['safe_percentage']:.1f}%")
    print(f"   - Risk score: {risk_data['risk_score']:.2f}")
else:
    print(f"❌ No path found")
    print(f"   - Error: {metrics_a_safe.get('error', 'Unknown')}")

print()

# Test A* without safety mode
print("Testing A* with safety mode OFF...")
path_a_unsafe, metrics_a_unsafe = planner.a_star(start_pos, goal_pos, safety_mode=False)

if path_a_unsafe:
    print(f"✅ Path found!")
    print(f"   - Path length: {len(path_a_unsafe)} nodes")
    print(f"   - Planning time: {metrics_a_unsafe['planning_time_ms']:.2f} ms")
    print(f"   - Nodes explored: {metrics_a_unsafe['nodes_explored']}")
    
    # Calculate risk
    risk_data = planner.calculate_path_risk(path_a_unsafe)
    print(f"   - Safe percentage: {risk_data['safe_percentage']:.1f}%")
    print(f"   - Risk score: {risk_data['risk_score']:.2f}")
else:
    print(f"❌ No path found")

print()

# Test RRT*
print("Testing RRT* with safety mode ON...")
path_rrt, metrics_rrt = planner.rrt_star(
    start_pos, goal_pos, 
    max_iterations=1000, 
    safety_mode=True
)

if path_rrt:
    print(f"✅ Path found!")
    print(f"   - Path length: {len(path_rrt)} nodes")
    print(f"   - Planning time: {metrics_rrt['planning_time_ms']:.2f} ms")
    print(f"   - Nodes explored: {metrics_rrt['nodes_explored']}")
    
    # Calculate risk
    risk_data = planner.calculate_path_risk(path_rrt)
    print(f"   - Safe percentage: {risk_data['safe_percentage']:.1f}%")
    print(f"   - Risk score: {risk_data['risk_score']:.2f}")
else:
    print(f"❌ No path found")

print()

# Test find_safe_position
print("Testing automatic safe position finding...")
unsafe_pos = (35, 50)  # This is in the crater (YELLOW)
print(f"Unsafe position: {unsafe_pos} (terrain class: {hazard_map[unsafe_pos]} = CRATER/YELLOW)")

safe_pos = planner.find_safe_position(unsafe_pos, search_radius=30)
print(f"Adjusted safe position: {safe_pos} (terrain class: {hazard_map[safe_pos]} = {'SAFE/RED' if hazard_map[safe_pos] == 1 else 'OTHER'})")

print()
print("=" * 50)
print("Test Complete!")
