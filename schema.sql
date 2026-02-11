CREATE DATABASE IF NOT EXISTS hazard_detection;

USE hazard_detection;

CREATE TABLE IF NOT EXISTS telemetry (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(32) NOT NULL,
    safe FLOAT NOT NULL,
    rocks FLOAT NOT NULL,
    crater FLOAT NOT NULL,
    source VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS path_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(32) NOT NULL,
    image_name VARCHAR(255),
    algorithm VARCHAR(64) NOT NULL,
    safety_mode BOOLEAN NOT NULL,
    start_row INT NOT NULL,
    start_col INT NOT NULL,
    goal_row INT NOT NULL,
    goal_col INT NOT NULL,
    planning_time_ms FLOAT,
    nodes_explored INT,
    path_length INT,
    total_cost FLOAT,
    safe_percentage FLOAT,
    risk_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
