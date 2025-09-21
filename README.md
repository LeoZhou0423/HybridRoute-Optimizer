HybridRoute-Optimizer
📜 License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

You are free to:
✅ Use, study, and run the software freely

✅ Modify and extend functionality

✅ Distribute original or modified versions

You must:
📝 Provide explicit attribution to the original author

🔓 Keep open source: Any derivative works must use the same license

🛡️ No patent claims: You may not patent this software or its derivatives

Full license text:
See the LICENSE file or visit the GNU official website

⚠️ Important Notes
Any derivative works based on this project must:

Clearly attribute the original project URL and author

Be open-sourced under the same GPL-3.0 license

Not file patents or initiate patent litigation related to this technology

Violation of license terms may lead to legal consequences

Project Overview
RoutePlanner is a powerful intelligent path planning system focused on solving the Traveling Salesman Problem (TSP) and constrained route optimization problems. The system integrates multiple optimization algorithms, Amap API, and an interactive user interface, providing a complete solution from coordinate input to optimal route visualization.

Core Features
1. Algorithm Optimization Engine
Simulated Annealing Algorithm: Efficiently solves large-scale path optimization problems

Exact Algorithm: Provides optimal solutions for small-scale problems (≤12 points)

Local Search Optimization: Incorporates 2-opt neighborhood search to further improve path quality

Distance Caching Mechanism: Enhances performance in repetitive calculation scenarios

2. Map Integration
Amap API: Retrieves actual road distances, travel times, and detailed routes

Interactive Map Visualization: Intuitively displays candidate points and optimal paths

Multi-platform Support: Compatible with standalone browser display and PyQt integrated display

3. User Interface
Interactive Control Panel: Supports parameter adjustment, optimization target selection, and result display

Data Import: Reads starting points, waypoints, and destination data from text files

Multi-objective Optimization: Supports path optimization based on straight-line distance, driving distance, and driving time

Project Structure
text
├── optimized_route_planner.py # Main program file, containing complete path planning system
├── amap_api.py               # Amap API client
├── map.py                    # Map display module
├── plan.py                   # Basic path planning algorithms
├── points.txt                # Coordinate data file (example)
├── README.md                 # Project documentation
└── icon/                     # Icon resources directory
Technology Stack
Python 3.x: Core development language

tkinter: Builds interactive user interface

matplotlib: Data visualization

requests: HTTP request processing (Amap API calls)

PyQt5 (optional): Advanced GUI components and web view integration

Quick Start
Prerequisites
Python 3.6 or higher

Install necessary dependencies:

bash
pip install matplotlib requests
(Optional) For PyQt integration, install:

bash
pip install PyQt5
Usage
Prepare Data File
Create a points.txt file in the project directory with the following format:

text
[starts]
# Starting point coordinates (longitude,latitude)
116.397428,39.90923

[waypoints]
# Waypoint coordinates
116.404,39.915
116.417,39.908

[ends]
# Destination coordinates
116.473168,39.993015
Configure Amap API
Set your API key in amap_api.py and map.py files:

python
API_KEY = "Your_Amap_API_Key"
Run the Program

bash
python optimized_route_planner.py
Use the Interface for Path Planning

Load data file

Select optimization target (straight-line distance/driving distance/driving time)

Adjust algorithm parameters

Click "Start Planning" button

View and compare different route solutions

Algorithm Parameters Description
Parameter	Description	Default Value	Recommended Range
Population Size	Genetic algorithm population scale	100	50-200
Cooling Rate	Temperature descent rate for simulated annealing	0.995	0.95-0.999
Initial Temperature	Initial temperature for simulated annealing	10000.0	1000-100000
Maximum Iterations	Maximum number of algorithm iterations	100000	10000-1000000
Enable Cache	Whether to enable distance calculation caching	True	-
Local Search	Whether to enable 2-opt local search	True	-
Performance Optimization
For large-scale problems (>20 waypoints), recommended to use simulated annealing algorithm with distance caching enabled

For small-scale problems (≤12 waypoints), the system automatically switches to exact algorithm

Balance solution speed and path quality by adjusting cooling rate and initial temperature

Notes
Using Amap API requires a valid API key, some features may have usage restrictions

Path planning results are for reference only; actual driving routes may be affected by real-time traffic conditions

Data file format must strictly follow specifications to avoid parsing errors

Large data processing may consume significant system resources; ensure your device meets requirements

Development & Extension
This project is designed with good extensibility. You can:

Add new optimization algorithms or heuristic strategies

Integrate other map service providers

Develop more visualization features and analysis tools

Build dedicated business application scenarios

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Amap API for providing geographic information and path planning services

matplotlib for data visualization support

tkinter and PyQt for GUI development frameworks

