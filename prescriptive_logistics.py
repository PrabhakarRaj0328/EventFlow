import pulp
import networkx as nx
import pandas as pd

def allocate_resources(risk_predictions, total_personnel, total_barricades):
    """
    Sets up an Integer Linear Programming (ILP) problem to maximize resource coverage
    across geographic zones weighted by the predicted incident risk.
    """
    # Initialize ILP problem
    prob = pulp.LpProblem("Resource_Allocation", pulp.LpMaximize)
    
    zones = list(risk_predictions.keys())
    
    # Decision variables for personnel and barricades assigned to each zone (Integers)
    P = pulp.LpVariable.dicts("Personnel", zones, lowBound=0, cat='Integer')
    B = pulp.LpVariable.dicts("Barricades", zones, lowBound=0, cat='Integer')
    
    # Objective: Maximize sum of assigned resources weighted by zone risk
    prob += pulp.lpSum([risk_predictions[z] * (P[z] + B[z]) for z in zones]), "Maximize_Weighted_Coverage"
    
    # Constraints: Total allocated resources cannot exceed total available
    prob += pulp.lpSum([P[z] for z in zones]) <= total_personnel, "Total_Personnel_Constraint"
    prob += pulp.lpSum([B[z] for z in zones]) <= total_barricades, "Total_Barricades_Constraint"
    
    # Additional practical constraint: Prevent all resources from dumping into a single highest-risk zone
    # Setting an arbitrary capacity limit per zone
    max_personnel_per_zone = 10
    max_barricades_per_zone = 10
    
    for z in zones:
        prob += P[z] <= max_personnel_per_zone, f"Max_Personnel_{z}"
        prob += B[z] <= max_barricades_per_zone, f"Max_Barricades_{z}"

    # Solve the problem
    prob.solve()
    
    # Extract results into a DataFrame
    allocation_data = []
    for z in zones:
        allocation_data.append({
            'Zone': z,
            'Predicted_Risk': risk_predictions[z],
            'Allocated_Personnel': int(P[z].varValue) if P[z].varValue is not None else 0,
            'Allocated_Barricades': int(B[z].varValue) if B[z].varValue is not None else 0
        })
        
    allocation_df = pd.DataFrame(allocation_data)
    # Sort by allocated resources for readability
    allocation_df = allocation_df.sort_values(by='Allocated_Personnel', ascending=False).reset_index(drop=True)
    return allocation_df


def initialize_road_network():
    """Initializes a dummy road network graph based on junctions and travel times."""
    G = nx.Graph()
    
    # Add dummy edges with baseline travel times (minutes)
    edges = [
        ("Junction_A", "Junction_B", 5),
        ("Junction_A", "Junction_C", 10),
        ("Junction_B", "Junction_D", 8),
        ("Junction_C", "Junction_D", 4),
        ("Junction_B", "Junction_E", 15),
        ("Junction_D", "Junction_E", 6)
    ]
    
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
        
    return G

def calculate_diversion_plan(G, source, destination, high_risk_junction, penalty=20):
    """
    Recalculates shortest path by dynamically adding a travel time penalty 
    to all edges connected to a predicted high-risk junction.
    """
    # Create a copy of the graph to avoid permanently altering baseline weights
    G_mod = G.copy()
    
    # Apply penalty to edges connected to high_risk_junction
    if high_risk_junction in G_mod.nodes:
        for neighbor in list(G_mod.neighbors(high_risk_junction)):
            G_mod[high_risk_junction][neighbor]['weight'] += penalty
            
    # Recalculate shortest path using Dijkstra's algorithm
    try:
        optimal_path = nx.shortest_path(G_mod, source=source, target=destination, weight='weight')
        optimal_travel_time = nx.shortest_path_length(G_mod, source=source, target=destination, weight='weight')
        
        # Calculate original optimal route for comparison
        original_time = nx.shortest_path_length(G, source=source, target=destination, weight='weight')
        original_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
        
        plan = (
            f"--- Diversion Plan ---\n"
            f"Source: {source} | Destination: {destination}\n"
            f"High-Risk Junction Avoided: {high_risk_junction} (Penalty: +{penalty} mins)\n"
            f"Original Route: {' -> '.join(original_path)} ({original_time} mins)\n"
            f"Recommended Diversion: {' -> '.join(optimal_path)}\n"
            f"Estimated Travel Time: {optimal_travel_time} mins\n"
        )
        return plan
    except nx.NetworkXNoPath:
        return f"No valid path found between {source} and {destination}."


if __name__ == "__main__":
    # --- Task 1: Test Resource Allocation ILP ---
    # Dummy predictive risks sourced hypothetically from the XGBoost output
    dummy_predicted_risks = {
        'Zone_North': 0.85,
        'Zone_South': 0.20,
        'Zone_East': 0.65,
        'Zone_West': 0.40
    }
    
    print("Running Resource Optimization ILP...")
    df_alloc = allocate_resources(risk_predictions=dummy_predicted_risks, total_personnel=25, total_barricades=15)
    print("\nResource Allocation Dataframe:")
    print(df_alloc.to_string())
    
    print("\n" + "="*50 + "\n")
    
    # --- Task 2: Test Dynamic Diversion Routing ---
    print("Running Dynamic Diversion Routing...")
    road_network = initialize_road_network()
    
    # Junction_B is identified by the model as a high incident risk hotspot
    diversion_output = calculate_diversion_plan(
        G=road_network,
        source="Junction_A",
        destination="Junction_E",
        high_risk_junction="Junction_B",
        penalty=30 # Add 30 mins to avoid Junction_B
    )
    
    print(diversion_output)
