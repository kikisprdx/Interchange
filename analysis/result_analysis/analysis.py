# functions for analysis of the results of the intervention experiments

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def load_experiment_data(base_dir):
    all_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    df_list = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")
            
    if not df_list:
        raise ValueError("No CSV files found!!!!!!!!!")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(combined_df)} total intervention experiments.")
    return combined_df

def plot_general_accuracy(df, dataset_name):
    # drop duplicate base examples so  don't overcount
    unique_bases = df.drop_duplicates(subset=['base_i', 'high_node'])
    
    # clculate accuracy per high_node
    accuracy = unique_bases.groupby('high_node')['base_correct_vs_high'].mean()
    accuracy.to_csv(f'general_accuracy_{dataset_name}.csv', header=['accuracy'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    accuracy.plot(kind='bar', ax=ax, color='#5dade2')
    
    ax.set_title("General Model Accuracy per Abstract Variable")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_intervention_success_rate(df, dataset_name):
    # filter -> base model was correct n the intervention is impactful (should change output)
    mask = (df['base_correct_vs_high'] == 1) & (df['high_changed'] == 1)
    valid_interventions = df[mask]
    
    # calculate success rate grouped by abstract variable and layer
    success_rates = valid_interventions.groupby(['high_node', 'low_node'])['interchange_success'].mean().unstack()
    
    # sort columns to ensure layers are in order (so : bert_layer_0, bert_layer_1...)
    sorted_columns = sorted(success_rates.columns, key=lambda x: int(x.split('_')[-1]))
    success_rates = success_rates[sorted_columns]
    success_rates.to_csv(f'intervention_success_rates_{dataset_name}.csv')

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(success_rates.index)))
    
    plot_columns = [col.split('_')[-1] for col in success_rates.columns]
    
    for i, high_node in enumerate(success_rates.index):
        ax.plot(plot_columns, success_rates.loc[high_node], marker='o', color=colors[i], label=high_node)
  
    ax.legend(title="Abstract Variable", bbox_to_anchor=(1.05, 1), loc='center left')
    ax.set_title("Intervention Success Rate by Layer")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Neural Model Layer")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Abstract Variable")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

'''
def calculate_clique_sizes(df):
    # correct base predictions
    correct_df = df[df['base_correct_vs_high'] == 1]
    
    clique_records = []
    
    # total n of unique inputs (needed to calculate percentage)
    total_examples = df['base_i'].nunique()
    
    groups = correct_df.groupby(['high_node', 'low_node'])
    
    for (high_node, low_node), group in groups:
        # get edges where intervention was successful
        successful_edges = group[group['interchange_success'] == 1]
        
        # build directed graph
        G = nx.DiGraph()
        edges = list(zip(successful_edges['base_i'], successful_edges['source_i']))
        G.add_edges_from(edges)
        
        # the paper seems to require both (e_i, e_j) and (e_j, e_i) to be successful to form an edge (???)
        G_undirected = nx.Graph()
        for u, v in G.edges():
            if G.has_edge(v, u):
                G_undirected.add_edge(u, v)
                
        # find all cliques
        cliques = list(nx.find_cliques(G_undirected))
        
        if cliques:
            max_clique_size = max(len(c) for c in cliques)
        else:
            max_clique_size = 0
            
        clique_records.append({
            'high_node': high_node,
            'low_node': low_node,
            'max_clique_size': max_clique_size,
            'clique_percentage': max_clique_size / total_examples
        })
        
    return pd.DataFrame(clique_records)
'''

import pandas as pd
import networkx as nx
import copy

def paper_find_cliques(G, causal_edges, alpha):
    """replication of the greedy clique heuristic from the paper's clique_analysis.py"""
    original_G = G
    cliques = []
    
    while True:
        G = copy.deepcopy(original_G)
        if len(G.nodes()) == 0:
            break
            
        # the paper's math for a full clique requires self-loops: n * (n+1) / 2
        while float(len(G.nodes()) * (len(G.nodes()) + 1) * 0.5) != float(len(G.edges())):
            edge_dict = {node: set() for node in G.nodes()}
            causal_edge_dict = {node: 0 for node in G.nodes()}
            
            for edge in G.edges():
                edge_dict[edge[0]].add(edge[1])
                edge_dict[edge[1]].add(edge[0])
                
            for edge in causal_edges:
                if G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0]):
                    causal_edge_dict[edge[1]] += 1
                    causal_edge_dict[edge[0]] += 1
                    
            # sort nodes by degree (fewest edges first)
            edge_counts = sorted(edge_dict.items(), key=lambda item: len(item[1]))
            causal_edge_counts = sorted(causal_edge_dict.items(), key=lambda item: item[1])
            
            # the alpha threshold heuristic logic
            if causal_edge_counts[-1][1] - causal_edge_counts[0][1] >= alpha:
                G.remove_node(causal_edge_counts[0][0])
            else:
                G.remove_node(edge_counts[0][0])
                
        new_clique = set(G.nodes())
        for node in G.nodes():
            original_G.remove_node(node)
        cliques.append(new_clique)
        
    final_result = []
    # filter out any cliques that don't contain causal edges
    for clique in cliques:
        seen = False
        for node in copy.copy(clique):
            for node2 in clique:
                if ((node, node2) in causal_edges or (node2, node) in causal_edges) and not seen:
                    final_result.append(clique)
                    seen = True
                    
    return final_result


def calculate_clique_sizes(df, alpha=1):
    correct_df = df[df['base_correct_vs_high'] == 1]
    clique_records = []
    
    total_examples = df['base_i'].nunique()
    groups = correct_df.groupby(['high_node', 'low_node'])
    
    for (high_node, low_node), group in groups:
        # vectorized extraction of directed edges
        success_df = group[group['interchange_success'] == 1]
        dir_edges = set(zip(success_df['base_i'], success_df['source_i']))
        
        causal_df = success_df[success_df['high_changed'] == 1]
        dir_causal = set(zip(causal_df['base_i'], causal_df['source_i']))
        
        G = nx.Graph()
        causal_edges = set()
        
        # get all unique nodes in this group
        all_nodes = set(group['base_i']).union(set(group['source_i']))
        
        # initialize nodes and CRUCIAL self-loops for the paper's math
        for u in all_nodes:
            G.add_node(u)
            G.add_edge(u, u) 
            
        # add bi-directional undirected edges
        for u, v in dir_edges:
            if (v, u) in dir_edges:
                G.add_edge(u, v)
                
        # add bi-directional causal edges
        for u, v in dir_causal:
            if (v, u) in dir_causal:
                causal_edges.add((u, v))
                causal_edges.add((v, u)) 
                
        # use the paper's exact heuristic search
        cliques = paper_find_cliques(G, causal_edges, alpha)
        
        if cliques:
            max_clique_size = max(len(c) for c in cliques)
        else:
            max_clique_size = 0
            
        clique_records.append({
            'high_node': high_node,
            'low_node': low_node,
            'max_clique_size': max_clique_size,
            'clique_percentage': max_clique_size / total_examples
        })
        
    return pd.DataFrame(clique_records)

def plot_clique_sizes(df, dataset_name):
    clique_df = calculate_clique_sizes(df)
    
    clique_pivot = clique_df.pivot(index='high_node', columns='low_node', values='clique_percentage')
    
    # sort columns
    sorted_columns = sorted(clique_pivot.columns, key=lambda x: int(x.split('_')[-1]))
    clique_pivot = clique_pivot[sorted_columns]
    clique_pivot.to_csv(f'clique_sizes_pivoted_{dataset_name}.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(clique_pivot.columns))
    jitter_amount = 0.02
    
    for i, high_node in enumerate(clique_pivot.index):
        if high_node == 'sign':
            jitter = -jitter_amount
        elif high_node == 'result':
            jitter = jitter_amount
        else:
            jitter = 0
        ax.plot(x_positions + jitter, clique_pivot.loc[high_node], marker='o', linestyle='-', label=high_node)
        
    ax.set_xticks(x_positions)
    ax.set_xticklabels([col.split('_')[-1] for col in clique_pivot.columns])
    
    ax.set_title("Maximum Clique Size by Layer")
    ax.set_ylabel("Clique Size (% of total examples)")
    ax.set_xlabel("Neural Model Layer")
    # change location to outside of plot
    ax.legend(title="Abstract Variable", loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_outcome_breakdown(df, dataset_name):
    # pick one layer to visualize as an example (e.g., Layer 9 for the 'sign' variable)
    layer_df = df[(df['low_node'] == 'bert_layer_9') & (df['high_node'] == 'sign')].copy()
    
    # categorise every row into one of the 5 categories used in the paper's notebook
    def categorize(row):
        if row['base_correct_vs_high'] == 0:
            return "Base Incorrect"
        if row['high_changed'] == 1 and row['interchange_success'] == 1:
            return "Causal, Successful"
        if row['high_changed'] == 1 and row['interchange_success'] == 0:
            return "Causal, Failed"
        if row['high_changed'] == 0 and row['interchange_success'] == 1:
            return "Non-Causal, Successful"
        if row['high_changed'] == 0 and row['interchange_success'] == 0:
            return "Non-Causal, Failed"
            
    layer_df['category'] = layer_df.apply(categorize, axis=1)
    
    # count the outcomes
    counts = layer_df['category'].value_counts(normalize=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind='bar', color=['red', 'green', 'blue', 'yellow', 'gray'], ax=ax)
    
    ax.set_title("Full Breakdown of Intervention Outcomes (Layer 9, Sign Variable)")
    ax.set_ylabel("Proportion of Total Examples")
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
                    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
