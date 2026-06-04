**subcommands:**

```bash
# initialise a new queue
python interchange_manager.py setup -d experiments.csv -m model.pt -i data/

# populate with jobs (iterates high nodes × layers × num_inputs)
python interchange_manager.py add -d experiments.csv -t bert -m model.pt -o results/ -n 500

# run interchange experiments (dispatches worker scripts)
python interchange_manager.py run -d experiments.csv -i python interchange.py

# mark completed interchange jobs as ready for graph analysis
python interchange_manager.py add_graph -d experiments.csv -a 100 --all_rows

# run graph/clique analysis
python interchange_manager.py analyze_graph -d experiments.csv -i python graph_analysis.py

# query queue state
python interchange_manager.py query -d experiments.csv -s 0

# manually update status
python interchange_manager.py update_status -d experiments.csv -i 3 4 5 -s 0
```
