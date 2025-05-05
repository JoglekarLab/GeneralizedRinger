# GeneralizedRinger

Generalized Ringer helps you build your custum protein design workflow from modular pipelines. The workflow is initially designed as the following:

![ringer_generalized2](https://github.com/user-attachments/assets/de5912b6-bf07-47ea-945b-bd5a372b1a99)

### Geometry Scoring Pipeline

The Geometry Scoring Pipeline assigns a score based on Cα–Cα distances between pairs of interacting residues: distances that match a reference helix–helix contact distribution receive higher scores. You can customize what counts as a “contact” by setting:

- `min_distance` (float, default: 5 Å): the minimum Cα–Cα distance to consider a contact  
- `max_distance` (float, default: 10 Å): the maximum Cα–Cα distance to consider a contact


### Design Generator Pipeline

The Design Generator Pipeline first creates a file with the selection of what residues are to be designed
The Design Generator Pipeline builds sequence designs for a specified interface while optionally keeping certain chains fixed. You can configure:

For the Design Generator Pipeline, the user can select how to select the interface for design and which chains to keep fixed so they are not designed. The user can provide:

- `max_distance` (distance between c alpha to be considered a contact)  
- `cb_distance` (distance between c beta to be considered a contact)  
- `extended_design = True` (if True uses the max_distance between c_alpha to check for contacts, if false uses the c beta)  
- `interface = "A_BC"`  
- `fixed_chains = "B,C"` (which chains not to be designed)

> **Note:** To design a whole chain rather than just the interface, set `max_distance` to a very high number (e.g. 100).


### Design Generator Pipeline

The Design Generator Pipeline first creates a file with the selection of what residues are to be designed. For that selection, the user can configure:

- `max_distance` (float): Cα–Cα distance threshold (Å) to define contacts  
- `cb_distance` (float): Cβ–Cβ distance threshold (Å) to define contacts  
- `extended_design` (bool, default: True): if True, use Cα (`max_distance`) to identify contacts; if False, use Cβ (`cb_distance`)  
- `interface` (string): chains defining the interface (e.g. `"A_BC"`)  
- `fixed_chains` (string): comma‑separated chains to keep unchanged (e.g. `"B,C"`)

> **Note:** To design an entire chain rather than just the interface, set `max_distance` to a very high value (e.g. `100` Å).

After selecting residues, the pipeline runs ProteinMPNN to generate designs and outputs a PDB file for each generated design.  
