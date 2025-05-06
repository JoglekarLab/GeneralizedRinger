# GeneralizedRinger

Generalized Ringer helps you build your custum protein design workflow from modular pipelines. The workflow is initially designed as the following:

![ringer_generalized2](https://github.com/user-attachments/assets/de5912b6-bf07-47ea-945b-bd5a372b1a99)

### Geometry Scoring Pipeline

The Geometry Scoring Pipeline assigns a score based on Cα–Cα distances between pairs of interacting residues: distances that match a reference helix–helix contact distribution receive higher scores. You can customize what counts as a “contact” by setting:

- `min_distance` (float, default: 5 Å): the minimum Cα–Cα distance to consider a contact  
- `max_distance` (float, default: 10 Å): the maximum Cα–Cα distance to consider a contact


### Design Generator Pipeline

The Design Generator Pipeline first creates a file with the selection of what residues are to be designed. For that selection, the user can configure:

- `max_distance` (float): Cα–Cα distance threshold (Å) to define contacts  
- `cb_distance` (float): Cβ–Cβ distance threshold (Å) to define contacts  
- `extended_design` (bool, default: True): if True, use Cα (`max_distance`) to identify contacts; if False, use Cβ (`cb_distance`)  
- `interface` (string): chains defining the interface (e.g. `"A_BC"`)  
- `fixed_chains` (string): comma‑separated chains to keep unchanged (e.g. `"B,C"`)

> **Note:** To design an entire chain rather than just the interface, set `max_distance` to a very high value (e.g. `100` Å).

After selecting residues, the pipeline runs **ProteinMPNN** to generate designs and outputs a PDB file for each generated design. You can control both the sampling temperature and the number of sequences generated at each temperature by passing a Python dictionary to the `temperature_to_seq_count` parameter. For example:

```python
temperature_to_seq_count = {
    0.15: 35,   # generate 35 sequences at temperature 0.15
    0.25: 95,   # generate 95 sequences at temperature 0.25
    0.30: 120,  # generate 120 sequences at temperature 0.30
    0.35: 120   # generate 120 sequences at temperature 0.35
}

Each key is a sampling temperature and each value is how many sequences to sample at that temperature. This lets you explore designs under different levels of sequence diversity.

### Rosetta Scoring Pipeline

The Rosetta Scoring Pipeline relaxes the redesigned backbones and computes interface energy using Rosetta’s FastRelax protocol. A default XML schema is provided (you can modify it via the standalone `createFastRelaxScript` notebook). The user needs to specify:

- `interface` (string): chains defining the interface (e.g. `"A_BC"`)  
- `repeats` (int, default: 1): number of times to repeat the FastRelax protocol  

The pipeline outputs relaxed PDB files and a csv that includes interface energy scores and other relevant parameters such as the shape complementarity value.

### AlphaFold Scoring Pipeline

The AlphaFold3 Scoring Pipeline (currently recommended to use the AlphaFold3 Scoring Pipeline) runs **AlphaFold** to predict the structure of the top designs filtered by Rosetta scores and computes confidence metrics such as interface PAE, PLDDT and RMSD between design and prediction.
(Before AF used to be the bottleneck, now I think rosetta is the bottleneck).

The user must specify the number or seeds to be used or a provide a list of the specific. Note that for each seed → 5 models are predicted. In addition, users can customize how metrics are computed by choosing exactly which interface to evaluate for PAE (for example, focusing on A–BC contacts and ignoring any B–C interactions) and by selecting which chains to include in the pLDDT calculation (for instance, restricting pLDDT to chain A when only that chain has been designed).

**User parameters:**  
- `num_seeds` (int): number of random seeds to sample 
- `predefined_seeds_list` (list[int], optional): specific seed values to use  


**Outputs include:**  
- CIF files for all predicted models (5 × `num_seeds` per design)  
- Average interface PAE (iPAE)  
- average pLDDT scores for selected chains
- RMSD between the input design and each prediction

### STANDALONE NOTEBOOKS
The notebooks are there to help you through the pipeline. The recommended use of the notebooks is the following:

![ringer_generalized4](https://github.com/user-attachments/assets/94edb0f8-2a02-47f4-98ad-f81b867f1321)




