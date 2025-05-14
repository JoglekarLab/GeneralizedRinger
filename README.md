# GeneralizedRinger

### Easy run
You can run `main.py` directly using the `run_main.sh` script. The recommended workflow is the following:

1. Open `run_main.sh` and modify the `radius` and `symmetry` parameters as needed.  
2. Run the script once using:
   ```bash
   sbatch run_main.sh
   ```
   Run the script once. It will likely fail on the first attempt, but this is expected. It will still create the required folder structure.
3. Add the necessary structure files to the generated folders.
4. Re-run the script to complete the process.

### OVERVIEW 

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
    0.15: 35,   # generates 35 sequences at temperature 0.15
    0.25: 95,   # generates 95 sequences at temperature 0.25 ...
    0.30: 120,  
    0.35: 120   
}
```

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

The following are user parameters related to the multiple sequence alignment for the AF3 prediction. Note that you can either not provide pre-computed MSAs for a chain or you can provide them in the form of a json `msas_path`. That json can be obtained from a previous run of AF3 in with the corresponding standalone notebook. If a previous MSA is provided it will be an unpairedMsa. In addition, the user can also provide pre-computed templates to be used for the prediction.

- `msas_path` (str, optional)  
  Path to a JSON file containing per‑chain entries under `"sequences"`.  
  Each entry must include:  
  - `"unpairedMsa"`: FASTA text (e.g. `">query\n…\n"`)  
  - `"templates"`: list of template specs  
  _Example:_ `../0_Inputs/msa_inputs/7b1f_data.json`

- `msa_chains` (list[str] or False, default: False)  
  If set to a list of chain IDs (e.g. `['B','C']`), those chains’ `unpairedMsa` fields will be replaced by the matching entries from your JSON.  
  Chains not listed fall back to `>query\n…\n`.

- `template_chains` (list[str] or False, default: False)  
  If set to a list of chain IDs (e.g. `['B','C']`), those chains’ `templates` lists will be populated from your JSON.  
  Chains not listed get an empty template list.

> **Note:** If `msas_path` is omitted or either `_chains` param is `False`, **all** chains use only their query sequence for MSA and have no templates.

**Outputs include:**  
- CIF files for all predicted models (5 × `num_seeds` per design)  
- Average interface PAE (iPAE)  
- average pLDDT scores for selected chains
- RMSD between the input design and each prediction

### STANDALONE NOTEBOOKS
The notebooks are there to help you through the pipeline. The recommended use of the notebooks is the following:

![ringer_generalized4](https://github.com/user-attachments/assets/94edb0f8-2a02-47f4-98ad-f81b867f1321)


## References

- [AlphaFold 3](https://www.deepmind.com/research/highlighted-research/alphafold) – DeepMind's latest version of the AlphaFold protein structure prediction model.
- [Rosetta](https://www.rosettacommons.org/) – A suite of software for computational modeling and analysis of protein structures.
- [PyRosetta](https://www.pyrosetta.org/downloads#h.iwt5ktel05jc) - A Python library for Rosetta protein modeling software.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Neural network-based design tool for protein sequences.

The pipeline can also be run with LocalColabFold (AF2).
- [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) - Installer script designed to make ColabFold functionality available on users' local machines
