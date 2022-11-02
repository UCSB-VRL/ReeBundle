# Reeb Graph for modeling neuronal fiber pathways

Given the trajectories of neuronal fiber pathways, we model the evolution of trajectories that capture geometrically significant events (akin to a
fingerprint) and calculate their point correspondence in the 3D brain space. We show that our model can handle the presence of improbable streamlines, as commonly produced by tractography algorithms. Three key parameters in our method capture the geometry and topology of the streamlines: (i) $\epsilon$ -- the distance between a pair of streamlines in a bundle that defines its sparsity; (ii) $\alpha$ -- the spatial length of the bundle that introduces persistence; and (iii) $\delta$ -- the bundle thickness. Together, these parameters control the granularity of the model to provide a compact signature of the tracts and their underlying anatomical structure.

For more details about our algorithm, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2022.03.11.482601v1.abstract).

## Citation

The system was employed for our research presented in [1], where we propose a novel and efficient algorithm to model high-level topological structures of neuronal fibers. Tractography constructs complex neuronal fibers in three dimensions that exhibit the geometry of white matter pathways in the brain. However, most tractography analysis methods are time consuming and intractable. We develop a computational geometry-based tractography representation that aims to simplify the connectivity of white matter fibers. If the use of the software or the idea of the paper positively influences your endeavours, please cite [1].

[1] Shailja, S., Angela Zhang, and B. S. Manjunath. "A computational geometry approach for modeling neuronal fiber pathways." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.

## Requirements

To download all prerequisites, in the terminal type
`pip install -r requirements.txt`

The code has been tested only on python version 3.7.


## Example usage


 An example .trk file has been included in the Data directory.

## Dataset

The code was tested on the publicly available ISMRM datset [dataset](https://doi.org/10.5281/zenodo.572345).
