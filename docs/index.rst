MoleculIB: Molecular Data Library for Machine Learning
======================================================

**MoleculIB** is a comprehensive Python package designed to streamline the processing and management of biomolecular data within machine learning workflows. It empowers researchers and developers with a robust set of functionalities for working with proteins, nucleic acids, and small molecules, simplifying complex data pipelines and accelerating the journey from raw biomolecular data to valuable insights.

Key Features
-----------

* **Data Processing**: Preprocessing and cleaning capabilities for data normalization and feature engineering
* **Machine Learning-First**: Designed to readily integrate with JAX and pytrees in machine learning pipelines
* **Reproducibility**: Modular classes and workflows for better organization and reproducible research
* **Multi-scale Representation**: Work with molecules at different scales - from sequences to 3D structures
* **Interoperability**: Seamless conversion between common biomolecular file formats and data structures

Core Components
-------------

Protein Module
~~~~~~~~~~~~~

The ``protein`` module provides comprehensive tools for working with protein structures and sequences:

* ``ProteinDatum``: Core class for protein structure representation with residue and atom-level information
* Manipulation of protein coordinates, sequences, and structural features
* Support for common operations like alignment, chain separation, and structural analysis
* Conversion between different protein file formats (PDB, mmCIF, mmtf)

Nucleic Module
~~~~~~~~~~~~

The ``nucleic`` module handles RNA and DNA structures and sequences:

* Support for standard nucleic acid operations and representations
* Tools for working with RNA/DNA secondary structure
* Conversion utilities for different nucleic acid file formats

Molecule Module
~~~~~~~~~~~~~

The ``molecule`` module focuses on small molecule representation and processing:

* Tools for representing and manipulating small molecules
* Atom typing and feature extraction
* Conversion between different molecular file formats

Assembly Module
~~~~~~~~~~~~~

The ``assembly`` module handles multi-component biological assemblies:

* Tools for working with protein-protein complexes
* Support for protein-ligand and protein-nucleic acid interactions
* Assembly generation and analysis capabilities

Sequence Module
~~~~~~~~~~~~~

Tools for biological sequence analysis:

* Sequence alignment and comparison
* Feature extraction from sequences
* Integration with structure-based features

Metrics and Loss
~~~~~~~~~~~~~~

The library includes specialized modules for molecular metrics and loss functions:

* Domain-specific metrics for evaluating molecular models
* Custom loss functions designed for biomolecular machine learning tasks
* Support for both training and evaluation workflows

Installation
-----------

Install MoleculIB via pip::

    pip install moleculib

Requirements:

* Python ≥ 3.7
* NumPy ≥ 1.20.0
* JAX ≥ 0.3.0
* Biotite ≥ 0.30.0
* py3Dmol ≥ 1.8.0 (for visualization)

Getting Started
-------------

Basic usage example::

    import moleculib as mol
    
    # Load a protein from a PDB file
    protein = mol.protein.ProteinDatum.from_filepath("example.pdb")
    
    # Access protein attributes
    print(f"Number of residues: {len(protein.residue_token)}")
    print(f"Protein sequence: {protein.get_sequence()}")
    
    # Perform operations
    chains = mol.protein.ProteinDatum.separate_chains(protein)
    aligned_protein = protein.align_to(reference_protein)
    
    # Save in different format
    protein.save_mmcif("output.cif")

For more examples and detailed documentation, see the specific module sections below.

API Reference
-----------

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/protein
   modules/nucleic
   modules/molecule
   modules/assembly
   modules/sequence
   modules/metrics
   modules/loss

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
