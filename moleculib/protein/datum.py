import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import py3Dmol
from biotite.database import rcsb
from biotite.sequence import ProteinSequence as _ProteinSequence
from biotite.structure import Atom, apply_chain_wise, apply_residue_wise
from biotite.structure import array as AtomArrayConstructor
from biotite.structure import (chain_iter, filter_amino_acids, get_chain_count,
                               get_residue_count, get_residues,
                               spread_chain_wise, spread_residue_wise,
                               superimpose)
from einops import rearrange, repeat

from .alphabet import (all_atoms, all_residues, atom_index,
                       atom_to_residues_index, backbone_atoms,
                       get_residue_index, all_residues_atom_mask,
                       all_residues_atom_tokens)

from typing import Optional

class ProteinSequence:
    """
    A class representing a protein sequence with tokenized residues and associated metadata.

    This class provides a container for protein sequence information including residue tokens,
    indexes, masks, and chain assignments. It serves as a lightweight representation of
    protein sequence data that can be used for sequence-based analysis.
    """

    def __init__(
        self,
        idcode: str,  # Protein identifier (e.g., PDB ID)
        sequence: _ProteinSequence,  # Biotite protein sequence object
        residue_token: np.ndarray,  # Tokenized representation of residues
        residue_index: np.ndarray,  # Sequential indices for residues
        residue_mask: np.ndarray,  # Boolean mask for valid residues
        chain_token: np.ndarray,  # Chain identifiers for each residue
        **kwargs,  # Additional attributes to store
    ):
        """
        Initialize a ProteinSequence object.

        Args:
            idcode: Protein identifier (e.g., PDB ID)
            sequence: Biotite protein sequence object
            residue_token: Numerical representation of each residue type
            residue_index: Sequential indices for each residue position
            residue_mask: Boolean mask indicating valid residues
            chain_token: Chain identifier for each residue
            **kwargs: Additional attributes to store in the object
        """
        self.idcode = idcode
        self.sequence = str(sequence)
        self.residue_token = residue_token
        self.residue_index = residue_index
        self.residue_mask = residue_mask
        self.chain_token = chain_token
        # Add any additional attributes passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


from flax import struct


@struct.dataclass
class ProteinDatum:
    """
    A comprehensive representation of protein structure data.

    This class organizes protein structural data in a residue-centric format,
    storing both sequence and atomic-level information. It provides methods for
    manipulation, visualization, and conversion between different protein structure
    representations. The data is organized as arrays with shapes that facilitate
    machine learning applications.

    The primary organization is:
    - Residue-level arrays: [num_residues]
    - Atom-level arrays: [num_residues, max_atoms_per_residue, ...]

    Attributes are categorized into sequence information, residue properties,
    atomic coordinates, and molecular geometry.
    """

    # Basic protein identifiers
    idcode: str  # Protein identifier (e.g., PDB ID)
    resolution: float  # Structure resolution (in Angstroms)
    sequence: _ProteinSequence  # Biotite protein sequence object

    # Residue-level information
    residue_token: np.ndarray  # Tokenized representation of each residue [num_residues]
    residue_index: np.ndarray  # Index of each residue [num_residues]
    residue_mask: np.ndarray  # Boolean mask for valid residues [num_residues]
    chain_token: np.ndarray  # Chain identifier for each residue [num_residues]

    # Atom-level information
    atom_token: (
        np.ndarray
    )  # Tokenized representation of atoms [num_residues, max_atoms]
    atom_coord: np.ndarray  # 3D coordinates for each atom [num_residues, max_atoms, 3]
    atom_mask: np.ndarray  # Boolean mask for valid atoms [num_residues, max_atoms]

    # Optional atomic properties
    atom_element: Optional[np.ndarray] = None  # Element type for each atom
    atom_radius: Optional[np.ndarray] = None  # Atomic radii

    # Molecular geometry features
    bonds_list: Optional[np.ndarray] = None  # List of atom pairs forming bonds
    bonds_mask: Optional[np.ndarray] = None  # Boolean mask for valid bonds
    angles_list: Optional[np.ndarray] = None  # List of atom triplets forming angles
    angles_mask: Optional[np.ndarray] = None  # Boolean mask for valid angles
    dihedrals_list: Optional[np.ndarray] = None  # List of atom quartets forming dihedrals
    dihedrals_mask: Optional[np.ndarray] = None  # Boolean mask for valid dihedrals

    @classmethod
    def _extract_reshaped_atom_attr(
        cls,
        atom_array,
        atom_alphabet=all_atoms,
        atom_to_indices=atom_to_residues_index,
        attrs=["coord", "token"],
    ):
        """
        Reshapes atom attributes from a flat atom array into a residue-based representation.

        This method converts Biotite's flat atom array structure into a tensor representation
        where atoms are organized by residue. It creates arrays with shape [num_residues, max_atoms_per_residue, ...]
        for each requested attribute.

        Args:
            atom_array: Biotite atom array containing protein structure data
            atom_alphabet: List of atom names to process
            atom_to_indices: Mapping from atom tokens to positions within each residue
            attrs: List of atom attributes to extract (e.g., 'coord', 'token')

        Returns:
            tuple: (
                extraction: Dictionary of reshaped attributes
                mask: Boolean mask indicating valid atoms [num_residues, max_atoms_per_residue]
            )
        """
        # Get number of residues for array sizing
        residue_count = get_residue_count(atom_array)

        # Initialize extraction dictionary and atom mask
        extraction = dict()
        mask = np.zeros((residue_count, 14)).astype(bool)  # 14 = max atoms per residue

        # Initialize arrays for each requested attribute
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape
            if len(attr_shape) == 1:
                # For scalar attributes like 'token'
                attr_reshape = np.zeros((residue_count, 14))
            else:
                # For vector attributes like 'coord'
                attr_reshape = np.zeros((residue_count, 14, attr_shape[-1]))
            extraction[attr] = attr_reshape

        def _atom_slice(atom_name, atom_array, atom_token):
            """
            Process a specific atom type and place its attributes in the residue-based arrays.

            Args:
                atom_name: Name of the atom to process (e.g., 'CA', 'CB')
                atom_array: The full atom array
                atom_token: Token ID of this atom type
            """
            # Filter the atom array to only include atoms of the given name
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]

            # Remove padding residues (token 0)
            atom_array_ = atom_array_[(atom_array_.residue_token > 0)]

            # For non-backbone atoms, also remove unknown residues (token 1)
            if atom_name not in backbone_atoms:
                atom_array_ = atom_array_[(atom_array_.residue_token > 1)]

            # Get residue tokens and sequence IDs for these atoms
            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid

            # Find the position index for this atom type within each residue
            atom_indices = atom_to_indices[atom_token][res_tokens]

            # Copy attribute values to the corresponding positions in the output arrays
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor

            # Mark these positions as valid in the mask
            mask[seq_id, atom_indices] = True

        # Process each atom type in the alphabet
        for atom_name in atom_alphabet:
            atom_token = atom_alphabet.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token)

        return extraction, mask

    @staticmethod
    def separate_chains(datum):
        """
        Splits a multi-chain protein datum into a list of single-chain protein data.

        This method identifies unique chains in the protein structure and creates
        separate ProteinDatum objects for each chain, preserving all relevant
        attributes while subsetting the arrays to include only data for that chain.

        Args:
            datum: The ProteinDatum object containing multiple chains

        Returns:
            list: List of ProteinDatum objects, one for each chain
        """
        # Get unique chain identifiers
        chains = np.unique(datum.chain_token)
        protein_list = []

        # Process each chain separately
        for chain in chains:
            new_datum_ = dict()

            # Find the start index and length of this chain
            cut = np.where(datum.chain_token == chain)[0][
                0
            ]  # First residue of this chain
            length = np.sum(
                datum.chain_token == chain
            )  # Number of residues in this chain

            # Copy attributes, slicing arrays to include only this chain's data
            for attr, obj in vars(datum).items():
                if type(obj) in [np.ndarray, list, tuple, str]:
                    new_datum_[attr] = obj[cut : cut + length]
                else:
                    # Non-array attributes are copied directly
                    new_datum_[attr] = obj

            # Create a new ProteinDatum for this chain
            new_datum = ProteinDatum(**new_datum_)
            protein_list.append(new_datum)

        return protein_list

    def __len__(self):
        """
        Returns the number of residues in the protein.

        Returns:
            int: Number of residues
        """
        return len(self.atom_coord)

    @classmethod
    def empty(cls):
        """
        Creates an empty ProteinDatum object with zero-sized arrays.

        This is useful when handling edge cases, errors, or as a placeholder.

        Returns:
            ProteinDatum: An empty protein datum with empty arrays
        """
        return cls(
            idcode="",
            resolution=0.0,
            sequence=_ProteinSequence(""),
            residue_index=np.zeros(0, dtype=int),  # Empty residue indices
            residue_token=np.zeros(0, dtype=int),  # Empty residue tokens
            residue_mask=np.zeros(0, dtype=bool),  # Empty residue mask
            chain_token=np.zeros(0, dtype=int),  # Empty chain tokens
            atom_token=np.zeros((0, 14), dtype=int),  # Empty atom tokens
            atom_mask=np.zeros((0, 14), dtype=bool),  # Empty atom mask
            atom_coord=np.zeros((0, 14, 3), dtype=float),  # Empty coordinates
        )

    def replace(self, **kwargs):
        """
        Creates a new ProteinDatum with specific attributes replaced.

        This method creates a copy of the current datum and updates
        specific attributes with new values provided in kwargs.

        Args:
            **kwargs: Key-value pairs of attributes to update

        Returns:
            ProteinDatum: A new ProteinDatum with updated attributes
        """
        new_datum = dict()
        # Copy all current attributes
        for attr, obj in vars(self).items():
            new_datum[attr] = obj
        # Update with new values
        new_datum.update(kwargs)
        # Create and return a new ProteinDatum
        return ProteinDatum(**new_datum)

    def __getitem__(self, idx):
        """
        Enables slicing of the protein to get a subset of residues.

        Supports both integer indexing (single residue) and slicing.

        Args:
            idx: An integer index or slice object

        Returns:
            ProteinDatum: A new ProteinDatum containing only the specified residues
        """
        # Convert integer index to slice indices
        if type(idx) == int:
            idx = [idx, idx + 1]
        # Convert slice to explicit start and stop indices
        elif type(idx) == slice:
            idx = [idx.start, idx.stop]

        new_datum = dict()
        # Copy attributes, slicing arrays that match the protein length
        for attr, obj in vars(self).items():
            if type(obj) in [np.ndarray, list, tuple, str] and len(obj) == len(self):
                # Only slice arrays that match the number of residues
                new_datum[attr] = obj[idx[0] : idx[1]]
            else:
                # Keep other attributes unchanged
                new_datum[attr] = obj

        return ProteinDatum(**new_datum)

    @classmethod
    def from_filepath(
        cls,
        filepath,
        format=None,
        idcode=None,
        chain_id=None,
        chain=None,
        model=1,
    ):

        if str(filepath).endswith(".pdb") or format == "pdb":
            pdb_file = pdb.PDBFile.read(filepath)
            atom_array = pdb.get_structure(pdb_file, model=model)
            if idcode is None:
                idcode = str(filepath).split("/")[-1].split(".")[0]
            header = dict(
                idcode=idcode,
                resolution=None,
            )
        elif str(filepath).endswith(".bcif"):
            bcif_file = pdbx.BinaryCIFFile.read(filepath)
            atom_array = pdbx.get_structure(bcif_file, model=model)
            header = dict(idcode=None, resolution=None)
        elif str(filepath).endswith(".mmcif"):
            mmcif_file = pdbx.PDBxFile.read(filepath)
            atom_array = pdbx.get_structure(mmcif_file, model=model)
            header = dict(idcode=None, resolution=None)
        else:
            print(filepath)
            raise ValueError("File format not supported")

        aa_filter = filter_amino_acids(atom_array)
        atom_array = atom_array[aa_filter]

        if chain is not None:
            atom_array = atom_array[(atom_array.chain_id == chain)]

        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def fetch_pdb_id(cls, id, format="pdb", chain=None, model=None, save_path=None):
        filepath = rcsb.fetch(id, format, save_path)
        return cls.from_filepath(
            filepath,
            format=format,
            chain=chain,
            model=model,
            idcode=id if chain is None else f"{id}_{chain}",
        )

    def set(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_sequence(self):
        return _ProteinSequence([all_residues[token] for token in self.residue_token])

    @classmethod
    def from_atom_array(
        cls,
        atom_array,
        header=None,
    ):
        """
        Reshapes atom array to residue-indexed representation to
        build a protein datum.
        """
        if header == None:
            header = dict(
                idcode=None,
                resolution=None,
            )

        if atom_array.array_length() == 0:
            return cls.empty()

        # Small tweak for CHARMM files
        atom_names = atom_array.atom_name
        cd_filter = (atom_array.res_name == "ILE") & (atom_names == "CD")
        atom_names[cd_filter] = np.array(["CD1"] * sum(cd_filter))
        atom_array.set_annotation("atom_name", atom_names)

        _, res_names = get_residues(atom_array)
        res_names = [
            ("UNK" if (name not in all_residues) else name) for name in res_names
        ]
        sequence = _ProteinSequence(list(res_names))

        # index residues globally
        atom_array.add_annotation("seq_uid", int)
        atom_array.seq_uid = spread_residue_wise(
            atom_array, np.arange(0, get_residue_count(atom_array))
        )

        # tokenize atoms
        atom_array.add_annotation("token", int)
        atom_array.token = np.array(
            list(map(lambda atom: atom_index(atom), atom_array.atom_name))
        )

        # tokenize residues
        residue_token = np.array(
            list(map(lambda res: get_residue_index(res), atom_array.res_name))
        )
        residue_mask = np.ones_like(residue_token).astype(bool)

        atom_array.add_annotation("residue_token", int)
        atom_array.residue_token = residue_token
        chain_token = spread_chain_wise(
            atom_array, np.arange(0, get_chain_count(atom_array))
        )

        # count number of residues per chain
        # and index residues per chain using cumulative sum
        atom_array.add_annotation("res_uid", int)

        def _count_residues_per_chain(chain_atom_array, axis=0):
            return get_residue_count(chain_atom_array)

        chain_res_sizes = apply_chain_wise(
            atom_array, atom_array, _count_residues_per_chain, axis=0
        )
        chain_res_cumsum = np.cumsum([0] + list(chain_res_sizes[:-1]))
        atom_array.res_uid = atom_array.res_id + chain_res_cumsum[chain_token]

        # reshape atom attributes to residue-based representation
        # with the correct ordering
        # [N * 14, ...] -> [N, 14, ...]
        atom_extract, atom_mask = cls._extract_reshaped_atom_attr(
            atom_array, attrs=["coord", "token"]
        )
        atom_extract = dict(
            map(lambda kv: (f"atom_{kv[0]}", kv[1]), atom_extract.items())
        )

        # pool residue attributes and create residue features
        # [N * 14, ...] -> [N, ...]
        def _pool_residue_token(atom_residue_tokens, axis=0):
            representative = atom_residue_tokens[0]
            return representative

        def _reshape_residue_attr(attr):
            return apply_residue_wise(atom_array, attr, _pool_residue_token, axis=0)

        residue_token = _reshape_residue_attr(residue_token)
        residue_index = np.arange(0, residue_token.shape[0])

        residue_mask = _reshape_residue_attr(residue_mask)
        residue_mask = residue_mask & (atom_extract["atom_coord"].sum((-1, -2)) != 0)

        chain_token = _reshape_residue_attr(chain_token)

        return cls(
            idcode=header["idcode"],
            sequence=sequence,
            resolution=header["resolution"],
            residue_token=residue_token,
            residue_index=residue_index,
            residue_mask=residue_mask,
            chain_token=chain_token,
            **atom_extract,
            atom_mask=atom_mask,
        )

    def _apply_chemistry(self, key, f):
        """
        Generic helper method to apply geometric functions to molecular components.

        This method reshapes atom coordinates and indices to apply calculations
        to bonds, angles, or dihedrals in the protein structure.

        Args:
            key: The type of component ('bonds', 'angles', or 'dihedrals')
            f: Function to apply to the component that computes geometric measurements

        Returns:
            np.ndarray: Array of computed measurements
        """
        # Reshape atom coordinates to a flat list
        all_atoms = rearrange(self.atom_coord, "r a c -> (r a) c")
        # Reshape component indices to a flat list
        all_idx = rearrange(getattr(self, f"{key}_list"), "r o i -> (r o) i")
        # Get the mask for valid components
        mask = getattr(self, f"{key}_mask")

        # Apply the measurement function
        measures = f(all_atoms, all_idx)
        # Reshape the results back to [residues, components]
        measures = rearrange(measures, "(r o) -> r o", r=len(self.residue_token))

        # Apply mask to zero out invalid measurements
        output = dict()
        output[key] = measures * mask
        return measures

    def apply_bonds(self, f):
        """
        Applies a function to calculate bond properties.

        Args:
            f: Function that calculates bond properties from atom coordinates

        Returns:
            np.ndarray: Array of bond measurements
        """
        return self._apply_chemistry(key="bonds", f=f)

    def apply_angles(self, f):
        """
        Applies a function to calculate angle properties.

        Args:
            f: Function that calculates angle properties from atom coordinates

        Returns:
            np.ndarray: Array of angle measurements
        """
        return self._apply_chemistry(key="angles", f=f)

    def apply_dihedrals(self, f):
        """
        Applies a function to calculate dihedral angle properties.

        Args:
            f: Function that calculates dihedral properties from atom coordinates

        Returns:
            np.ndarray: Array of dihedral angle measurements
        """
        return self._apply_chemistry(key="dihedrals", f=f)

    def apply(self, f):
        """
        Applies a function to all numpy-convertible attributes of the protein datum.

        This is useful for batch processing of attributes, such as moving data between devices,
        converting data types, or applying transformations.

        Args:
            f: Function to apply to each attribute with a numpy() method
        """
        for key, value in vars(self).items():
            # If the value has a numpy() method, apply the function
            if hasattr(value, "numpy"):
                setattr(self, key, f(value))

    def to_pdb_str(self):
        # https://colab.research.google.com/github/pb3lab/ibm3202/blob/
        # master/tutorials/lab02_molviz.ipynb#scrollTo=FPS04wJf5k3f
        assert len(self.residue_token.shape) == 1
        atom_mask = self.atom_mask.astype(np.bool_)
        all_atom_coords = self.atom_coord[atom_mask]
        all_atom_tokens = self.atom_token[atom_mask]
        all_atom_res_tokens = repeat(self.residue_token, "r -> r a", a=14)[atom_mask]
        all_atom_res_indices = repeat(self.residue_index, "r -> r a", a=14)[atom_mask]

        # just in case, move to cpu
        atom_mask = np.array(atom_mask)
        all_atom_coords = np.array(all_atom_coords)
        all_atom_tokens = np.array(all_atom_tokens)
        all_atom_res_tokens = np.array(all_atom_res_tokens)
        all_atom_res_indices = np.array(all_atom_res_indices)

        lines = []
        for idx, (coord, token, res_token, res_index) in enumerate(
            zip(
                all_atom_coords,
                all_atom_tokens,
                all_atom_res_tokens,
                all_atom_res_indices,
            )
        ):
            name = all_atoms[int(token)]
            res_name = all_residues[int(res_token)]
            x, y, z = coord
            line = list(" " * 80)
            line[0:6] = "ATOM".ljust(6)
            line[6:11] = str(idx + 1).ljust(5)
            line[12:16] = name.ljust(4)
            line[17:20] = res_name.ljust(3)
            line[21:22] = "A"
            line[23:27] = str(res_index + 1).ljust(4)
            line[30:38] = f"{x:.3f}".rjust(8)
            line[38:46] = f"{y:.3f}".rjust(8)
            line[46:54] = f"{z:.3f}".rjust(8)
            line[76:78] = name[0].rjust(2)
            lines.append("".join(line))
        lines = "\n".join(lines)
        return lines

    def plot(
        self,
        view=None,
        viewer=None,
        sphere=False,
        ribbon=True,
        sidechain=True,
        color="spectrum",
        colors=None,
    ):
        if viewer is None:
            viewer = (0, 0)
        if view is None:
            view = py3Dmol.view(width=800, height=800)

        view.addModel(self.to_pdb_str(), "pdb", viewer=viewer)
        view.setStyle({"model": -1}, {}, viewer=viewer)
        if sphere:
            view.addStyle(
                {"model": -1},
                {"sphere": {"radius": 0.3, "color": color}},
                viewer=viewer,
            )

        if ribbon:
            view.addStyle({"model": -1}, {"cartoon": {"color": color}}, viewer=viewer)

        if sidechain:
            if color != "spectrum":
                view.addStyle(
                    {"model": -1},
                    {"stick": {"radius": 0.2, "color": color}},
                    viewer=viewer,
                )
            else:
                view.addStyle({"model": -1}, {"stick": {"radius": 0.2}}, viewer=viewer)

        if colors is not None:
            colors = {i + 1: c for i, c in enumerate(colors)}
            view.addStyle(
                {"model": -1},
                {
                    "stick": {"colorscheme": {"prop": "resi", "map": colors}},
                    "sphere": {"colorscheme": {"prop": "resi", "map": colors}},
                    "cartoon": {"colorscheme": {"prop": "resi", "map": colors}},
                },
            )  # 'label': {'colorscheme': {'prop': 'resi', 'map': colors}}, 'surface': {'colorscheme': {'prop': 'resi', 'map': colors}}, 'dot': {'colorscheme': {'prop': 'resi', 'map': colors}}, 'contact': {'colorscheme': {'prop': 'resi', 'map': colors}}, 'callback': 'function(){}'}, viewer=viewer)

        return view

    def to_atom_array(self):
        atom_mask = self.atom_mask.astype(np.bool_)
        all_atom_coords = self.atom_coord[atom_mask]
        all_atom_tokens = self.atom_token[atom_mask]
        all_atom_res_tokens = repeat(self.residue_token, "r -> r a", a=14)[atom_mask]
        all_atom_res_indices = repeat(self.residue_index, "r -> r a", a=14)[atom_mask]

        # just in case, move to cpu
        atom_mask = np.array(atom_mask)
        all_atom_coords = np.array(all_atom_coords)
        all_atom_tokens = np.array(all_atom_tokens)
        all_atom_res_tokens = np.array(all_atom_res_tokens)
        all_atom_res_indices = np.array(all_atom_res_indices)

        atoms = []
        for idx, (coord, token, res_token, res_index) in enumerate(
            zip(
                all_atom_coords,
                all_atom_tokens,
                all_atom_res_tokens,
                all_atom_res_indices,
            )
        ):
            name = all_atoms[int(token)]
            res_name = all_residues[int(res_token)]
            atoms.append(
                Atom(
                    atom_name=name,
                    element=name[0],
                    coord=coord,
                    res_id=res_index,
                    res_name=res_name,
                    chain_id="A",
                )
            )

        return AtomArrayConstructor(atoms)

    def align_to(self, other):
        """
        Aligns the current protein datum to another protein datum based on CA atoms.

        This method performs a structural alignment between two protein structures
        using only their alpha carbon atoms. It computes the optimal rotation and
        translation to minimize RMSD between aligned atoms.

        Args:
            other: Target ProteinDatum to align to
            window: Optional tuple of (start, end) to restrict alignment to a specific region

        Returns:
            ProteinDatum: A new protein datum with aligned coordinates
        """
        # NOTE(Allan): Triple check that this works, modified recently
        self_array, other_array = self.to_atom_array(), other.to_atom_array()
        _, transform = superimpose(other_array, self_array)
        new_atom_coord = self.atom_coord + transform.center_translation
        new_atom_coord = np.einsum(
            "rca,ab->rcb", new_atom_coord, transform.rotation.squeeze(0)
        )
        new_atom_coord += transform.target_translation
        new_atom_coord = new_atom_coord * self.atom_mask[..., None]

        return self.set(atom_coord=new_atom_coord)


    def to_tensor_cloud(self):
        import jax.numpy as jnp
        import e3nn_jax as e3nn
        from tensorclouds.tensorcloud import TensorCloud

        res_token = self.residue_token
        res_mask = self.atom_mask[..., 1]
        vectors = self.atom_coord
        mask = self.atom_mask

        ca_coord = vectors[..., 1, :]

        vectors = vectors - ca_coord[..., None, :]
        vectors = vectors * mask[..., None]
        vectors = rearrange(vectors, "r a c -> r (a c)")

        irreps_array = e3nn.IrrepsArray("14x1e", jnp.array(vectors))

        tensorcloud = TensorCloud(
            irreps_array=irreps_array,
            mask_irreps_array=jnp.array(mask),
            coord=jnp.array(ca_coord),
            mask_coord=jnp.array(res_mask),
            label=jnp.array(res_token * res_mask),
        )

        return tensorcloud

    @classmethod
    def from_tensor_cloud(cls, tensorcloud):
        import jax.numpy as jnp

        irreps_array = tensorcloud.irreps_array
        ca_coord = tensorcloud.coord
        res_mask = tensorcloud.mask_coord


        if tensorcloud.annotations and ('b_factor' in tensorcloud.annotations):
            resolution = tensorcloud.annotations['b_factor']
        else:
            resolution = None

        atom_coord = irreps_array.filter("1e").array
        atom_coord = rearrange(atom_coord, "r (a c) -> r a c", a=14)

        labels = tensorcloud.label
        logit_extract = repeat(labels, "r -> r l", l=23) == repeat(
            jnp.arange(0, 23), "l -> () l"
        )

        atom_token = (logit_extract[..., None] * all_residues_atom_tokens[None]).sum(-2)
        atom_mask = (logit_extract[..., None] * all_residues_atom_mask[None]).sum(-2)

        atom_coord = atom_coord.at[..., 1, :].set(0.0)
        atom_coord = atom_coord + ca_coord[..., None, :]
        atom_coord = atom_coord * atom_mask[..., None]

        return cls(
            idcode=None,
            resolution=resolution,
            sequence=None,
            residue_token=labels,
            residue_index=jnp.arange(labels.shape[0]),
            residue_mask=res_mask,
            chain_token=jnp.zeros(labels.shape[0], dtype=jnp.int32),
            atom_token=atom_token,
            atom_coord=atom_coord,
            atom_mask=atom_mask,
        )

    def __repr__(self):
        """
        Returns a string representation of the ProteinDatum object.

        The representation shows the shape of the atom coordinate array,
        which indicates the number of residues and atoms per residue.

        Returns:
            str: String representation of the ProteinDatum
        """
        return f"ProteinDatum(shape={self.atom_coord.shape[:-1]})"
