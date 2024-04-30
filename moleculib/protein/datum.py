import numpy as np
from biotite.database import rcsb
from biotite.sequence import ProteinSequence as _ProteinSequence
from biotite.structure import (
    apply_chain_wise,
    apply_residue_wise,
    get_chain_count,
    get_residue_count,
    get_residues,
    spread_chain_wise,
    spread_residue_wise,
    chain_iter,
)

from biotite.structure import filter_amino_acids

import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.structure import Atom
from biotite.structure import array as AtomArrayConstructor
from biotite.structure import superimpose

from .alphabet import (
    all_atoms,
    all_residues,
    backbone_atoms,
    atom_index,
    atom_to_residues_index,
    get_residue_index,
)

from einops import rearrange, repeat
from simple_pytree import Pytree

class ProteinSequence:

    def __init__(
        self,
        idcode: str,
        sequence: _ProteinSequence,
        residue_token: np.ndarray,
        residue_index: np.ndarray,
        residue_mask: np.ndarray,
        chain_token: np.ndarray,
        **kwargs,
    ):
        self.idcode = idcode
        self.sequence = str(sequence)
        self.residue_token = residue_token
        self.residue_index = residue_index
        self.residue_mask = residue_mask
        self.chain_token = chain_token
        for key, value in kwargs.items():
            setattr(self, key, value)


class ProteinDatum(Pytree, mutable=True):
    """
    Incorporates protein data to MolecularDatum
    and reshapes atom arrays to residue-based representation
    """

    def __init__(
        self,
        idcode: str,
        resolution: float,
        sequence: _ProteinSequence,
        residue_token: np.ndarray,
        residue_index: np.ndarray,
        residue_mask: np.ndarray,
        chain_token: np.ndarray,
        atom_token: np.ndarray,
        atom_coord: np.ndarray,
        atom_mask: np.ndarray,
        **kwargs,
    ):
        self.idcode = idcode 
        self.resolution = resolution
        self.sequence = None #str(sequence)
        self.residue_token = residue_token
        self.residue_index = residue_index
        self.residue_mask = residue_mask
        self.chain_token = chain_token
        self.atom_token = atom_token
        self.atom_coord = atom_coord
        self.atom_mask = atom_mask
        self.atom_element = None
        self.atom_radius = None
        self.flips_list = None
        self.flips_mask = None
        # self.mask = None
        # self.bonds_list = None
        # self.bonds_mask = None
        self.angles_list = None
        self.angles_mask = None
        self.dihedrals_list = None
        self.dihedrals_mask = None
        self.boundary_token = None
        self.boundary_mask = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _extract_reshaped_atom_attr(
        cls,
        atom_array,
        atom_alphabet=all_atoms,
        atom_to_indices=atom_to_residues_index,
        attrs=["coord", "token"],
    ):
        residue_count = get_residue_count(atom_array)

        extraction = dict()
        mask = np.zeros((residue_count, 14)).astype(bool)
        for attr in attrs:
            attr_shape = getattr(atom_array, attr).shape
            if len(attr_shape) == 1:
                attr_reshape = np.zeros((residue_count, 14))
            else:
                attr_reshape = np.zeros((residue_count, 14, attr_shape[-1]))
            extraction[attr] = attr_reshape

        def _atom_slice(atom_name, atom_array, atom_token):
            atom_array_ = atom_array[(atom_array.atom_name == atom_name)]
            # kill pads and kill unks that are not backbone
            atom_array_ = atom_array_[(atom_array_.residue_token > 0)]
            if atom_name not in backbone_atoms:
                atom_array_ = atom_array_[(atom_array_.residue_token > 1)]

            res_tokens, seq_id = atom_array_.residue_token, atom_array_.seq_uid
            atom_indices = atom_to_indices[atom_token][res_tokens]
            for attr in attrs:
                attr_tensor = getattr(atom_array_, attr)
                extraction[attr][seq_id, atom_indices, ...] = attr_tensor
            mask[seq_id, atom_indices] = True

        for atom_name in atom_alphabet:
            atom_token = atom_alphabet.index(atom_name)
            _atom_slice(atom_name, atom_array, atom_token)

        return extraction, mask

    @staticmethod
    def separate_chains(datum): 
        chains = np.unique(datum.chain_token)
        protein_list = []
        for chain in chains:
            new_datum_ = dict()
            cut = np.where(datum.chain_token == chain)[0][0]
            length = np.sum(datum.chain_token == chain)
            for attr, obj in vars(datum).items():
                if type(obj) in [np.ndarray, list, tuple, str]:
                    new_datum_[attr] = obj[cut : cut + length]
                else:
                    new_datum_[attr] = obj
            new_datum = ProteinDatum(**new_datum_)
            protein_list.append(new_datum)
        return protein_list

    def __len__(self):
        return len(self.atom_coord)

    @classmethod
    def empty(cls):
        return cls(
            idcode="",
            resolution=0.0,
            sequence=_ProteinSequence(""),
            residue_index=np.zeros(0, dtype=int),
            residue_token=np.zeros(0, dtype=int),
            residue_mask=np.zeros(0, dtype=bool),
            chain_token=np.zeros(0, dtype=int),
            atom_token=np.zeros((0, 14), dtype=int),
            atom_mask=np.zeros((0, 14), dtype=bool),
            atom_coord=np.zeros((0, 14, 3), dtype=float)
        )

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx, idx + 1]
        elif type(idx) == slice:
            idx = [idx.start, idx.stop]
        new_datum = dict()
        for attr, obj in vars(self).items():
            if type(obj) in [np.ndarray, list, tuple, str] and len(obj) == len(self):
                new_datum[attr] = obj[idx[0] : idx[1]]
            else:
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
        model=None,
    ):
        if format == 'pdb' or filepath.endswith(".pdb"):
            pdb_file = pdb.PDBFile.read(filepath)
            atom_array = pdb.get_structure(pdb_file, model=1)
            if idcode is None:
                idcode = str(filepath).split("/")[-1].split(".")[0]
            header = dict(
                idcode=idcode,
                resolution=None,
            )
        elif filepath.endswith(".mmtf"): 
            mmtf_file = mmtf.MMTFFile.read(filepath)
            atom_array = mmtf.get_structure(mmtf_file, model=1)
            header = dict(
                idcode=mmtf_file["structureId"] if "structureId" in mmtf_file else None,
                resolution=None
                if ("resolution" not in mmtf_file)
                else mmtf_file["resolution"],
            )
        elif filepath.endswith(".mmcif"):
            mmcif_file = pdbx.PDBxFile.read(filepath)
            atom_array = pdbx.get_structure(mmcif_file, model=1)
            header = dict(
                idcode=None,
                resolution=None
            )

        aa_filter = filter_amino_acids(atom_array)
        atom_array = atom_array[aa_filter]
    
        if chain is not None:
            atom_array = atom_array[(atom_array.chain_id == chain)]

        return cls.from_atom_array(atom_array, header=header)

    @classmethod
    def fetch_pdb_id(
        cls, 
        id, 
        format='pdb', 
        chain=None, 
        model=None, 
        save_path=None
    ):
        filepath = rcsb.fetch(id, format, save_path)
        return cls.from_filepath(
            filepath, 
            format=format, 
            chain=chain, 
            model=model,
            idcode=id if chain is None else f"{id}_{chain}"
        )
    
    def set(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @classmethod
    def from_atom_array(
        cls,
        atom_array,
        header,
    ):
        """
        Reshapes atom array to residue-indexed representation to
        build a protein datum.
        """

        if atom_array.array_length() == 0:
            return cls.empty()

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
        all_atoms = rearrange(self.atom_coord, "r a c -> (r a) c")
        all_idx = rearrange(getattr(self, f"{key}_list"), "r o i -> (r o) i")
        mask = getattr(self, f"{key}_mask")

        measures = f(all_atoms, all_idx)
        measures = rearrange(measures, "(r o) -> r o", r=len(self.residue_token))

        output = dict()
        output[key] = measures * mask
        return measures

    def apply_bonds(self, f):
        return self._apply_chemistry(key="bonds", f=f)

    def apply_angles(self, f):
        return self._apply_chemistry(key="angles", f=f)

    def apply_dihedrals(self, f):
        return self._apply_chemistry(key="dihedrals", f=f)

    def to_dict(self, attrs=None):
        if attrs is None:
            attrs = vars(self).keys()
        dict_ = {}
        for attr in attrs:
            obj = getattr(self, attr)
            # strings are not JAX types
            if type(obj) == str:
                continue
            if type(obj) in [list, tuple]:
                if type(obj[0]) not in [int, float]:
                    continue
                obj = np.array(obj)
            dict_[attr] = obj
        return dict_

    def numpy(self):
        for key, value in vars(self).items():
            # if the value has a numpy() method, call it
            if hasattr(value, "numpy"):
                setattr(self, key, value.numpy())

    def to_pdb_str(self):
        # https://colab.research.google.com/github/pb3lab/ibm3202/blob/
        # master/tutorials/lab02_molviz.ipynb#scrollTo=FPS04wJf5k3f
        assert len(self.residue_token.shape) == 1
        atom_mask = self.atom_mask.astype(np.bool_)
        all_atom_coords = self.atom_coord[atom_mask]
        all_atom_tokens = self.atom_token[atom_mask]
        all_atom_res_tokens = repeat(self.residue_token, "r -> r a", a=14)[atom_mask]
        all_atom_res_indices = repeat(np.arange(len(self.residue_token)), "r -> r a", a=14)[atom_mask]

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
        view, 
        viewer=None, 
        sphere=False, 
        ribbon=True,
        sidechain=True,
        color='spectrum',
    ):
        if viewer is None:
            viewer = (0, 0)
        view.addModel(self.to_pdb_str(), 'pdb', viewer=viewer)
        view.setStyle({'model': -1}, {}, viewer=viewer)
        if sphere:
            view.addStyle({'model': -1}, {'sphere': {'radius': 0.3}}, viewer=viewer)

        if ribbon:
            view.addStyle({'model': -1}, {'cartoon': {'color': color}}, viewer=viewer)

        if sidechain:
            if color != 'spectrum':
                view.addStyle({'model': -1}, {'stick': {'radius': 0.2, 'color': color}}, viewer=viewer)
            else:
                view.addStyle({'model': -1}, {'stick': {'radius': 0.2}}, viewer=viewer)

        return view


    def align_to(
        self,
        other,
        window=None
    ):
        """
        Aligns the current protein datum to another protein datum based on CA atoms.
        """
        def to_atom_array(prot, mask):
            cas = prot.atom_coord[..., 1, :]
            atoms = [
                Atom(
                    atom_name="CA",
                    element="C",
                    coord=ca,
                    res_id=prot.residue_index[i],
                    chain_id=prot.chain_token[i],
                )
                for i, ca in enumerate(cas) if mask[i]
            ]
            return AtomArrayConstructor(atoms)

        common_mask = self.atom_mask[..., 1] & other.atom_mask[..., 1]
        if window is not None:
            common_mask = common_mask & (np.arange(len(common_mask)) < window[1]) & (np.arange(len(common_mask)) >= window[0])

        self_array, other_array = to_atom_array(self, common_mask), to_atom_array(other, common_mask)
        _, transform = superimpose(other_array, self_array) 
        new_atom_coord = self.atom_coord + transform.center_translation
        new_atom_coord = np.einsum("rca,ab->rcb", new_atom_coord, transform.rotation.squeeze(0))
        new_atom_coord += transform.target_translation
        new_atom_coord = new_atom_coord * self.atom_mask[..., None]

        return self.set(atom_coord=new_atom_coord)

    def tm_align_to(
        self,
        other,
        window: tuple = None,
    ):
        """
        Aligns the current protein datum to another protein datum based on TM-align.
        """
        from tmtools import tm_align

        mask = self.atom_mask[..., 1] & other.atom_mask[..., 1]        
        if window is not None:
            mask = mask &( np.arange(len(mask)) < window[1]) & (np.arange(len(mask)) >= window[0])

        cas_self = self.atom_coord[mask, 1, :]
        cas_other = other.atom_coord[mask, 1, :]
        sequence = ''.join(['G' for _ in range(len(cas_self))])

        transform = tm_align(cas_self, cas_other, sequence, sequence)
        translation, rotation = transform.t, transform.u

        translation_to_center = np.mean(cas_self, axis=0)
        new_atom_coord = self.atom_coord - translation_to_center
        new_atom_coord = np.einsum("rca,ab->rcb", new_atom_coord, rotation)
        # new_atom_coord += translation_to_center
        # new_atom_coord += translation
        new_atom_coord = new_atom_coord * self.atom_mask[..., None]

        return self.set(atom_coord=new_atom_coord)