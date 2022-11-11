from .datum import ProteinDatum


class ProteinTransform:
    """
    Abstract class for transformation of ProteinDatum datapoints
    """

    def transform(self, datum: ProteinDatum) -> ProteinDatum:
        """
        Takes as input an individual data point, processes
        the values in it and returns a new ProteinDatum
        """
        raise NotImplementedError("method transform must be implemented")


class ProteinCrop(ProteinTransform):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def transform(self, datum):
        return ProteinDatum(
            sequence=datum.sequence[: self.crop_size],
            residue_token=datum.residue_token[: self.crop_size],
            residue_mask=datum.residue_mask[: self.crop_size],
            chain_token=datum.chain_token[: self.crop_size],
            atom_token=datum.atom_token[: self.crop_size],
            atom_coord=datum.atom_coord[: self.crop_size],
            atom_mask=datum.atom_mask[: self.crop_size],
        )
