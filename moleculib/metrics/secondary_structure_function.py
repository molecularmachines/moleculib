import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.sequence as seq
import biotite.sequence.graphics as graphics
from matplotlib.patches import Rectangle


# Note: HelixPlotter, SheetPlotter, and visualize_secondary_structures accessed from Biotite at the provided link.


class HelixPlotter(graphics.FeaturePlotter):
    # class from Biotite https://www.biotite-python.org/examples/gallery/structure/transketolase_sse.html

    def __init__(self):
        pass

    # Check whether this class is applicable for drawing a feature
    def matches(self, feature):
        if feature.key == "SecStr":
            if "sec_str_type" in feature.qual:
                if feature.qual["sec_str_type"] == "helix":
                    return True
        return False

    # The drawing function itself
    def draw(self, axes, feature, bbox, loc, style_param):
        # Approx. 1 turn per 3.6 residues to resemble natural helix
        n_turns = np.ceil((loc.last - loc.first + 1) / 3.6)
        x_val = np.linspace(0, n_turns * 2 * np.pi, 100)
        # Curve ranges from 0.3 to 0.7
        y_val = (-0.4 * np.sin(x_val) + 1) / 2

        # Transform values for correct location in feature map
        x_val *= bbox.width / (n_turns * 2 * np.pi)
        x_val += bbox.x0
        y_val *= bbox.height
        y_val += bbox.y0

        # Draw white background to overlay the guiding line
        background = Rectangle(
            bbox.p0, bbox.width, bbox.height, color="white", linewidth=0
        )
        axes.add_patch(background)
        axes.plot(x_val, y_val, linewidth=2, color=biotite.colors["dimgreen"])


class SheetPlotter(graphics.FeaturePlotter):
    # class from Biotite https://www.biotite-python.org/examples/gallery/structure/transketolase_sse.html

    def __init__(self, head_width=0.8, tail_width=0.5):
        self._head_width = head_width
        self._tail_width = tail_width

    def matches(self, feature):
        if feature.key == "SecStr":
            if "sec_str_type" in feature.qual:
                if feature.qual["sec_str_type"] == "sheet":
                    return True
        return False

    def draw(self, axes, feature, bbox, loc, style_param):
        x = bbox.x0
        y = bbox.y0 + bbox.height / 2
        dx = bbox.width
        dy = 0

        if loc.defect & seq.Location.Defect.MISS_RIGHT:
            # If the feature extends into the prevoius or next line
            # do not draw an arrow head
            draw_head = False
        else:
            draw_head = True

        axes.add_patch(
            biotite.AdaptiveFancyArrow(
                x,
                y,
                dx,
                dy,
                self._tail_width * bbox.height,
                self._head_width * bbox.height,
                # Create head with 90 degrees tip
                # -> head width/length ratio = 1/2
                head_ratio=0.5,
                draw_head=draw_head,
                color=biotite.colors["orange"],
                linewidth=0,
            )
        )


def visualize_secondary_structure(sse, first_id, chain_id="N/A"):
    # function from Biotite https://www.biotite-python.org/examples/gallery/structure/transketolase_sse.html
    # note: chain_id not an argument in original biotite version
    def _add_sec_str(annotation, first, last, str_type):
        if str_type == "a":
            str_type = "helix"
        elif str_type == "b":
            str_type = "sheet"
        else:
            # coil
            return
        feature = seq.Feature(
            "SecStr", [seq.Location(first, last)], {"sec_str_type": str_type}
        )
        annotation.add_feature(feature)

    # Find the intervals for each secondary structure element
    # and add to annotation
    annotation = seq.Annotation()
    curr_sse = None
    curr_start = None
    for i in range(len(sse)):
        if curr_start is None:
            curr_start = i
            curr_sse = sse[i]
        else:
            if sse[i] != sse[i - 1]:
                _add_sec_str(
                    annotation, curr_start + first_id, i - 1 + first_id, curr_sse
                )
                curr_start = i
                curr_sse = sse[i]
    # Add last secondary structure element to annotation
    _add_sec_str(annotation, curr_start + first_id, i - 1 + first_id, curr_sse)

    fig = plt.figure(figsize=(8.0, 3.0))
    ax = fig.add_subplot(111)
    graphics.plot_feature_map(
        ax,
        annotation,
        symbols_per_line=150,
        loc_range=(first_id, first_id + len(sse)),
        show_numbers=True,
        show_line_position=True,
        feature_plotters=[HelixPlotter(), SheetPlotter()],
    )

    fig.suptitle("Chain " + chain_id + " Visualized")  # Edited to add a title

    fig.tight_layout()


# Secondary Structure Function
def sec_struc(pdb_file, visualize=False):
    ss_by_struc = {}
    structure_types = {"HELIX", "SHEET"}

    def sec_struc_type(struc_type, filename):
        chains = []
        struc_dict = {}
        raw_line = []
        filename.seek(0)
        assert struc_type in {"HELIX", "SHEET"}
        start, stop, chain_loc = 5, 8, 4
        if struc_type == "SHEET":
            start, stop, chain_loc = 6, 9, 5
        for line in filename:
            if line[0:5] == struc_type:
                raw_line = line.split()
                chain = raw_line[chain_loc]
                locations = (int(raw_line[start]), int(raw_line[stop]))
                if chain in struc_dict:
                    if locations not in struc_dict[chain]:
                        struc_dict[chain].add(locations)
                else:
                    struc_dict[chain] = {(int(raw_line[start]), int(raw_line[stop]))}
                if chain not in chains:
                    chains.append(chain)
        return struc_dict, chains

    # getting secondary structures for each chain
    for structure_type in structure_types:
        structure, structure_chains = sec_struc_type(structure_type, pdb_file)
        for structure_chain in structure_chains:
            if structure_chain not in ss_by_struc:
                ss_by_struc[structure_chain] = {}
            ss_by_struc[structure_chain][structure_type] = structure[structure_chain]

    # getting metadata on the secondary structures
    ss_meta = {}
    for chain, ss_structures in ss_by_struc.items():
        if chain not in ss_meta:
            ss_meta[chain] = {}
        for ss_structure, ss_structure_contents in ss_structures.items():
            ss_meta[chain][ss_structure + "_MAX_LENGTH"] = max(
                {id_stop - id_start + 1 for id_start, id_stop in ss_structure_contents}
            )
            ss_meta[chain][ss_structure + "_MIN_LENGTH"] = min(
                {id_stop - id_start + 1 for id_start, id_stop in ss_structure_contents}
            )
            ss_meta[chain]["NUM_" + ss_structure] = len(ss_structure_contents)

    # getting the sequence for a chain
    def get_sequences_per_chain(file_name):
        file_name.seek(0)
        seq_dict = {}
        seq_dict["CHAINS"] = []
        for line in file_name:
            if line[0:6] == "SEQRES":
                raw_line = line.split()
                chain = raw_line[2]
                for res in raw_line[4:]:
                    if chain + "_SEQUENCE" in seq_dict:
                        seq_dict[chain + "_SEQUENCE"].append(res)
                    else:
                        seq_dict[chain + "_SEQUENCE"] = [res]
                if chain + "_LENGTH" not in seq_dict:
                    seq_dict[chain + "_LENGTH"] = int(raw_line[3])
                if chain not in seq_dict["CHAINS"]:
                    seq_dict["CHAINS"].append(chain)
        return seq_dict

    sequence_dictionary = get_sequences_per_chain(pdb_file)

    # getting the longest number of repeats for each present amino acid per chain
    aa_repeats = {}
    for chain in sequence_dictionary["CHAINS"]:
        current_chain_aa_repeats = {}
        chain_sequence = sequence_dictionary[chain + "_SEQUENCE"]
        count = 1
        for i in range(len(chain_sequence) - 1):
            if chain_sequence[i] == chain_sequence[i + 1]:
                count += 1
            else:
                if chain_sequence[i] in current_chain_aa_repeats:
                    current_chain_aa_repeats[chain_sequence[i]] = max(
                        current_chain_aa_repeats[chain_sequence[i]], count
                    )
                else:
                    current_chain_aa_repeats[chain_sequence[i]] = count
                count = 1
        if chain_sequence[-1] in current_chain_aa_repeats:
            current_chain_aa_repeats[chain_sequence[-1]] = max(
                current_chain_aa_repeats[chain_sequence[-1]], count
            )
        else:
            current_chain_aa_repeats[chain_sequence[-1]] = count
        aa_repeats[chain] = current_chain_aa_repeats

    # getting visualizations
    if visualize:
        chain_sse = {}

        for chain in sequence_dictionary["CHAINS"]:
            length = sequence_dictionary[chain + "_LENGTH"]

            helices_idx = set()
            helices_loc = ss_by_struc[chain].get("HELIX")
            if helices_loc is not None:
                for start, stop in helices_loc:
                    for idx in range(start, stop + 1):
                        helices_idx.add(idx)

            beta_loc = ss_by_struc[chain].get("SHEET")
            beta_idx = set()
            if beta_loc is not None:
                for start, stop in beta_loc:
                    for idx in range(start, stop + 1):
                        beta_idx.add(idx)

            sse_list = []
            for i in range(length):
                if i + 1 in helices_idx:
                    sse_list.append("a")
                elif i + 1 in beta_idx:
                    sse_list.append("b")
                else:
                    sse_list.append("c")

            visualize_secondary_structure(np.array(sse_list, dtype="<U1"), 1, chain)

    return ss_by_struc, ss_meta, sequence_dictionary, aa_repeats
