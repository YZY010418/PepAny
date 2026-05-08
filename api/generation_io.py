"""Conversion helpers for generated model outputs."""

from rdkit import Chem

from models.LDM.data_utils import OverwriteTask

from .pdb_dataset import ComplexDesc
from .templates import BaseTemplate


def data_to_cplx(cplx_desc: ComplexDesc, template: BaseTemplate, S: list, X: list, A: list, ll: list, inter_bonds: tuple, intra_bonds: tuple, out_path: str, check_validity: bool=False, check_filters: bool=False):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    task = OverwriteTask(
        cplx = cplx_desc.cplx,
        select_indexes = cplx_desc.pocket_block_ids + cplx_desc.lig_block_ids,
        generate_mask = cplx_desc.generate_mask,
        target_chain_ids = cplx_desc.tgt_chains,
        ligand_chain_ids = cplx_desc.lig_chains,
        S = S,
        X = X,
        A = A,
        ll = ll,
        inter_bonds = inter_bonds,
        intra_bonds = intra_bonds,
        out_path = out_path
    )

    def template_filter(cplx):
        cplx_desc.cplx = cplx
        return template.validate(cplx_desc)

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        filters = [template_filter] if check_filters else None
    )

    if cplx is None or gen_mol is None:
        return None, None

    cplx_desc.cplx = cplx

    details = {
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq()
    }

    return cplx_desc, details


def format_log(cplx_desc: ComplexDesc, details: dict, n: int):
    log = {
        'id': cplx_desc.id,
        'n': n,
        'pmetric': details['pmetric'],
        'smiles': details['smiles'],
        'gen_seq': details['gen_seq'],
        'tgt_chains': cplx_desc.tgt_chains,
        'lig_chains': cplx_desc.lig_chains,
        'gen_block_idx': cplx_desc.lig_block_ids,
    }
    for key, value in details.items():
        if key not in log:
            log[key] = value
    return log
