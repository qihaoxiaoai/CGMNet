# cgmnet/utils/fingerprint.py
"""
A dedicated module for calculating various types of molecular fingerprints.
This module uses a registry pattern to manage different fingerprinting functions.
"""
import multiprocessing as mp
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem, ChemicalFeatures, rdMolDescriptors
from tqdm import tqdm

from cgmnet.utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

# ============================ FIX STARTS HERE ============================
# Suppress the RDKit deprecation warnings to keep the output clean.
RDLogger.DisableLog('rdApp.warning')
# ============================= FIX ENDS HERE =============================

# --- Fingerprint Dimensions ---
FINGERPRINT_DIMENSIONS: Dict[str, int] = {
    "ecfp": 1024,
    "rdkfp": 2048,
    "maccs": 167,
    "atom_pair": 2048,
    "torsion": 2048,
    "pharm": 3348,
    "md": 200,
}

# --- Fingerprint Function Registry ---
FINGERPRINT_REGISTRY: Dict[str, Callable] = {}

def register_fingerprint(name: str) -> Callable:
    """Decorator to register a fingerprint calculation function."""
    def decorator(fp_func: Callable) -> Callable:
        FINGERPRINT_REGISTRY[name] = fp_func
        return fp_func
    return decorator

# --- Fingerprint Calculation Functions ---

# ============================ FIX STARTS HERE ============================
# Reverted to the original, stable RDKit API calls that are proven to work.
@register_fingerprint("ecfp")
def calculate_ecfp(mol: Chem.Mol) -> np.ndarray:
    """Calculates the ECFP (Morgan) fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_DIMENSIONS["ecfp"])
    return np.array(fp, dtype=np.float32)

@register_fingerprint("torsion")
def calculate_torsion(mol: Chem.Mol) -> np.ndarray:
    """Calculates the topological torsion fingerprint."""
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=FINGERPRINT_DIMENSIONS["torsion"])
    return np.array(fp, dtype=np.float32)
# ============================= FIX ENDS HERE =============================

@register_fingerprint("rdkfp")
def calculate_rdkfp(mol: Chem.Mol) -> np.ndarray:
    """Calculates the RDKit fingerprint."""
    fp = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=FINGERPRINT_DIMENSIONS["rdkfp"])
    return np.array(fp, dtype=np.float32)

@register_fingerprint("maccs")
def calculate_maccs(mol: Chem.Mol) -> np.ndarray:
    """Calculates MACCS keys."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)

@register_fingerprint("atom_pair")
def calculate_atom_pair(mol: Chem.Mol) -> np.ndarray:
    """Calculates the atom-pair fingerprint."""
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=FINGERPRINT_DIMENSIONS["atom_pair"])
    return np.array(fp, dtype=np.float32)

@register_fingerprint("pharm")
def calculate_pharmacophore(mol: Chem.Mol) -> np.ndarray:
    """Calculates the 2D pharmacophore fingerprint."""
    fdef_name = rdkit.RDConfig.RDDataDir + '/BaseFeatures.fdef'
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    sig_factory = SigFactory(factory, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
    sig_factory.SetBins([(0, 2), (2, 5), (5, 8)])
    sig_factory.Init()
    fp = Generate.Gen2DFingerprint(mol, sig_factory)
    return np.array(fp, dtype=np.float32)

@register_fingerprint("md")
def calculate_molecular_descriptors(smiles: str) -> np.ndarray:
    """Calculates the 200 RDKit 2D normalized molecular descriptors."""
    generator = RDKit2DNormalized()
    results = generator.process(smiles)
    if results and results[0]:
        return np.array(results[1:], dtype=np.float32)
    return np.zeros(FINGERPRINT_DIMENSIONS["md"], dtype=np.float32)

# --- High-Level API ---

def get_fingerprint(smiles: str, name: str) -> np.ndarray:
    """
    A unified function to compute a single fingerprint for a given SMILES string.
    """
    if name not in FINGERPRINT_REGISTRY:
        raise ValueError(f"Fingerprint '{name}' not registered. Available: {list(FINGERPRINT_REGISTRY.keys())}")

    fp_function = FINGERPRINT_REGISTRY[name]
    
    if name == 'md':
        return fp_function(smiles)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(FINGERPRINT_DIMENSIONS[name], dtype=np.float32)
    return fp_function(mol)

def get_batch_fingerprints(smiles_list: List[str], name: str, n_jobs: int = 16) -> np.ndarray:
    """
    Computes a specific fingerprint for a batch of SMILES strings in parallel.
    """
    worker_func = partial(get_fingerprint, name=name)
    
    with mp.Pool(n_jobs) as pool:
        results = list(tqdm(
            pool.imap(worker_func, smiles_list),
            total=len(smiles_list),
            desc=f"Calculating {name.upper()} fingerprints"
        ))
                            
    return np.stack(results, axis=0)
