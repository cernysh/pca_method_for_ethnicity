"""
PCA sketch: a minimal, working proof-of-concept of ancestry estimation via PCA,
run on a small slice of real 1000 Genomes data (chromosome 22 only).

This is intentionally a SKETCH, not the final pipeline:
- Only the first N simple SNPs from chromosome 22 are used (no LD pruning,
  no quality filtering beyond "biallelic SNP with no missing genotypes").
- It does not yet project an external raw-DNA file (e.g. AncestryDNA.txt)
  onto the same space - it only shows that real 1000 Genomes samples
  separate into their known continental groups, which is the first thing
  to prove before adding a new, unlabeled sample to the picture.


Run (from the project root):
    pip install numpy matplotlib
    python3 src/pca_sketch.py
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np


# File paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PANEL_FILE = PROJECT_ROOT / "data" / "1000genomes" / "integrated_call_samples_v3.20130502.ALL.panel"
VCF_FILE = PROJECT_ROOT / "data" / "1000genomes" / "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

OUTPUT_PLOT = PROJECT_ROOT / "output" / "pca_sketch.png"

# How many variants (SNP positions) to read before stopping. This keeps the
# sketch fast and memory-light. The real pipeline will use far more variants
# (after LD pruning), spread across all 22 chromosomes, not just chr22.
MAX_VARIANTS = 3000


# Step 1. Read the population-label file

def load_panel(path: Path) -> dict[str, dict[str, str]]:
    """
    Reads the .panel file: one row per person, with their population code
    (e.g. "GBR" = British in England and Scotland) and super-population
    (e.g. "EUR" = European). Returns {sample_id: {"pop": ..., "super_pop": ...}}.
    """
    info = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            info[row["sample"]] = {
                "pop": row["pop"],
                "super_pop": row["super_pop"],
            }
    return info



# Step 2. Stream the VCF file and turn genotypes into numbers.

def genotype_to_dosage(genotype_field: str) -> int | None:
    """
    Converts a VCF genotype like "0|0", "0|1" or "1|1" into a single number:
    0 = no copies of the alternative allele, 1 = one copy, 2 = two copies.
    Returns None for missing genotypes (so the caller can skip that variant).
    This simple rule only works for biallelic SNPs (exactly one alternative
    allele), which is exactly what the filter in stream_genotype_matrix keeps.
    """
    alleles = genotype_field.replace("|", "/").split("/")
    if "." in alleles:
        return None
    return alleles.count("1")


def is_biallelic_snp(ref_allele: str, alt_allele: str) -> bool:
    """A plain single-letter-to-single-letter substitution, not an indel
    and not a site with more than one alternative allele."""
    return len(ref_allele) == 1 and len(alt_allele) == 1


def count_qualifying_variants(vcf_path: Path) -> int:
    """
    First pass over the file: counts how many biallelic SNPs it contains in
    total, without yet looking at genotypes. This is needed to work out how
    far apart to space the variants we actually keep (see
    stream_genotype_matrix below), so that the final sample is spread across
    the whole chromosome instead of being bunched up at its very start.
    """
    count = 0
    with gzip.open(vcf_path, mode="rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 5)
            if is_biallelic_snp(fields[3], fields[4]):
                count += 1
    return count


def stream_genotype_matrix(vcf_path: Path, max_variants: int) -> tuple[list[str], np.ndarray]:
    """
    Reads the gzip-compressed VCF file one line at a time (streaming - the
    file is never fully decompressed to disk or held in memory as text).

    Variants are picked at an even spacing across the whole file (every
    `stride`-th qualifying SNP), not just the first ones encountered. The
    first attempt at this sketch kept the first 3000 SNPs in file order,
    which are all packed into a tiny region at the very start of the
    chromosome (about 1% of it) - that gave a misleading result, because it
    reflects one small local neighbourhood rather than genome-wide
    differences between populations.

    Returns (sample_ids, matrix) where matrix has shape
    (number_of_variants, number_of_samples).
    """
    print("  First pass: counting qualifying SNPs in the whole file...")
    total_qualifying = count_qualifying_variants(vcf_path)
    stride = max(1, total_qualifying // max_variants)
    print(f"  Found {total_qualifying} biallelic SNPs in total; "
          f"keeping every {stride}-th one to spread {max_variants} "
          f"across the whole chromosome.")

    sample_ids: list[str] | None = None
    rows: list[list[int]] = []
    qualifying_seen = 0

    with gzip.open(vcf_path, mode="rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                sample_ids = fields[9:]
                continue

            fields = line.rstrip("\n").split("\t")
            ref_allele, alt_allele = fields[3], fields[4]

            if not is_biallelic_snp(ref_allele, alt_allele):
                continue

            qualifying_seen += 1
            if qualifying_seen % stride != 0:
                # Not one of the evenly-spaced positions we picked - skip it.
                continue

            genotype_fields = fields[9:]
            dosages = [genotype_to_dosage(gt) for gt in genotype_fields]

            if any(d is None for d in dosages):
                # Skip variants with any missing genotype, to keep the
                # sketch simple. The real pipeline will handle this properly.
                continue

            rows.append(dosages)
            if len(rows) >= max_variants:
                break

    if sample_ids is None:
        raise RuntimeError("Could not find the #CHROM header line in the VCF file.")

    matrix = np.array(rows, dtype=float)  # shape: (variants, samples)
    return sample_ids, matrix


# ---------------------------------------------------------------------------
# Step 3. Run PCA.
# ---------------------------------------------------------------------------

def run_pca(variant_by_sample_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Computes the first `n_components` principal components.

    The input matrix has one row per variant and one column per sample, so
    it is transposed first to the usual "one row per sample" shape.

    This uses singular value decomposition (SVD) rather than explicitly
    building a (samples x samples) covariance matrix and finding its
    eigenvectors - the two approaches give the same principal components,
    but SVD is the numerically more stable, standard way to compute them
    (it is what most real PCA libraries do internally).
    """
    X = variant_by_sample_matrix.T  # now: (samples, variants)
    X_centered = X - X.mean(axis=0, keepdims=True)

    U, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)
    scores = U[:, :n_components] * singular_values[:n_components]
    return scores


# ---------------------------------------------------------------------------
# Step 4. Plot the result.
# ---------------------------------------------------------------------------

def plot_pca(scores: np.ndarray, super_pops: list[str], variants_used: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for super_pop in sorted(set(super_pops)):
        indices = [i for i, sp in enumerate(super_pops) if sp == super_pop]
        plt.scatter(scores[indices, 0], scores[indices, 1], s=12, label=super_pop)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA sketch: {len(scores)} samples, {variants_used} chr22 SNPs (1000 Genomes)")
    plt.legend(title="Super-population")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    for required_file in (PANEL_FILE, VCF_FILE):
        if not required_file.exists():
            print(f"ERROR: file not found: {required_file}")
            return

    print("Reading population labels...")
    sample_info = load_panel(PANEL_FILE)
    print(f"  People with known population: {len(sample_info)}")

    print(f"Streaming genotypes from {VCF_FILE.name} ...")
    print("  (reading line by line, the .vcf.gz is never fully unzipped to disk)")
    sample_ids, matrix = stream_genotype_matrix(VCF_FILE, MAX_VARIANTS)
    print(f"  Samples found in the VCF: {len(sample_ids)}")
    print(f"  Usable SNPs collected: {matrix.shape[0]} (stopped at MAX_VARIANTS={MAX_VARIANTS})")

    print("Running PCA...")
    scores = run_pca(matrix, n_components=2)

    super_pops = [sample_info[sample_id]["super_pop"] for sample_id in sample_ids]

    print("Saving plot...")
    plot_pca(scores, super_pops, matrix.shape[0], OUTPUT_PLOT)
    print(f"Done. Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
