# Third-party atomic data

The standardized HDF5 files under `clean_data/` contain atomic energy-level and transition data derived from the following external resources:

- **AtomDB/APED v3.1.3**
- **CHIANTI v11.0.2**
- **SPEX/SPEXACT v3.08.01**
- **SPEX/SPEXACT v3.08.03**

The original data were converted from their source formats into the common CFADB HDF5 schema for cross-database scientific comparison.

## Scope of rights

The MIT License included with CFADB applies only to the original CFADB source code and documentation. It does not apply to the third-party atomic data contained in `clean_data/`.

Copyright, database rights, attribution requirements, and other rights in the underlying atomic data remain with the respective database developers, institutions, and original data providers. This archive does not grant additional rights to those third-party data.

## Attribution and citation

Users should cite the relevant database publications and identify the specific database versions used:

- AtomDB/APED v3.1.3
- CHIANTI v11.0.2
- SPEX/SPEXACT v3.08.01
- SPEX/SPEXACT v3.08.03

The accompanying article provides the corresponding scientific references and database descriptions. SPEX/SPEXACT v3.08.01 is used in the principal cross-database analysis, while selected standardized data from v3.08.03 are also retained in the archive for supplementary comparison.

## Derived dataset names

The standardized datasets are identified using the following names:

- `lev_atomdb313` and `tran_atomdb313`
- `lev_chianti1102` and `tran_chianti1102`
- `lev_spex30801` and `tran_spex30801`
- `lev_spex30803` and `tran_spex30803`

Some ion files may not contain all datasets because the corresponding source database does not provide data for every ion.

## Use of the archived data

The standardized tables are provided for scholarly review, reproducibility, and scientific comparison. Users remain responsible for complying with the terms and attribution requirements of the original data providers.
