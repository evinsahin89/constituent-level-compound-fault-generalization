# Data setup

1. Download the public MCC5-THU motor dataset separately.
2. Create a local project root containing the raw dataset and generated analysis folders.
3. Update the `ROOT = Path(...)` value in each script to that local project root.
4. Run the analyses in the order described in the top-level `README.md`.

The raw dataset and large generated caches are intentionally not included in this repository.

The scripts retain the original local-path configuration rather than being structurally refactored, so that the verified analysis logic is not altered solely for code release.
