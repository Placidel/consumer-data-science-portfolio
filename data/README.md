# Generated data

`data/raw/` holds the shared Northstar Consumer tables (one Parquet file per table plus
`manifest.json` with row counts and content hashes). It is not committed; regenerate it with:

```bash
northstar generate-data
```

The same seed always reproduces byte-identical files. See
[`docs/data_dictionary.md`](../docs/data_dictionary.md) for the schema.
