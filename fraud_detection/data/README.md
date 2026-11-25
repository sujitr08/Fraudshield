# Dataset directory

The included CSV files are the supplied project datasets. The application also accepts the classic credit-card schema from the report when uploaded:

```text
Time, V1, V2, ..., V28, Amount, Class
```

The PDF reports 284,807 rows and 492 fraud cases for that reference dataset. The exact source file was not present in the supplied attachments, so it is intentionally not fabricated or relabelled here. Upload it in the Batch Review page to use its original fields.

Alternatively, from the project root run `python scripts/download_reference_dataset.py`; it downloads the reference file and validates both the row count and fraud count before accepting it.
