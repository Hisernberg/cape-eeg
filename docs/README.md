# Planning documents (archived)

Documents 01–08 are the original study-design package written **before** any data were touched. They are kept verbatim as the record of what was planned. Where a number in these documents differs from the executed study, the executed values in `../README.md`, `../EVALUATION_REPORT.md`, `../results/aggregate/` and `../results/aggregate/protocol_lock.json` are canonical. Known differences:

| Item | Planning document | Executed study |
|---|---|---|
| Candidate parameter count | 164,198 (reference code estimate) | 164,134 (measured from `src/cape_eeg/model.py`) |
| Fixed-fusion control parameters | 147,524 | 147,524 (identical) |
| Synthetic test count | 58 (starter package) | 155 (this repository) |
| Physical batch | 32 with accumulation to 64 | 64 (profiled 0.44 GiB peak CUDA, no accumulation needed) |
| Training schedule | up to 12 epochs | 12-epoch development fits; complete 3-epoch cosine schedule frozen for final refits |
| Wording | "pre-registered" | "protocol-locked before held-out evaluation" (no external timestamped registry was used) |
