# BridgeWalk

BridgeWalk retrieves evidence as a sequence of passages connected by entities. A shared cross-encoder chooses the next passage, identifies an entity to follow and predicts when the chain is complete. Selected entities determine which passages enter the next retrieval step.

## Code layout

| Path | Purpose |
| --- | --- |
| `code/bridgewalk.py` | Training examples, model training and chain inference |
| `code/prep_data.py` | Benchmark conversion |
| `code/openie.py`, `code/index.py` | Entity extraction, graph construction and embeddings |
| `code/unicor.py` | Corpus and graph access |
| `code/budget.py`, `code/reader.py` | Context packing and reader evaluation |
| `code/bridge_analysis.py` | Evidence-chain analysis |
| `code/chainwalk.py` | Fixed-neighborhood comparison model |
| `repos/` | GFM-RAG and HippoRAG comparison code |

## Environment and preparation

Use CUDA-enabled PyTorch with Transformers, sentence-transformers, NumPy and PEFT. Reader inference uses vLLM. Upstream comparison implementations have separate dependency files under `repos/`.

Run commands from the repository root. Download HotpotQA, 2WikiMultihopQA and MuSiQue from their providers, then use the preparation, extraction and indexing scripts to construct the passage collections and graphs. Check local dataset and model paths in the selected scripts before launching them.

Training reads each dataset from `data/train/<dataset>/`, including `index.pt`, `text_mentions.pt` and `questions.jsonl`. Dataset directory names are `hotpotqa`, `2wikimultihopqa` and `musique`. Inference uses `data/<dataset>/test.jsonl` with the corresponding indexes and candidate proposals in `results/<dataset>/chain_cand_v7.jsonl`.

## Build examples, train and retrieve

After preparing those inputs and a local cross-encoder initialization at `models/initial-cross-encoder`:

```bash
mkdir -p ckpt
python code/bridgewalk.py build
python code/bridgewalk.py train models/initial-cross-encoder ckpt/bridgewalk 0
python code/bridgewalk.py infer musique ckpt/bridgewalk bridgewalk full
```

The build step writes `ckpt/bridge_train.jsonl`. Training saves the encoder, tokenizer and task heads to the output directory. Inference writes context-budget variants under `results/<dataset>/`, using suffixes such as `_SB300` and `_SB600`.

`KB` controls how many bridge entities are followed, `RB` controls the passage expansion per bridge, and `DEPTH` sets the maximum chain depth. Set `READER` to your local reader model before using `code/reader.py`. The `full` inference variant enables the complete model; the other variants listed in `bridgewalk.py` support controlled comparisons.

## Optional remote helpers

Local training and inference do not require a remote connection. The optional SSH/SCP wrappers read `SSH_HOST`, `SSH_USER` and `SSH_PORT` from the environment; SCP also requires `SSH_REMOTE_DIR`. Configure these on your own machine and authenticate through your SSH agent or key. The repository contains no deployment endpoint or login settings.

Benchmark data, trained models, generated indexes and manuscript materials are not bundled with the code.

## License

GFM-RAG in `repos/gfm-rag/` retains its Apache 2.0 license. HippoRAG in `repos/HippoRAG/` retains its MIT license. A license for the project-specific code has not yet been specified.
