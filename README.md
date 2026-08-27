# Personal Research: Long-Horizon Mamba Memory

This repository contains a controlled RoboCerebra study of whether a causal
Mamba representation retains language-defined subtask information over long
robot-manipulation trajectories.

## Experimental conditions

| Variant | Language conditioning | Temporal context |
| --- | --- | --- |
| B0 | Full instruction at every frame | Fixed five-frame window |
| B1 | Full instruction at every frame | Persistent episode state |
| B2 | Current official step at every frame | Persistent episode state (oracle-like) |
| B3 | Official step only at a transition; exact-zero language during HOLD | Persistent episode state |

Stage A learns a future visual representation with one-way InfoNCE. Stage B
freezes the backbone, selects independent linear retrieval probes on the
validation split, and evaluates the selected probes exactly once on 95 held-out
test trajectories.

## One-epoch pilot result

| Variant | Current Recall@1 | Prev-1 | Prev-2 | Prev-3 | Current MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.2633 | 0.2914 | 0.2062 | 0.1811 | 0.4729 |
| B1 | 0.2722 | 0.2952 | 0.2086 | 0.1796 | 0.4790 |
| B2 | 0.3379 | 0.3167 | 0.2172 | 0.1811 | 0.5407 |
| B3 | 0.2711 | 0.2947 | 0.2099 | 0.1798 | 0.4787 |

B2 verifies that the conditioning path can materially affect current-step
retrieval, but it is an oracle-like condition and is not memory evidence. In
this one-epoch pilot, B3 does not separate from B1 on previous-step retrieval,
and neither persistent condition provides strong previous-step gains over B0.
The result establishes end-to-end feasibility, not convergence or a definitive
long-term-memory improvement.

Final tables and dependency-free SVG/HTML plots are under
[`analysis/stage_b/final_test`](analysis/stage_b/final_test).

## Repository layout

- `src/robocerebra_memory/`: data interfaces, conditioning, Mamba model, and metrics
- `configs/`: frozen experiment and evaluation configuration
- `tools/`: preparation, training, extraction, evaluation, and audit entry points
- `tests/`: CPU-oriented unit and pipeline tests
- `docs/`: protocol and implementation documentation
- `analysis/stage_b/final_test/`: held-out Test95 reports and visualizations

Large checkpoints, cached representations, logs, external repositories, source
videos, and local operational prompts are intentionally excluded from Git.

## Validation

The current RoboCerebra/Mamba scope passes 110 unit and pipeline tests in the
project's PyTorch environment. Two unrelated legacy fixture suites require
local LIBERO/RoboCasa annotation files and are not part of this result.
