# STAGE A 완료 보고

STAGE_A_GATE: PASS
READY_FOR_PROBE_TEST_METRIC: YES

## Gate
- SMOKE: PASS
- CACHE: PASS
- B0: PASS
- B1: PASS
- B2: PASS
- B3: PASS
- FAIRNESS: PASS
- TEST_UNUSED: PASS

## Training Budget
- epochs/model: 1

## B0
- completed: PASS
- best val loss: 3.3570817891289204
- peak VRAM bytes: 422044160
- runtime seconds: 101.32504138513468
- best checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B0/best_val.pt
- last checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B0/last.pt

## B1
- completed: PASS
- best val loss: 3.3507321806514963
- peak VRAM bytes: 631509504
- runtime seconds: 38691.80868259305
- best checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B1/best_val.pt
- last checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B1/last.pt

## B2
- completed: PASS
- best val loss: 3.3492570456336526
- peak VRAM bytes: 631509504
- runtime seconds: 38653.555563159054
- best checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B2/best_val.pt
- last checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B2/last.pt

## B3
- completed: PASS
- best val loss: 3.351254482830272
- peak VRAM bytes: 631509504
- runtime seconds: 38624.627028347924
- best checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B3/best_val.pt
- last checkpoint: /home/itaein/Personal_Research/checkpoints/stage_a/B3/last.pt

## STOP
The Stage A gate performed no Behavior Cloning, action prediction, or test evaluation.
Stage B val selection and test95 final evaluation are tracked separately in analysis/stage_b/pipeline_status.json.
