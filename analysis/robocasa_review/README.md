# RoboCasa representative video review

These are official left-agent-view episode-0 MP4 samples extracted with HTTP byte ranges from the task-specific RoboCasa archives. Each MP4 has a matching 21-frame contact sheet.

Open an MP4 in a system player if the VS Code binary viewer cannot decode H.264. The contact sheet is only for continuity screening; inspect the MP4 near the following annotation boundary frames at 20 FPS.

| Task | Boundary frames |
|---|---|
| WashFruitColander | 122, 242, 476, 595, 752, 912, 1103, 1233 |
| WaffleReheat | 176, 351, 465, 589, 675 |
| StirVegetables | 102, 222, 361, 460, 632, 902 |
| HeatKebabSandwich | 160, 280, 368, 507, 626, 693, 918, 1202 |

At each boundary, check object/fixture state, contact completion, whether base movement occurred, and whether the next official natural-language phase begins at a visually defensible time. Exact decoded strings and times are in `../robocasa_semantic_stage_audit.json`.
