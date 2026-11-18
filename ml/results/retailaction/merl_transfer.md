# MERL shelf-action model on RetailAction cameras

104 clips x 2 ceiling views = 208 videos from the verified RetailAction train archive (take 30, put 30, touch 14, no action 30). The MERL model never saw this data. Clip level: does Shop publish a reach / retract / hand-in-shelf episode?

| | As uploaded | Time-corrected | MERL reference |
| --- | ---: | ---: | ---: |
| Clip AP (action vs no action) | 0.680 | 0.673 | chance 0.712 |
| Episode published · take | 62% | 28% | |
| Episode published · put | 67% | 43% | |
| Episode published · touch | 54% | 32% | |
| Episode published · none | 65% | 43% | |
| Peak step inside label | 14% | 20% | label covers 32% |
| View agreement rank0/rank1 | 53% | 62% | |
| Person detected (steps) | 74% | 73% | 56% |
| Global feature shift (z) | 2.70 | 2.70 | 0.82 |
| Motion feature shift (z) | 1.80 | 0.63 | 0.20 |

## Where it fails

- No transfer: clip-level AP 0.680 (as uploaded) and 0.673 (time-corrected) vs 0.712 for a random scorer.
- Not discriminative: an interaction episode is published for 61% of action clips and 65% of no-action clips (as uploaded).
- Wrong timing: the strongest interaction step falls inside the labelled interval in 14% of action clips, although labels cover 32% of each clip.
- Views disagree: rank0 and rank1 of the same clip agree on 'interaction / no interaction' in only 53% of clips.
- Frame timing breaks the motion features: frames are 0.25 s apart vs 0.033 s in MERL, so global frame-difference energy is 7.4x MERL's; reach_to_shelf fires in 73% of all clips.
- Appearance shift: global backbone features sit 2.70 standard deviations from MERL's training mean vs 0.82 for MERL test (fisheye ceiling cameras, rotated/upside-down shoppers, other stores).
- Re-timing to 30 fps fixes the motion statistics (motion z 1.80 -> 0.63, MERL 0.20) and cuts false episodes to 43%, but AP stays at chance: appearance and viewpoint, not only timing, block transfer.
- Clips are 7.9 s long on average, shorter than the model's 12 s causal context, so most predictions use a partially empty context.
- Person visibility matters: visible <50% of steps AP 0.52 vs chance 0.63 (43 videos); visible >=90% AP 0.81 vs chance 0.70 (100 videos). Only fully visible shoppers carry a weak signal; the overall detector rate (74% of steps vs MERL 56%) hides views where a top-down or rotated shopper is missed.

Label assumption: RetailAction interval bounds are fractions of the 32-frame index (all observed values are multiples of 1/32).
This is a transfer probe on 104 clips, not a benchmark; the RetailAction test split was not read.
