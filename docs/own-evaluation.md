# Own staged evaluation set

MERL test numbers describe one mock store and one overhead camera. To see whether they transfer
to *your* demo camera, record a small held-out set with permission and score it with the same
runtime code the app uses. Do this before downloading more public datasets.

## 1. Permission and safety

- Written permission from the location owner and every person recorded (adult volunteers only).
  Keep the record outside Git and reference it in the manifest's `consent_record`.
- No real customers or staff in frame. No audio. Keep footage local or on the project Drive.
- Footage is used for evaluation only and is never merged into training data.

## 2. What to record (single fixed camera, demo height and angle)

| Scenario tag | What happens | Why it matters |
| --- | --- | --- |
| `reach` | Reach to a shelf, pause, withdraw | Core `reach_to_shelf` / `retract_from_shelf` |
| `take_and_return` | Pick a product, hold it, put it back | `hand_in_shelf`, `inspect_product`, `retract_from_shelf` |
| `inspect` | Look at the shelf and at a product without taking it | `inspect_shelf` vs `inspect_product` |
| `normal_pass` | Walk past, browse, no interaction | False episodes per minute on benign footage |
| `occlusion` | Another person or a trolley blocks the view | Visibility warnings and misses |
| `low_light` | Dimmed lights or backlight | Detector and motion robustness |

Aim for at least 4 sessions with at least 3 different people, 1–3 minutes each, and at least one
`normal_pass` recording per session. Name files `YYYY-MM-DD_<place>_s<session>_p<person>.mp4`.

## 3. Split rule

Split by **session and person**, never by clip. A session or person in the evaluation manifest
must not appear in any future training data. `ml/eval/own_eval.py` rejects a manifest that lists
an evaluation session or person under `training_sessions` / `training_people`.

## 4. Annotate

```bash
.venv/bin/python -I ml/eval/own_eval.py template --out var/own_eval/via_template.json
```

Open the template in VIA (`annotations/via/via_video_annotator.html`, see `docs/data-collection.md`),
add the recording, and mark each of the five MERL actions as a temporal segment. Leave periods
without shelf interaction unlabelled. Save one project per recording.

## 5. Evaluate

Copy `ml/eval/own_eval_manifest.example.json` to `var/own_eval/manifest.json`, edit it, then run:

```bash
.venv/bin/python -I ml/eval/own_eval.py run --manifest var/own_eval/manifest.json
```

This writes `ml/results/own_eval/summary.json`: per-class frame AP, segment precision/recall at
IoU 0.5, results per scenario tag, false episodes on `normal_pass` recordings, person visibility
and input warnings per recording. The **Research insights** page shows it automatically.

Report the counts alongside every number. With a handful of recordings the intervals are wide;
the purpose is to find out whether transfer is roughly working or clearly broken, not to claim an
accuracy figure.
