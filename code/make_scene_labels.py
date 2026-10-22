"""Template of the scene labels used by analyze_vlm_ssm.py (--scene_labels): one row per extracted segment with
Waymo's own labels from context.json (time_of_day, location, weather_waymo) and an empty 'weather' column for our own
label. Waymo's weather field is not reliable: 13184115878756336167_1354_000_1374_000 is labelled 'sunny', but the
road is wet and there are rain drops on the lens. So 'weather' is labelled by a person from the images.
Rows already in <out_csv> are kept as they are (labels are never overwritten); new segments are appended.

usage: python -I make_scene_labels.py <extracted_root> <labels/scene_labels.csv>
"""
import glob
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
COLS = ['segment', 'time_of_day', 'location', 'weather_waymo', 'weather', 'weather_note', 'labeled_by']


def main():
    root, out = sys.argv[1], sys.argv[2]
    rows = []
    for p in sorted(glob.glob(os.path.join(root, '*', 'context.json'))):
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        rows.append(dict(segment=d['segment'], time_of_day=d.get('time_of_day', ''), location=d.get('location', ''),
                         weather_waymo=d.get('weather', ''), weather='', weather_note='', labeled_by=''))
    new = pd.DataFrame(rows, columns=COLS)
    if os.path.exists(out):
        old = pd.read_csv(out, dtype=str, keep_default_na=False)
        new = new[~new.segment.isin(old.segment)]
        table = pd.concat([old, new], ignore_index=True)[COLS]
    else:
        table = new
    table.to_csv(out, index=False)
    print(f'{out}: {len(table)} segments ({len(new)} new), {int((table.weather == "").sum())} without a weather label')


if __name__ == '__main__':
    main()
