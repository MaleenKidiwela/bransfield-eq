"""Apply the flag_shot_v2 column to write v2 cleaned event+pick catalogs.

Reads:
    catalogs/pyocto_events_picker_only_with_shot_flag_v2.csv
    catalogs/pyocto_picks_picker_only.csv

Writes:
    catalogs/pyocto_events_picker_only_no_shots_v2.csv   (v2-cleaned full year)
    catalogs/pyocto_picks_picker_only_no_shots_v2.csv
    catalogs/pyocto_events_jan2019_noshot_v2.csv        (Jan 1-30 subset for backbone)
    catalogs/pyocto_picks_jan2019_noshot_v2.csv

And overwrites the canonical no-shots filenames so the downstream pipeline
picks the v2-cleaned data:
    catalogs/pyocto_events_picker_only_no_shots.csv      <- v2
    catalogs/pyocto_picks_picker_only_no_shots.csv       <- v2
    catalogs/pyocto_events_jan2019_noshot.csv            <- v2
    catalogs/pyocto_picks_jan2019_noshot.csv             <- v2
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv")
    pk = pd.read_csv(REPO / "catalogs" / "pyocto_picks_picker_only.csv")
    ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")

    is_clean = ~ev["flag_shot_v2"].astype(bool)
    print(f"Events kept after v2 cleaning: {is_clean.sum():,} / {len(ev):,}")

    clean_ev = ev[is_clean].drop(columns=["flag_shot", "flag_shot_v2", "prob_shot",
                                          "shot_idx", "shot_survey", "n_picks_used"],
                                  errors="ignore").reset_index(drop=True)
    clean_pk = pk[pk.event_idx.isin(set(clean_ev["event_idx"]))].reset_index(drop=True)

    # v2-suffixed files
    clean_ev.to_csv(REPO / "catalogs" / "pyocto_events_picker_only_no_shots_v2.csv", index=False)
    clean_pk.to_csv(REPO / "catalogs" / "pyocto_picks_picker_only_no_shots_v2.csv", index=False)

    # Canonical no_shots filenames (overwrite — downstream pipeline reads these)
    clean_ev.to_csv(REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv", index=False)
    clean_pk.to_csv(REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv", index=False)
    print(f"wrote pyocto_events_picker_only_no_shots.csv  ({len(clean_ev):,} events)")
    print(f"wrote pyocto_picks_picker_only_no_shots.csv   ({len(clean_pk):,} picks)")

    # Jan 2019 subset for backbone
    jan = (clean_ev["origin_time"] >= "2019-01-01") & (clean_ev["origin_time"] < "2019-01-31")
    jan_ev = clean_ev[jan].reset_index(drop=True)
    jan_pk = clean_pk[clean_pk.event_idx.isin(set(jan_ev["event_idx"]))].reset_index(drop=True)
    jan_ev.to_csv(REPO / "catalogs" / "pyocto_events_jan2019_noshot_v2.csv", index=False)
    jan_pk.to_csv(REPO / "catalogs" / "pyocto_picks_jan2019_noshot_v2.csv", index=False)
    jan_ev.to_csv(REPO / "catalogs" / "pyocto_events_jan2019_noshot.csv", index=False)
    jan_pk.to_csv(REPO / "catalogs" / "pyocto_picks_jan2019_noshot.csv", index=False)
    print(f"wrote pyocto_events_jan2019_noshot.csv         ({len(jan_ev):,} events)")
    print(f"wrote pyocto_picks_jan2019_noshot.csv          ({len(jan_pk):,} picks)")


if __name__ == "__main__":
    main()
