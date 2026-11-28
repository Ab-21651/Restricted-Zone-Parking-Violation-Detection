# Restricted Zone Definition Policy

**Status:** DRAFT — for team review. Zones are drawn only after this is approved,
and frozen after review (no changes after looking at test results).

## Purpose

A restricted zone is an area of the camera view where a vehicle that **stops and
stays** counts as a parking violation. Zones are defined from the **layout and
function of the scene** (lanes, curbs, entrances, markings), never from where
vehicles happen to park in the footage, so the evaluation is not biased toward the
test videos.

## R1 — What IS restricted

A ground area is restricted if a vehicle standing there would block movement,
access or safety:

| Rule | Area | Typical shape in the image |
|------|------|----------------------------|
| R1.1 | **Fire lanes / narrow access roads** that must stay passable (no space to park without blocking) | The full width of the drivable lane |
| R1.2 | **Road shoulder / curbside strip** along a no-parking curb, yellow/red line or sign | A strip about one car-width wide along the curb |
| R1.3 | **Entrances, gates and driveways** of buildings and parking lots | The area directly in front of the entrance |
| R1.4 | **Crosswalks** and the area of about one car-length before them | Box around the crossing |
| R1.5 | **Roundabout / intersection edges and corners**, where a stopped car forces others to swerve | Strip along the inner and outer edges |

## R2 — What is NOT restricted

| Rule | Area | Reason |
|------|------|--------|
| R2.1 | **Live traffic lanes** (the centre of a road where vehicles drive) | Vehicles stop there at junctions, in queues or to give way — that is traffic, not parking. Including lanes would cause false alerts. |
| R2.2 | **Marked / designated parking spaces and parking lots** | Parking there is legal |
| R2.3 | **Pavements, lawns, building areas** where vehicles cannot normally drive | Not needed; any vehicle there is out of scope |
| R2.4 | Areas **too far / too small** in the image for reliable detection (vehicle box smaller than ~15 px high) | The detector cannot judge them reliably; noted as a limitation |

Where a narrow road has no separate shoulder, R1.1 (fire lane) applies to the whole
lane and R2.1 does not — the road is too narrow for a stopped car to be "just traffic".

## R3 — How zones are drawn

1. Zones mark the **ground surface** (where wheels touch the road). The system tests
   the bottom-centre point of each vehicle box against the zone.
2. Keep a margin of about **half a car-width** away from neighbouring areas that are
   *not* restricted (e.g. a parking bay next to an entrance), so a legally parked car
   slightly over the line is not flagged.
3. Draw on a frame where the ground is clearly visible (no vehicles covering it).
4. One zone file per video (`configs/zones/<video>.yaml`). Videos from the same
   camera view reuse the same zone, re-aligned only if the camera moved.
5. Each zone gets a short name describing its rule, e.g. `fire-lane`, `curbside`,
   `entrance`.

## R4 — Violation rule

A **violation** = a vehicle whose ground point is inside a restricted zone and which
stays **stationary for at least 60 seconds** (the AVSS 2007 / i-LIDS Parked Vehicle
benchmark definition). 30 s is evaluated as a secondary setting.

Stationary = moved less than 15 % of its box height over 2 s. Brief occlusions
(< 5 s) do not reset the timer.

## R5 — Review and freezing

1. Zones are drawn per scene and documented in the table below with the rule used.
2. A second team member checks each zone against this policy.
3. After review, zone files are committed and **frozen**. Any later change is
   recorded here with the reason.

## Scene application (filled in when zones are drawn)

Drawn 2026-11-21 on the median (vehicle-free) frame of each video; review pending.
Overlay of all zones: `outputs/zones_work/all_zones.jpg` (regenerate with the snippet in the commit, or `scripts/run_pipeline.py --show`).

| Scene | Videos | Zone name(s) | Rules applied | Notes |
|-------|--------|--------------|---------------|-------|
| A Roundabout | 01–03 | `crosswalk-east`, `crosswalk-south-1/2/3`, `curbside-north` | R1.4, R1.5; lanes excluded (R2.1); car rows at lower-left treated as parking (R2.2) | Vehicles are only ~10–20 px high here (R2.4 borderline) — expect weak detection. **Decision (2026-11-21):** zones kept and scene A stays in the test set as the "hard / small-object" case; its results are reported separately. Higher input size and zone-cropped inference are evaluated on it in the optimisation step. 02/03 are browser screen recordings framed differently; zones mapped from 01 (x' = 1.02x + 2, y' = y + 58) and visually checked. |
| B Road with curb | 04–06 | `curbside-left`, `curbside-right` | R1.2 (yellow edge lines along both curbs); centre lanes excluded (R2.1) | Strip ≈ one car-width from each curb. |
| C Alley | 07–12 | `fire-lane` | R1.1 (too narrow for a shoulder, so the whole paved lane) | Walkway under the pergola on the right is excluded. A motorbike stands on the paved lane next to the wall in all six videos and is inside the zone — under R1.1 it counts as a violation. **Agreed by team (2026-11-21).** |
| D Campus road | 13 | `entrance`, `curbside-east` | R1.3 (parking-lot entrance), R1.2 (yellow curb line on the right) | The lot itself is legal parking (R2.2), with a margin from its front row (R3.2). |
| E Parking-lot area | 14 | `entrance` | R1.3 (driveway mouth of the lot) | Road edges have white lines only (no parking ban visible), so not restricted. |
| F Night road (IR) | 15–16 | `curbside` | R1.2 | **Assumption:** no-parking curb — markings are not visible in infrared. Strip widened on 2026-11-21 (see change log). |

## Change log

| Date | Zone | Change | Evidence (dev videos only) |
|------|------|--------|----------------------------|
| 2026-11-21 | Scene F `curbside` (ISLab-15, -16) | Lower edge moved from y = 370…425 to y = 387…439, i.e. strip width 108–111 px → 125 px | On **dev** video ISLab-15 the curb-parked car's ground point lies 114 px from the curb (y = 398 at x = 268), outside the original 103 px strip. R1.2 defines the strip as one car-width; the measured car-width + ~10 % replaces the estimate. The width comes from the dev measurement only. Test video ISLab-16 (same zone) was traced in the same diagnostic pass; the new width was **not** set to include it, and its car stays outside the strip. |

Not changed (test videos, frozen): the evaluation showed that the scene D `entrance` zone (ISLab-13) ends ~10 px above the ground point of the car parked in the entrance, and that the angled night car in ISLab-16 has its box bottom ~33 px beyond the widened strip. These are reported as failure cases, not fixed, because changing test zones after seeing test results would bias the evaluation.
