"""Assemble the proportioned final evaluation benchmark suite (agentix_final).

Combines scientifically validated cases across all critical non-obvious categories:
1. Sediment Scour & Water Discoloration (PODDS / Boxall & Saul shear-stress model)
2. Pipe Material Fatigue & Historical Failure Records (RAG records.md, Weibull hazard)
3. Geotechnical Subsidence & Subterranean Voids (Terzaghi internal piping, road damage)
4. Adversarial Decoys & Multi-Source Clutter (acoustic & sprinkler distraction web)
5. Downhill Bedding Dispersion (topographic DEM & gravel bedding overland travel)
6. Clean False Alarms (triage & phone callback debunking)
"""
import json
import os
import shutil

OUT = "data/benchmark/agentix_final"
os.makedirs(f"{OUT}/cases", exist_ok=True)
os.makedirs(f"{OUT}/hidden", exist_ok=True)

# Copy layouts and records from agentix_v4h
shutil.copy("data/benchmark/agentix_v4h/layouts.json", f"{OUT}/layouts.json")
shutil.copy("data/benchmark/agentix_v4h/records.md", f"{OUT}/records.md")
if os.path.exists("data/benchmark/agentix_v4h/hidden/_weak_pipes.json"):
    shutil.copy("data/benchmark/agentix_v4h/hidden/_weak_pipes.json", f"{OUT}/hidden/_weak_pipes.json")

# Define proportioned cases
SELECTIONS = [
    # 1. Sediment Scour / Discolouration (Boxall PODDS Model)
    ("agentix_v4h", "dev-brown-000", "sediment_scour"),
    ("agentix_v4h", "dev-brown-001", "sediment_scour"),
    ("agentix_v4h", "dev-brown-002", "sediment_scour"),
    ("agentix_v4h", "dev-brown-003", "sediment_scour"),
    ("agentix_v4h", "dev-brown-004", "sediment_scour"),

    # 2. Pipe Material Fatigue (RAG failure history records.md)
    ("agentix_v4h", "dev-chronic-000", "pipe_material_history"),
    ("agentix_v4h", "dev-chronic-001", "pipe_material_history"),
    ("agentix_v4h", "dev-chronic-002", "pipe_material_history"),
    ("agentix_v4h", "dev-chronic-003", "pipe_material_history"),
    ("agentix_v4h", "dev-chronic-004", "pipe_material_history"),

    # 3. Geotechnical Subsidence / Subsurface Voids (Potholes, road damage, telecom pit)
    ("agentix_v4h", "dev-chronic-005", "geotechnical_subsidence"),
    ("agentix_v4h", "dev-chronic-006", "geotechnical_subsidence"),
    ("agentix_v4h", "dev-chronic-007", "geotechnical_subsidence"),
    ("agentix_v4h", "dev-chronic-008", "geotechnical_subsidence"),

    # 4. Adversarial Decoys (Acoustic & moisture distraction web)
    ("agentix_cd", "dev-cd_sound_decoy-000", "adversarial_decoy"),
    ("agentix_cd", "dev-cd_sound_decoy-002", "adversarial_decoy"),
    ("agentix_cd", "dev-cd_sound_decoy-004", "adversarial_decoy"),
    ("agentix_cd", "dev-cd_sound_decoy-006", "adversarial_decoy"),
    ("agentix_cd", "dev-cd_damp_decoy-000", "adversarial_decoy"),
    ("agentix_cd", "dev-cd_damp_decoy-002", "adversarial_decoy"),

    # 5. Downhill Bedding Dispersion (Topographic displacement)
    ("agentix_v4h", "dev-displaced-000", "downhill_dispersion"),
    ("agentix_v4h", "dev-displaced-001", "downhill_dispersion"),
    ("agentix_v4h", "dev-displaced-002", "downhill_dispersion"),

    # 6. Clean False Alarms (Citizen callback triage)
    ("agentix_v4h", "dev-false_alarm-000", "clean_false_alarm"),
    ("agentix_v4h", "dev-false_alarm-001", "clean_false_alarm"),
    ("agentix_v4h", "dev-false_alarm-002", "clean_false_alarm"),
    ("agentix_v4h", "dev-false_alarm-003", "clean_false_alarm"),
    ("agentix_v4h", "dev-false_alarm-004", "clean_false_alarm"),
]

dev_cases = []
category_manifest = {}

for source_bench, case_id, cat in SELECTIONS:
    dev_cases.append(case_id)
    category_manifest[case_id] = {
        "category": cat,
        "source": source_bench,
    }
    
    # Copy public case dir
    src_case = f"data/benchmark/{source_bench}/cases/{case_id}"
    dst_case = f"{OUT}/cases/{case_id}"
    os.makedirs(dst_case, exist_ok=True)
    for f in os.listdir(src_case):
        s = os.path.join(src_case, f)
        d = os.path.join(dst_case, f)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy(s, d)

    # Copy hidden files
    for ext in (".json", ".npz", ".realistic.json"):
        s = f"data/benchmark/{source_bench}/hidden/{case_id}{ext}"
        d = f"{OUT}/hidden/{case_id}{ext}"
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)

# Write dev.json
with open(f"{OUT}/dev.json", "w") as f:
    json.dump({"cases": dev_cases}, f, indent=2)

# Write manifest
with open(f"{OUT}/category_manifest.json", "w") as f:
    json.dump(category_manifest, f, indent=2)

print(f"Successfully assembled {len(dev_cases)} cases in {OUT}!")
