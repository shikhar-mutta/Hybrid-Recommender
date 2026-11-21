"""Vocabulary (genres, tags) for the searchable dropdowns of the web app."""
import json, os, sys
sys.path.insert(0, "src")
import pandas as pd
import hybrid_rec as H

R, M, T = H.load_data()
genres = sorted({g for gs in M.genres for g in gs.split("|") if g != "(no genres listed)"})
cnt = T.tag.astype(str).str.lower().str.strip().value_counts()
tags = [t for t, c in cnt.items() if c >= 2 and 2 <= len(t) <= 40]
for t in ["pixar", "toys", "fun", "assassin", "hitman", "gun fu", "keanu reeves", "wrestling", "sports", "father daughter", "india", "inspirational", "bollywood"]:
    if t not in tags:
        tags.append(t)                      # tags used by the demo presets
tags = sorted(set(tags))
os.makedirs("web/public/data", exist_ok=True)
json.dump({"genres": genres, "tags": tags}, open("web/public/data/vocab.json", "w"), separators=(",", ":"))
print(len(genres), "genres,", len(tags), "tags")
