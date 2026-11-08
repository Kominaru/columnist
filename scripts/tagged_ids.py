import json, csv, sys
w = csv.writer(open(sys.argv[2], "w")); w.writerow(["beatmapset_id", "beatmap_id"])
for l in open(sys.argv[1]):
    j = json.loads(l)
    for b, t in j["beatmaps"].items():
        if sum(t.values()) >= 2:
            w.writerow([j["beatmapset_id"], b])
