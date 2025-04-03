from group_htmls.grouping import group_similar_htmls

if __name__ == "__main__":
    # Rulează procesarea pe toate nivelurile specificate
    for tier in ['./clones/tier1', './clones/tier2', './clones/tier3', './clones/tier4']:
        print(f"Grouping for: {tier}")
        group_similar_htmls(tier, eps=2, min_samples=2, do_postprocessing=1)
