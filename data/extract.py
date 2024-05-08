import json

import pandas as pd


def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def extract_sextuples(dialogue_data):
    sextuples = []
    for entry in dialogue_data:
        if "hexatuple" in entry:
            for hexa in entry["hexatuple"]:
                sextuple = {
                    "holder": hexa["holder"]["value"],
                    "target": hexa["target"]["value"],
                    "aspect": hexa["aspect"]["value"],
                    "opinion": hexa["opinion"]["value"],
                    "sentiment": hexa["sentiment"],
                    "rationale": hexa["rationale"]["value"],
                }
                sextuples.append(sextuple)
    df = pd.DataFrame(sextuples)
    df.to_csv("extracted_sextuples.csv", index=False)
    return sextuples


def extract_sentiment_flips(dialogue_data):
    flips = []
    for entry in dialogue_data:
        if "sentiment flip" in entry:
            for flip in entry["sentiment flip"]:
                sentiment_flip = {
                    "holder": flip["holder"],
                    "target": flip["target"],
                    "aspect": flip["aspect"],
                    "initial_sentiment": flip["initial sentiment"],
                    "flipped_sentiment": flip["flipped sentiment"],
                    "trigger": flip["trigger type"],
                }
                flips.append(sentiment_flip)
    df = pd.DataFrame(flips)
    df.to_csv("sentiment_flips.csv", index=False)
    return flips


if __name__ == "__main__":
    file_path = "xxxxx.json"
    data = load_data(file_path)
    sextuples = extract_sextuples(data)
    sentiment_flips = extract_sentiment_flips(data)
