import pandas as pd


if __name__ == '__main__':
    df = pd.read_parquet("Data/LLMasJudgeOutFileHere.parquet")

    target_lines = [
        # line name here
    ]

    for line in target_lines:
        try:
            true_count = 0
            for row in df[line + "_LLM_Evaluate"].values:
                if "Review Consistent" in row:
                    true_count += 1
            print(f"{line}: {true_count}/{len(df)}")
        except KeyError:
            print(f"{line} not found in DataFrame")
            pass
