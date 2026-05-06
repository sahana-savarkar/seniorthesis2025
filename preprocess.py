import pandas as pd
import chardet

# minimal cleaning for BERTweet
def clean_for_bertweet(text):
    return str(text).strip()


def preprocess_data(
    file_path,
    text_column=None,
    save_path="cleaned_dataset.csv"
):
    """
    General-purpose preprocessing adapted for BERTweet.
    Only minimal text cleaning is applied.
    Virality labels are AUTOMATICALLY generated from retweets + quotes.
    """

    # detect encoding
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(500000))
    encoding = result["encoding"]
    print("Detected encoding:", encoding)

    df = pd.read_csv(file_path, encoding=encoding or "utf-8", low_memory=False)
    print("Loaded dataset. Columns detected:")
    print(df.columns.tolist())

    # ------------------------------
    # Detect text column automatically
    # ------------------------------
    if text_column is None:
        possible = ["text", "tweet", "content", "body", "message"]
        for col in df.columns:
            if col.lower() in possible:
                text_column = col
                break

    if text_column is None:
        raise ValueError("Could not auto-detect text column. Provide text_column explicitly.")

    print(f"Using text column: {text_column}")

    # create virality label
    if "retweets" in df.columns and "quotes" in df.columns:
        df["virality_raw"] = df["retweets"].fillna(0) + df["quotes"].fillna(0)
    else:
        raise ValueError("Dataset must contain 'retweets' and 'quotes'.")

    # quantile-based binning
    try:
        df["virality"] = pd.qcut(
            df["virality_raw"],
            q=3,
            labels=[0, 1, 2],
            duplicates="drop"
        )

        if df["virality"].nunique() < 3:
            raise Exception("Quantiles collapsed due to duplicate edges.")
        print("Used quantile-based virality bins.")

    except Exception:
        print("Quantile bins failed. Switching to rule-based bins.")

        # compute numeric thresholds
        low_thr = df["virality_raw"].quantile(0.70)
        high_thr = df["virality_raw"].quantile(0.90)

        def rule_based(v):
            if v <= low_thr:
                return 0  #low virality
            elif v <= high_thr:
                return 1  #medium virality
            else:
                return 2  #high virality

        df["virality"] = df["virality_raw"].apply(rule_based)

    df["virality"] = df["virality"].astype(int)
    print("Created virality labels (0, 1, 2).")

    # clean text for BERTweet
    df["clean_text"] = df[text_column].apply(clean_for_bertweet)
    print("Clean text created (BERTweet-safe).")

    df.to_csv(save_path, index=False)
    print(f"Saved cleaned dataset -> {save_path}")
    return df


if __name__ == "__main__":
    preprocess_data(
        file_path="tweets_with_sentiment.csv",
        text_column="text",  # can be changed!
        save_path="cleaned_tweets.csv"
    )