import pandas as pd
from src.preprocess import preprocess

def test_preprocess(tmp_path):
    df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None], 'target': [0, 1, 0]})
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    df.to_csv(raw / "raw_data.csv", index=False)

    preprocess(str(raw), str(processed))
    pd.testing.assert_frame_equal(
        pd.read_csv(str(processed / "processed_data.csv")),
        df.dropna().reset_index(drop=True)
    )
