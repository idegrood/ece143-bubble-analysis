"""
bubble_data_collector.py

Collects CRSP + Compustat data for all bubbles defined in bubble_config.json,
builds a daily merged panel for each bubble, and saves each as a CSV file.

Usage:
    python bubble_data_collector.py
"""

import os
import json
import wrds
import pandas as pd


# ------------------------------
# Config helpers
# ------------------------------

def load_bubble_config(config_path: str = "bubble_config.json") -> dict:
    """
    Load the JSON configuration file defining company keywords, tickers,
    and names for each bubble type.

    Parameters
    ----------
    config_path : str
        Path to the bubble_config.json file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Basic sanity checks
    for bubble_type, cfg in config.items():
        if "keywords" not in cfg or "tickers" not in cfg or "names" not in cfg:
            raise ValueError(
                f"Config for '{bubble_type}' must contain 'keywords', 'tickers', and 'names'."
            )
        if not isinstance(cfg["keywords"], list):
            raise TypeError(f"'keywords' for '{bubble_type}' must be a list.")
        if not isinstance(cfg["tickers"], list):
            raise TypeError(f"'tickers' for '{bubble_type}' must be a list.")
        if not isinstance(cfg["names"], list):
            raise TypeError(f"'names' for '{bubble_type}' must be a list.")

    return config


def get_bubble_config(bubble_type: str, config: dict) -> dict:
    """
    Get the config entry for one bubble type and force names to uppercase.

    Parameters
    ----------
    bubble_type : str
        Key in the config dictionary (e.g. 'ai', 'crypto').
    config : dict
        Full configuration dict loaded from JSON.

    Returns
    -------
    dict
        Config for the requested bubble with uppercase names.
    """
    if bubble_type not in config:
        raise KeyError(f"Unknown bubble type: {bubble_type}")

    cfg = config[bubble_type].copy()
    cfg["names"] = [n.upper() for n in cfg.get("names", [])]
    return cfg


# ------------------------------
# Core panel builder
# ------------------------------

def build_bubble_panel(
    bubble_type: str,
    db: wrds.Connection | None = None,
    config: dict | None = None,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Build the merged CRSP + Compustat daily bubble dataset for a given bubble type.

    Parameters
    ----------
    bubble_type : str
        Name of the bubble (e.g., 'ai', 'crypto') as defined in bubble_config.json.
    db : wrds.Connection or None
        Active WRDS connection. If None, a new connection is created.
    config : dict or None
        Parsed configuration dictionary. If None, loaded from bubble_config.json.
    start_date : str
        Earliest trading date to pull from CRSP.dsf (YYYY-MM-DD).

    Returns
    -------
    pandas.DataFrame
        Merged CRSP+Compustat panel including valuation metrics and bubble_index.
    """
    # Load config if not provided
    if config is None:
        config = load_bubble_config()

    cfg = get_bubble_config(bubble_type, config)
    keywords = cfg["keywords"]
    tickers = cfg["tickers"]
    names = cfg["names"]

    if not keywords or not tickers or not names:
        raise ValueError(
            f"Config for '{bubble_type}' must have non-empty keywords, tickers, and names."
        )

    # WRDS connection
    if db is None:
        db = wrds.Connection()

    # 1) Compustat name search
    pattern = "|".join(keywords)

    sql_universe = f"""
        SELECT DISTINCT gvkey, conm, tic
        FROM comp.fundq
        WHERE conm ~* '{pattern}'
        ORDER BY conm;
    """
    firm_list = db.raw_sql(sql_universe)
    print(f"{bubble_type} initial matches:")
    print(firm_list.head())

    # 2) Link to CRSP
    link = db.get_table(
        library="crsp",
        table="ccmxpf_linktable",
        columns=["gvkey", "lpermno", "linktype", "linkprim"],
    )

    link = link[
        (link["linktype"].isin(["LC", "LU"])) &
        (link["linkprim"].isin(["P", "C"]))
    ][["gvkey", "lpermno"]].drop_duplicates().rename(columns={"lpermno": "permno"})

    firm_list = firm_list.merge(link, on="gvkey", how="left")
    firm_list = firm_list.dropna(subset=["permno"]).copy()
    firm_list["permno"] = firm_list["permno"].astype(int)

    # 3) Curated filter (this is what really defines your universe)
    firm_list = firm_list[
        (firm_list["tic"].isin(tickers)) |
        (firm_list["conm"].str.upper().isin(names))
    ].drop_duplicates(subset=["permno"])

    if firm_list.empty:
        raise AssertionError(f"No firms passed the curated filter for '{bubble_type}'")

    print(f"\nFiltered {bubble_type} firms (curated):")
    print(firm_list)

    # 4) CRSP daily
    permnos = firm_list["permno"].tolist()
    permnos_str = ",".join(map(str, permnos))

    sql_crsp = f"""
        SELECT permno, date, ret, retx, prc, vol, shrout
        FROM crsp.dsf
        WHERE permno IN ({permnos_str})
          AND date >= '{start_date}'
        ORDER BY permno, date;
    """
    crsp = db.raw_sql(sql_crsp)
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["prc"] = crsp["prc"].abs()
    # NOTE: shrout is in thousands of shares in CRSP; multiply by 1000 if you want true $ mktcap
    crsp["mktcap"] = crsp["prc"] * crsp["shrout"]

    name_map = firm_list.set_index("permno")["conm"].to_dict()
    ticker_map = firm_list.set_index("permno")["tic"].to_dict()
    crsp["conm"] = crsp["permno"].map(name_map)
    crsp["tic"] = crsp["permno"].map(ticker_map)

    # 5) Fundamentals
    gvkeys = firm_list["gvkey"].tolist()
    gvkey_sql_list = ",".join(f"'{x}'" for x in gvkeys)

    sql_fund = f"""
        SELECT gvkey, datadate, atq, ltq, ceqq, saleq, niq, mkvaltq, prccq, cshtrq
        FROM comp.fundq
        WHERE gvkey IN ({gvkey_sql_list})
        ORDER BY gvkey, datadate;
    """
    fundq = db.raw_sql(sql_fund)
    fundq["datadate"] = pd.to_datetime(fundq["datadate"])

    fundq = fundq.merge(firm_list[["gvkey", "permno"]], on="gvkey", how="left")
    crsp["permno"] = crsp["permno"].astype("int64")
    fundq["permno"] = fundq["permno"].astype("int64")

    merged = pd.merge_asof(
        crsp.sort_values("date"),
        fundq.sort_values("datadate"),
        by="permno",
        left_on="date",
        right_on="datadate",
        direction="backward",
    )

    merged = merged.drop_duplicates(subset=["permno", "date"]).reset_index(drop=True)

    # 6) Valuation metrics and bubble_index
    merged["ps"] = merged["mktcap"] / merged["saleq"]
    merged["pb"] = merged["mktcap"] / merged["ceqq"]
    merged["pe"] = merged["mktcap"] / merged["niq"]
    merged["ev"] = merged["mktcap"] + merged["ltq"] - merged["ceqq"]
    merged["ev_sales"] = merged["ev"] / merged["saleq"]

    metrics = ["ps", "pb", "ev_sales", "pe"]
    for m in metrics:
        merged[m + "_z"] = merged.groupby("permno")[m].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    merged["bubble_index"] = merged[[m + "_z" for m in metrics]].mean(axis=1)
    merged["bubble_type"] = bubble_type

    return merged


# ------------------------------
# Batch generator for all bubbles
# ------------------------------

def generate_all_bubble_csvs(
    config_path: str = "bubble_config.json",
    output_dir: str = "data",
    start_date: str = "2015-01-01",
) -> None:
    """
    Generate and save merged bubble datasets (CSV) for all bubble types
    defined in the JSON config file.

    For each bubble_type in the config, this function:
        1. Builds the merged CRSP+Compustat panel via build_bubble_panel().
        2. Saves the result as <bubble_type>_merged.csv in output_dir.
    """
    config = load_bubble_config(config_path)
    db = wrds.Connection()

    os.makedirs(output_dir, exist_ok=True)

    for bubble_type, cfg in config.items():
        keywords = cfg.get("keywords", [])
        tickers = cfg.get("tickers", [])
        names = cfg.get("names", [])

        # Skip bubbles that are not yet configured
        if not keywords or not tickers or not names:
            print(f"[SKIP] Bubble '{bubble_type}' has empty keywords/tickers/names.")
            continue

        print(f"\n[INFO] Building panel for bubble: {bubble_type}")

        try:
            merged = build_bubble_panel(
                bubble_type=bubble_type,
                db=db,
                config=config,
                start_date=start_date,
            )
            out_path = os.path.join(output_dir, f"{bubble_type}_merged.csv")
            merged.to_csv(out_path, index=False)
            print(f"[OK] Saved {bubble_type} panel to {out_path}")

        except AssertionError as e:
            print(f"[WARN] Skipping bubble '{bubble_type}': {e}")

        except Exception as e:
            print(f"[ERROR] Unexpected error for '{bubble_type}': {e}")


if __name__ == "__main__":
    generate_all_bubble_csvs(
        config_path="bubble_config.json",
        output_dir="data",
        start_date="1995-01-01",
    )
    print("\n[ALL DONE] Bubble datasets generated.")