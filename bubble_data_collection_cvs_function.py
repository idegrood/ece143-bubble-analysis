import wrds
import pandas as pd


def collect_enterprise_valuation_csv(bubble_name, ev_tickers, ev_names, ev_keywords):
    '''
    YOU WILL NEED TO SIGN INTO YOUR WRDS ACCOUNT!!!!! 

    Please see the provided CSV files if you do not have a WRDS account

    Generates CSV file and RETURNS PandaDataframe
    
    param: bubble_name: string of the name of the bubble analyising
    param: ev_tickers: list of strings of tickers to determine enterprise valuation 
    param: ev_names: list of strings of company names to determine enterprise valuation
    param: ev_keywords: list of strings for keywords 

    returns: PandaDataframe of ???? 
    
    '''
    db = wrds.Connection()

    pattern = '|'.join(ev_keywords)

    sql_ev = f"""
        SELECT DISTINCT gvkey, conm, tic
        FROM comp.fundq
        WHERE conm ~* '{pattern}'
        ORDER BY conm;
    """

    ev_list = db.raw_sql(sql_ev)
    print("EV firms identified:")
    ev_list.head()


    # Check results
    link = db.get_table(
        library='crsp',
        table='ccmxpf_linktable',
        columns=['gvkey','lpermno','linktype','linkprim']
    )

    # keep only valid primary links
    link = link[
        (link['linktype'].isin(['LC','LU'])) &
        (link['linkprim'].isin(['P','C']))
    ]

    link = link[['gvkey','lpermno']].drop_duplicates().rename(columns={'lpermno':'permno'})

    ev_list = ev_list.merge(link, on='gvkey', how='left')
    ev_list = ev_list.dropna(subset=['permno'])
    ev_list['permno'] = ev_list['permno'].astype(int)
    print("\nLinked EV firms:")
    ev_list.head()
    link.rename(columns={'lpermno':'permno'}, inplace=True)

    ev_list = ev_list[
        (ev_list['tic'].isin(ev_tickers)) |
        (ev_list['conm'].str.upper().isin(ev_names))
    ].drop_duplicates(subset=['permno'])

    ev_list.head()

    permnos = ev_list['permno'].tolist()
    permnos_str = ','.join(map(str, permnos))

    sql_crsp = f"""
        SELECT permno, date, ret, retx, prc, vol, shrout
        FROM crsp.dsf
        WHERE permno IN ({permnos_str})
        AND date >= '2015-01-01'
        ORDER BY permno, date;
    """

    crsp = db.raw_sql(sql_crsp)

    # market cap
    crsp['prc'] = crsp['prc'].abs()
    crsp['mktcap'] = crsp['prc'] * crsp['shrout']

    name_map = ev_list.set_index('permno')['conm'].to_dict()
    ticker_map = ev_list.set_index('permno')['tic'].to_dict()
    crsp['conm'] = crsp['permno'].map(name_map)
    crsp['tic'] = crsp['permno'].map(ticker_map)

    gvkeys = ev_list['gvkey'].tolist()

    gvkey_sql_list = ",".join(f"'{x}'" for x in gvkeys)

    # Query fundamentals
    sql_fund = f"""
        SELECT gvkey, datadate, atq, ltq, ceqq, saleq, niq, mkvaltq, prccq, cshtrq
        FROM comp.fundq
        WHERE gvkey IN ({gvkey_sql_list})
        ORDER BY gvkey, datadate;
    """

    fundq = db.raw_sql(sql_fund)

    fundq = fundq.merge(ev_list[['gvkey','permno']], on='gvkey', how='left')

    # Merge on permno + nearest quarter <= date
    fundq['datadate'] = pd.to_datetime(fundq['datadate'])
    crsp['date'] = pd.to_datetime(crsp['date'])

    crsp['permno'] = crsp['permno'].astype('int64')
    fundq['permno'] = fundq['permno'].astype('int64')

    merged = pd.merge_asof(
        crsp.sort_values('date'),
        fundq.sort_values('datadate'),
        by='permno',
        left_on='date',
        right_on='datadate',
        direction='backward'
    )

    merged = merged.drop_duplicates(subset=['permno','date']).reset_index(drop=True)

    print("\nMerged EV dataset preview:")
    merged.head()

    merged["ps"] = merged["mktcap"] / merged["saleq"]          # Price to Sales
    merged["pb"] = merged["mktcap"] / merged["ceqq"]           # Price to Book
    merged["pe"] = merged["mktcap"] / merged["niq"]            # Price to Earnings

    # Enterprise value
    merged["ev"] = merged["mktcap"] + merged["ltq"] - merged["ceqq"]
    merged["ev_sales"] = merged["ev"] / merged["saleq"]

    # Profitability
    merged["profit_margin"] = merged["niq"] / merged["saleq"]
    merged["roe"] = merged["niq"] / merged["ceqq"]
    merged["roa"] = merged["niq"] / merged["atq"]

    # Leverage
    merged["debt_asset"] = merged["ltq"] / merged["atq"]
    merged["de_ratio"] = merged["ltq"] / merged["ceqq"]

    metrics = ["ps", "pb", "ev_sales", "pe"]

    for m in metrics:
        merged[m+"_z"] = merged.groupby("permno")[m].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    merged["bubble_index"] = merged[["ps_z","pb_z","ev_sales_z","pe_z"]].mean(axis=1)
    

    output_file = f"{bubble_name}_bubble_merged.csv"
    merged.to_csv(output_file, index=False)

    print(f"Merged EV dataset saved to {output_file}")

    return merged

#Testing with example electric veichle bubble
tickers = ['TSLA', 'NIO', 'BYD', 'LI', 'XPEV', 'LCID'] 
names = ['TESLA INC', 'NIO INC', 'BYD COMPANY LTD', 'LI AUTO INC', 'XPENG INC', 'LUCID GROUP INC']
keywords = ['Tesla', 'NIO', 'BYD', 'Li Auto', 'Xpeng', 'Lucid', 'Rivian']

merg = collect_enterprise_valuation_csv("Electric Vehicles", tickers, names, keywords) #returns CSV file