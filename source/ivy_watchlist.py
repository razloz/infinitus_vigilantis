__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'

#   S&P500 + NASDAQ500
composite_index = ('SPY', 'QQQ',
    'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABNB', 'ABT', 'ACGL', 'ACN',
    'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG',
    'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR',
    'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS',
    'APA', 'APD', 'APH', 'APTV', 'ARE', 'ASML', 'ATO', 'ATVI', 'AVB', 'AVGO',
    'AVY', 'AWK', 'AXON', 'AXP', 'AZN', 'AZO', 'BA', 'BAC', 'BALL', 'BAX',
    'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG',
    'BKR', 'BLK', 'BMY', 'BR', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG',
    'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS',
    'CDW', 'CE', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL',
    'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
    'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO',
    'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR',
    'D', 'DAL', 'DD', 'DDOG', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
    'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
    'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV',
    'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS',
    'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F',
    'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FICO', 'FIS', 'FISV',
    'FITB', 'FLT', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD',
    'GE', 'GEHC', 'GEN', 'GFS', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC',
    'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN',
    'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL',
    'HSIC', 'HST', 'HSY', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF',
    'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM',
    'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JCI', 'JD', 'JKHY', 'JNJ', 'JNPR',
    'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX',
    'KO', 'KR', 'L', 'LCID', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY',
    'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB',
    'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ',
    'MDT', 'MELI', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC',
    'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO',
    'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH',
    'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG',
    'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'NXPI',
    'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY',
    'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PDD', 'PEAK', 'PEG', 'PEP',
    'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC',
    'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC',
    'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RF',
    'RHI', 'RIVN', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
    'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEDG', 'SEE', 'SGEN', 'SHW', 'SIRI', 'SJM',
    'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
    'TEAM', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS',
    'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO',
    'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS',
    'URI', 'USB', 'V', 'VFC', 'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX',
    'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC',
    'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW',
    'WY', 'WYNN', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION',
    'ZM', 'ZS', 'ZTS',
    )


#  small cap+ $10-$50
small_cap = (
    'AA', 'AAL', 'ABR', 'ACAD', 'ACIW', 'ACMR', 'ADMA', 'ADNT', 'AEHR', 'AEO',
    'AES', 'AGIO', 'AGR', 'AGS', 'AKR', 'AKRO', 'AL', 'ALK', 'ALLY', 'AM',
    'AMH', 'AMKR', 'AMRC', 'AMRK', 'AMSC', 'APA', 'APG', 'APLE', 'APLS',
    'APPN', 'AR', 'ARCC', 'ARCT', 'ARLO', 'ARMK', 'AROC', 'ARR', 'ARVN',
    'ARWR', 'ASB', 'ASPN', 'ATEC', 'ATEN', 'ATRC', 'AVTR', 'AXTA', 'BAC',
    'BANC', 'BAX', 'BBWI', 'BE', 'BEN', 'BFH', 'BHVN', 'BKR', 'BKU',
    'BL', 'BLMN', 'BMY', 'BOX', 'BRX', 'BTU', 'BV', 'BWA', 'BXMT', 'BYON',
    'CADE', 'CAG', 'CAKE', 'CAL', 'CALX', 'CARG', 'CBRL', 'CC', 'CCL', 'CDNA',
    'CDP', 'CENX', 'CFG', 'CG', 'CGNX', 'CHWY', 'CHX', 'CIEN', 'CIM', 'CLDX',
    'CLF', 'CLSK', 'CMA', 'CMCSA', 'CMP', 'CNK', 'CNNE', 'CNO', 'CNP', 'CNX',
    'COLB', 'COLD', 'CORT', 'COTY', 'CPB', 'CPRX', 'CRK', 'CRNX', 'CSCO', 'CSX',
    'CTRA', 'CTRE', 'CUBE', 'CUZ', 'CVBF', 'CVI', 'CWEN', 'CWH', 'CXW', 'CZR',
    'D', 'DAL', 'DAN', 'DAR', 'DAY', 'DBRG', 'DBX', 'DEA', 'DEI', 'DK', 'DNLI',
    'DNOW', 'DOC', 'DRS', 'DVAX', 'DVN', 'DX', 'DXC', 'EFC', 'ELAN', 'ELME',
    'ENLC', 'ENOV', 'EOLS', 'EPD', 'EPR', 'EPRT', 'EQC', 'EQH', 'EQT', 'ESI',
    'ET', 'ETRN', 'EVBG', 'EVH', 'EXAS', 'EXC', 'EXEL', 'EXLS', 'EXPI', 'EXTR',
    'EYE', 'EZPW', 'F', 'FBP', 'FCX', 'FE', 'FHB', 'FHI', 'FHN', 'FIBK', 'FITB',
    'FIVN', 'FL', 'FLEX', 'FLO', 'FLR', 'FLS', 'FNB', 'FNF', 'FOX', 'FOXA',
    'FOXF', 'FR', 'FSK', 'FTDR', 'FTI', 'FULT', 'FWRD', 'GBCI', 'GBDC', 'GEN',
    'GEO', 'GH', 'GIII', 'GLPI', 'GLW', 'GM', 'GME', 'GNK', 'GNTX', 'GO', 'GPK',
    'GPRE', 'GPS', 'GRPN', 'GSBD', 'GT', 'GTES', 'HA', 'HAL', 'HASI', 'HBAN',
    'HESM', 'HGV', 'HIW', 'HLF', 'HLIT', 'HLX', 'HOG', 'HOMB', 'HOPE', 'HP',
    'HPE', 'HPQ', 'HR', 'HRL', 'HST', 'HTGC', 'HUN', 'HUT', 'IAC', 'IART',
    'IDYA', 'IEP', 'IMVT', 'INOD', 'INTC', 'INVA', 'INVH', 'IONS', 'IP', 'IPG',
    'IRDM', 'IRT', 'IVZ', 'JBGS', 'JEF', 'JELD', 'JNPR', 'JWN', 'KAR', 'KDP',
    'KEY', 'KHC', 'KIM', 'KMI', 'KMT', 'KN', 'KNX', 'KR', 'KRC', 'KRG', 'KSS',
    'KTOS', 'KURA', 'LAUR', 'LAZ', 'LBRT', 'LEG', 'LEVI', 'LITE', 'LKQ', 'LNC',
    'LPG', 'LQDA', 'LSXMA', 'LSXMK', 'LUV', 'LVS', 'LYFT', 'M', 'MAC', 'MARA',
    'MAT', 'MCS', 'MDU', 'METC', 'MFA', 'MGM', 'MGNI', 'MGY', 'MITK', 'MLKN',
    'MNRO', 'MO', 'MODG', 'MODN', 'MOS', 'MPLX', 'MRC', 'MRO', 'MTCH', 'MTG',
    'MUR', 'MWA', 'MXL', 'MYGN', 'NAVI', 'NCLH', 'NEM', 'NEO', 'NEOG', 'NEP',
    'NFE', 'NI', 'NLY', 'NMIH', 'NMRK', 'NNN', 'NOG', 'NOV', 'NSA', 'NTCT',
    'NTLA', 'NUS', 'NVAX', 'NWBI', 'NWS', 'NWSA', 'OCSL', 'OGE', 'OHI', 'OI',
    'OII', 'OLN', 'OMF', 'OMI', 'ONB', 'OPCH', 'ORI', 'OUT', 'OVV', 'OZK',
    'PAA', 'PAGP', 'PARA', 'PARR', 'PBF', 'PCG', 'PCRX', 'PD', 'PDCO', 'PEB',
    'PENN', 'PFE', 'PFLT', 'PFS', 'PINC', 'PINS', 'PK', 'PLAY', 'PMT', 'PNM',
    'POR', 'PPBI', 'PPC', 'PPL', 'PR', 'PRDO', 'PRMW', 'PTCT', 'PTEN', 'PTGX',
    'PZZA', 'QDEL', 'RAMP', 'RARE', 'RCKT', 'RCM', 'RCUS', 'RDN', 'REVG',
    'REXR', 'REZI', 'RF', 'RGNX', 'RGP', 'RILY', 'RITM', 'RNG', 'ROIC', 'ROL',
    'RPAY', 'RPD', 'RRC', 'RUN', 'RVLV', 'RYTM', 'SAGE', 'SATS', 'SAVA', 'SBH',
    'SBOW', 'SBRA', 'SCS', 'SEE', 'SEM', 'SGH', 'SGRY', 'SHO', 'SHOO', 'SILK',
    'SITC', 'SIX', 'SKT', 'SLB', 'SLCA', 'SLGN', 'SLM', 'SM', 'SMAR', 'SMPL',
    'SMTC', 'SNAP', 'SNDR', 'SNDX', 'SNV', 'SONO', 'SPR', 'ST', 'STAA', 'STAG',
    'STOK', 'STR', 'STWD', 'SUM', 'SUPN', 'SYF', 'T', 'TALO', 'TBBK', 'TDC',
    'TDOC', 'TDS', 'TENB', 'TFC', 'TGI', 'TGNA', 'TGTX', 'TNDM', 'TNL', 'TPC',
    'TPH', 'TPR', 'TPX', 'TRIP', 'TRN', 'TROX', 'TRUP', 'TTMI', 'TWO', 'TWST',
    'UAL', 'UCBI', 'UDR', 'UE', 'UGI', 'UNFI', 'UPWK', 'URBN', 'URGN', 'USB',
    'UTI', 'UTZ', 'VCYT', 'VECO', 'VFC', 'VGR', 'VICI', 'VIRT', 'VNO', 'VNOM',
    'VRDN', 'VRE', 'VRNS', 'VRNT', 'VRRM', 'VSAT', 'VSH', 'VSTO', 'VTLE',
    'VTRS', 'VVV', 'VYX', 'VZ', 'WBA', 'WBS', 'WEN', 'WERN', 'WES', 'WKC',
    'WMB', 'WNC', 'WOLF', 'WRK', 'WSC', 'WTRG', 'WTTR', 'WU', 'WWW', 'WY', 'X',
    'XHR', 'XNCR', 'XPRO', 'XRAY', 'XRX', 'YELP', 'YETI', 'Z', 'ZG', 'ZION',
    'ZWS',
    )

ivy_watchlist = sorted(set(composite_index + small_cap))

