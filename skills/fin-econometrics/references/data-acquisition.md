# Data acquisition

## A. User-provided files
`$EK profile --data FILE` supports csv/tsv/xlsx/xls/dta/parquet/json. Ask for a data dictionary
if variable names are cryptic. Record the units and the frequency.

## B. Public sources (one-shot)
| Source | Command | Notes |
|---|---|---|
| FRED (US macro/finance; no key) | `$EK fetch fred --series GDPC1 UNRATE FEDFUNDS --start 1990-01-01 --out fred.csv` | Mixed frequencies → resample before merging |
| World Bank WDI (country panels) | `$EK fetch worldbank --series NY.GDP.PCAP.KD FS.AST.PRVT.GD.ZS --countries "CHN;USA;IND" --start 2000 --end 2023 --out wb.csv` | Long panel: entity=`iso3`, time=`year` |
| Any CSV/XLSX/DTA URL | `$EK fetch url --url https://... --out raw.csv` | e.g. Ken French data library, Our World in Data, statistical bureau downloads |
| HTML table | `$EK fetch table --url https://... --match "keyword" --index 0 --out t.csv` | Static pages only |

Optional Python libraries, installed on demand (mention them in caveats if used): `akshare` for
China A-shares, macro data and funds (no key); `yfinance` for equity prices; `pandas-datareader`.
When a source needs an API key (Tushare, Wind, CSMAR), ask the user for it. Never hard-code keys.

Rules: cite the URL and the access date in `analysis.json.data.source`; save the raw download
before transforming it; check the series' units, seasonal adjustment, and real vs nominal
values.

## C. Persistent / scheduled crawling
A skill runs only while the agent runs. For data that must keep updating, generate a small
scheduled job and hand it to the user. Do not try to keep a process alive inside the
session.
- **GitHub Actions** (default, free): copy `assets/scheduled-fetch.yml` into the user's
  `.github/workflows/`, and put the fetch commands in `fetch.sh`. The job appends to
  `data/*.csv` and commits the result.
- Alternatives: cron on a server, Cloudflare Workers Cron, or an Airflow/Prefect pipeline.

Crawling etiquette: respect robots.txt and the site's terms of service. Rate-limit requests
(≥ 1 s apart), identify yourself with a User-Agent, prefer official APIs over scraping, and
never collect personal data.
