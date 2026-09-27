# DemandIQ API

Base prefix: `/api/v1`

Interactive documentation is available at `/docs` when the backend is running.

## Single forecast

`POST /predictions/demand` accepts `store_nbr`, `family`, `date`, plus optional overrides for `onpromotion`, `transactions`, and `dcoilwtico`. The service retrieves the latest matching genuine demo-history row at or before the requested date and supplies the remaining model features.

`POST /predictions/direct` accepts a `features` object containing all 23 model columns.

## Batch forecast

`POST /predictions/batch` accepts a CSV multipart upload. The default limits are 10 MB and 5,000 rows. Each invalid row receives a row-level diagnostic; valid rows are still scored.

## Artifact endpoints

`/model/performance`, `/model/explainability`, `/anomalies`, `/segments`, and `/replenishment` read the original project outputs. Pagination limits protect large responses.
