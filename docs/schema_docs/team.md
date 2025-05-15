# Data Source: team

This table contains NBA team metadata, one row per team franchise.

## Data Sources

| Table | Description                              |
| ----- | ---------------------------------------- |
| team  | NBA team metadata (from CSV, local file) |

## Columns

| Column       | Type   | Description                |
| ------------ | ------ | -------------------------- |
| id           | int    | Unique team identifier     |
| full_name    | string | Full team name             |
| abbreviation | string | Team abbreviation (3-char) |
| nickname     | string | Team nickname              |
| city         | string | Team city                  |
| state        | string | Team state                 |
| year_founded | float  | Year the team was founded  |