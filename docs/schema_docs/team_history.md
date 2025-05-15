# Data Source: team_history

This table contains historical NBA team franchise data, one row per team identity and era.

## Data Sources

| Table        | Description                                       |
| ------------ | ------------------------------------------------- |
| team_history | NBA team franchise history (from CSV, local file) |

## Columns

| Column           | Type   | Description                             |
| ---------------- | ------ | --------------------------------------- |
| team_id          | int    | Unique team identifier                  |
| city             | string | Team city for this era                  |
| nickname         | string | Team nickname for this era              |
| year_founded     | int    | Year this team identity was founded     |
| year_active_till | int    | Last year this team identity was active |