# Data Source: common_player_info

This table contains NBA player biographical and career information, one row per player.

## Data Sources

| Table              | Description                                       |
| ------------------ | ------------------------------------------------- |
| common_player_info | NBA player bio/career info (from CSV, local file) |

## Columns

| Column                           | Type   | Description                     |
| -------------------------------- | ------ | ------------------------------- |
| person_id                        | int    | Unique player identifier        |
| first_name                       | string | Player's first name             |
| last_name                        | string | Player's last name              |
| display_first_last               | string | Full name (First Last)          |
| display_last_comma_first         | string | Full name (Last, First)         |
| display_fi_last                  | string | Abbreviated name (F. Last)      |
| player_slug                      | string | URL slug for player             |
| birthdate                        | date   | Date of birth                   |
| school                           | string | College/University              |
| country                          | string | Country of origin               |
| last_affiliation                 | string | Last team/affiliation           |
| height                           | string | Height (feet-inches)            |
| weight                           | int    | Weight (lbs)                    |
| season_exp                       | float  | Years of NBA experience         |
| jersey                           | string | Jersey number                   |
| position                         | string | Position(s)                     |
| rosterstatus                     | string | Roster status (Active/Inactive) |
| games_played_current_season_flag | string | Played this season? (Y/N)       |
| team_id                          | int    | Team identifier                 |
| team_name                        | string | Team name                       |
| team_abbreviation                | string | Team abbreviation               |
| team_code                        | string | Team code                       |
| team_city                        | string | Team city                       |
| playercode                       | string | Player code                     |
| from_year                        | float  | First NBA season                |
| to_year                          | float  | Last NBA season                 |
| dleague_flag                     | string | D-League experience? (Y/N)      |
| nba_flag                         | string | NBA experience? (Y/N)           |
| games_played_flag                | string | Played games? (Y/N)             |
| draft_year                       | string | Draft year or 'Undrafted'       |
| draft_round                      | string | Draft round or 'Undrafted'      |
| draft_number                     | string | Draft pick or 'Undrafted'       |
| greatest_75_flag                 | string | Named to NBA 75th team? (Y/N)   |