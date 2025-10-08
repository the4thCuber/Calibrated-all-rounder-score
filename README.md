# Calibrated-all-rounder-score
A calculator to compute the best all-round speedcubers.

## What is a Calibrated All-rounder Score?
There are many ways to compute the best "all-round" cuber. Sum of ranks and kinch are two of the best so far. However, both have their drawbacks. Sum of ranks weights events based on the number of competitors with results, so for example not having a 5BLD success does not hurt your score very much, while kinch gives exactly equal weighting to each event, which unfairly benefits NxN and BLD solvers who excel in multiple closerly correlated events. CAS is similar to kinch, but it calibrates the score based on how many competitors have successful results (the less that do, the greater the reward for having at least a result in that event), and the correlatedness of that event (if results in that event are easy to predict given PRs in all other events, then the maximum score for that event is lowered). It's admittedly much more difficult to calculate than sum of ranks or kinch, but far more fair.

# Runtime instructions
This will leave you with a csv of all official PRs, the CAS computation table csv, and a csv of all WCA IDs with associated CAS score (oh, and the entire WCA database that you might want to delete)

## Prerequisites
This program uses pandas, json, numpy, and sklearn. If these are not installed on your system, you will need to install them.
```bash
pip install pandas
pip install json
pip install numpy
pip install sklearn
```
(It's worth noting that I'm on Arch Linux, so I'd install them systemwide using 'sudo pacman -S python-{packagename}'. I don't know how installing packages using pip works on other platforms.)
## Create CAS table
Run scrape_results.py to download the WCA database and create the config table in CAS_results.csv
- Leave the custom arguments alone unless you know what you're doing (and you probably don't, because I haven't posted the arguments yet)
- You don't need to generate a list of WCA IDs either
- So basically just use the defaults
## Get PRs
Run get_all_prs.py to process the raw WCA results into a dict of all PRs for all competitors, stored in all_prs.json
## Compute CAS scores
Run sort_by_cas.py to generate a timestamped csv file of the CAS scores of all competitors in all_prs.json
## Done
Now you have a csv to do whatever you want with (probably open in your favorite spreadsheet app and do lots of mathy things with it)
You're welcome! Message me with any questions you have (idk how messaging works on GitHub tho) or open an issue.
Feel free to fork this repo, please let me know if you do so I can check out what you make.
