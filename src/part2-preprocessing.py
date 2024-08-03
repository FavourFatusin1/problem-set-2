'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages



# Your code here
import pandas as pd

# Load the data
pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.csv?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.csv?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

# Print column names to verify
print("Column names in pred_universe_raw:")
print(pred_universe_raw.columns)
print("Column names in arrest_events_raw:")
print(arrest_events_raw.columns)

# Convert date columns to datetime
try:
    pred_universe_raw['filing_date'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['filing_date'] = pd.to_datetime(arrest_events_raw['filing_date'])
except KeyError as e:
    print(f"KeyError: {e}. Check if the column names match.")
    raise

# Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer', suffixes=('_univ', '_event'))

# Function to check if the person was rearrested for a felony in the next year
def was_rearrested_for_felony(row, arrest_events):
    arrest_date = row['filing_date_univ']
    person_id = row['person_id']
    one_year_later = arrest_date + pd.DateOffset(days=365)
    return arrest_events[(arrest_events['person_id'] == person_id) &
                         (arrest_events['filing_date'] > arrest_date) &
                         (arrest_events['filing_date'] <= one_year_later) &
                         (arrest_events['charge_degree'] == 'felony')].shape[0] > 0

# Create the 'y' column
df_arrests['y'] = df_arrests.apply(lambda row: 1 if was_rearrested_for_felony(row, arrest_events_raw) else 0, axis=1)

# Print the share of arrestees rearrested for a felony crime in the next year
share_rearrested_felony = df_arrests['y'].mean()
print(f"What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year? {share_rearrested_felony:.2%}")

# Create a predictive feature `current_charge_felony`
df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)

# Print the share of current charges that are felonies
share_current_felony = df_arrests['current_charge_felony'].mean()
print(f"What share of current charges are felonies? {share_current_felony:.2%}")

# Function to calculate the number of felony arrests in the past year
def num_fel_arrests_last_year(row, arrest_events):
    arrest_date = row['filing_date_univ']
    person_id = row['person_id']
    one_year_before = arrest_date - pd.DateOffset(days=365)
    return arrest_events[(arrest_events['person_id'] == person_id) &
                         (arrest_events['filing_date'] >= one_year_before) &
                         (arrest_events['filing_date'] < arrest_date) &
                         (arrest_events['charge_degree'] == 'felony')].shape[0]

df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(lambda row: num_fel_arrests_last_year(row, arrest_events_raw), axis=1)

# Print the average number of felony arrests in the last year
average_felony_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
print(f"What is the average number of felony arrests in the last year? {average_felony_arrests_last_year:.2f}")

# Print the mean of 'num_fel_arrests_last_year'
print(f"Mean of num_fel_arrests_last_year: {df_arrests['num_fel_arrests_last_year'].mean()}")

# Print pred_universe_raw.head()
print(pred_universe_raw.head())

# Save `df_arrests` for use in main.py for PART 3
df_arrests.to_csv('data/df_arrests.csv', index=False)
