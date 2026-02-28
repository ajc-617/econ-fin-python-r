import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

def main():

    headers = {
        'User-Agent': 'WDI Wikipedia data fetcher bot'
    }

    economist_url = "https://en.wikipedia.org/wiki/The_Economist_Democracy_Index"

    response = requests.get(economist_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')

    #Dropping regime type to avoid duplicate column in final dataframe
    list_by_country = extract_list_by_country(soup).drop(columns=["Regime type"])
    components = extract_components(soup)

    print(list_by_country.columns)
    print(components.columns)

    #opting to drop rank: with 2024 rank if two columns have same 2024 score, they'll be same rank. For the rank column one will be above the other. Difference is likely negligible
    #although it should be noted there are a few pairs of ties in the dataset. Prefer keeping 2024 rank to indicate both countries have same score
    all_country_info_df = pd.merge(left=list_by_country, right=components, on='Country', how='inner').drop(columns=["2024 rank"])
    #Printing all info for Spain as a demo
    print(all_country_info_df.loc[all_country_info_df["Country"] == "Spain"])




def get_table_after_heading(soup, heading_text):
    # Find the h2/h3 tag whose text contains the heading we're looking for
    heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and heading_text.lower() in tag.get_text(strip=True).lower())
    # Return the first <table> that appears after that heading in the HTML
    return heading.find_next('table')


#TODO fix up comments
def extract_list_by_country(soup):
    table = get_table_after_heading(soup, 'List by country')
    # Parse the table HTML into a DataFrame; wrapping in StringIO is required by pandas (file like object)
    df = pd.read_html(StringIO(str(table)))[0]
    cols = df.columns.tolist()

    # Some rows are misaligned: when the Region cell is absent (its rowspan expired),
    # pandas shifts every value one column to the left — so the rank ends up in Region, the country name ends up in '2024 rank', etc.
    # We detect these rows by checking whether '2024 rank' contains a non-numeric value, while '2024' (the score column) also has a value, meaning it's a real row with shifted data.
    misaligned = pd.to_numeric(df['2024 rank'], errors='coerce').isna() & df['2024 rank'].notna() & df['2024'].notna()

    # Fix misaligned rows by shifting each column's value one position to the right, and setting Region to NaN since those rows have no region cell in the HTML.
    # Convert to object dtype first so we can freely assign strings into float columns during the shift.
    df = df.astype(object)
    #Snapshot all original values before modifying anything, then assign shifted in one step.
    #Region column for misaligned rows will now be empty but fixed below
    #TODO Why is .values needed?
    original_values = df.loc[misaligned, cols[:-1]].values
    df.loc[misaligned, cols[1:]] = original_values
    df.loc[misaligned, cols[0]] = None
    # Re-infer column types now that the shift is done
    df = df.infer_objects()

    # Drop ghost rows (the flag+country-name duplicate rows from nested tables),
    # which have a country name in '2024 rank' but NaN for all score columns
    df['2024 rank'] = pd.to_numeric(df['2024 rank'], errors='coerce')
    df = df.dropna(subset=['2024 rank'])
    df['2024 rank'] = df['2024 rank'].astype(int)
    # Forward-fill Region: propagates each region name down until the next region starts,
    # recovering the NaN regions caused by the rowspan expiring mid-group
    df['Region'] = df['Region'].ffill()
    return df.reset_index(drop=True)

#TODO fix up comments
def extract_components(soup):
    table = get_table_after_heading(soup, 'Components')
    # header=[0] tells pandas to use only the first row as column names, ignoring
    # the extra "Full democracies / Flawed democracies" section rows pandas would otherwise treat as a second header level
    df = pd.read_html(StringIO(str(table)), header=[0])[0]
    # Remove soft-hyphen characters (\xad) that Wikipedia uses for word-breaking
    # in long column names like "Electoral process and pluralism"
    df.columns = [col.replace('\xad', '') for col in df.columns]
    # Same duplicate-row cleanup as above: coerce non-numeric ranks to NaN
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank'])
    df['Rank'] = df['Rank'].astype(int)
    #For some reason on the wikipedia page Gambia is "Gambia" in list by country but "The Gambia" in components so renaming that here
    df['Country'] = df['Country'].replace({"The Gambia": "Gambia"})
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()