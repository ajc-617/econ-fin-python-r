import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import anthropic

def main():

    headers = {
        'User-Agent': 'WDI Wikipedia data fetcher bot'
    }

    #URL to economist democracy index wikipedia page
    economist_url = "https://en.wikipedia.org/wiki/The_Economist_Democracy_Index"

    response = requests.get(economist_url, headers=headers)
    #lxml is better here
    soup = BeautifulSoup(response.text, 'lxml')

    #Dropping regime type to avoid duplicate column in final dataframe
    list_by_country = extract_list_by_country(soup).drop(columns=["Regime type"])
    components = extract_components(soup)

    #opting to drop rank: with 2024 rank if two columns have same 2024 score, they'll be same rank. For the rank column one will be above the other. Difference is likely negligible
    #although it should be noted there are a few pairs of ties in the dataset. Prefer keeping 2024 rank to indicate both countries have same score
    all_country_info_df = pd.merge(left=list_by_country, right=components, on='Country', how='inner').drop(columns=["2024 rank"])
    #Printing all info for Spain as a demo
    print(all_country_info_df.loc[all_country_info_df["Country"] == "Spain"])

    #TODO store in postgreSQL

    csv_string = all_country_info_df.to_csv(index=False)
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,    
        messages=[
            {
                "role": "user",
                "content": f"Hi Claude. I am currently looking at the World Democracy Index data on Wikipedia. I'm interested in if there are any interesting observations in said data. I have compiled \
                the data into CSV format using a python beautiful soup parser. Each row has a country, the prior scores (scale of 1 to 10) of said country, the region, and the 5 subcomponent scores that make up \
                the scoring. Here is my CSV data:\n\n{csv_string}\n\nLet me know if you find any trends or other significant findings"
            }
        ]
    )

    print(response.content[0].text)


def get_table_after_heading(soup, heading_text):
    # Find the h2/h3 tag whose text contains the heading we're looking for
    heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and heading_text.lower() in tag.get_text(strip=True).lower())
    # Return the first <table> that appears after that heading in the HTML
    return heading.find_next('table')

def extract_list_by_country(soup):
    """
    Used to extract the List By Country heading of the Wikipedia page into a pandas dataframe. Fetches table immediately after List by country
    heading, casts table to StringIO file-like object, reads into pandas df, and recovers rows that are messed up from the read_html because the HTML isn't
    perfectly formatted 
    """
    table = get_table_after_heading(soup, 'List by country')
    # Parse the table HTML into a DataFrame; wrapping in StringIO is required by pandas (file like object)
    #Index at 0 because this contains the good dataframe. Maybe come back later to why the read_html returns 168 dataframes but last 167 are garbage
    df = pd.read_html(StringIO(str(table)))[0]
    cols = df.columns.tolist()

    #First drop ghost rows (the flag+country-name duplicate rows from nested tables),
    # which have a country name in '2024 rank' but NaN for all score columns and region, regime type, AND country. I know that Regime type gets rid of all rows we're not keeping though
    #Now df will have 167 rows, which is good as we have 167 countries in data
    df = df[df['Regime type'].notna()]

    # Now identifying misaligned columns: Because misaligned columns are shifted right, 2006 (furthest right) column is NaN in these rows. Not explicitly putting 2006 so I don't have to update every year
    misaligned = df.index[df.iloc[:, -1].isna()]
    # Convert to object dtype first so we can freely assign strings into float columns during the shift. Otherwise trying to assign object (like string) to float column will raise error
    df = df.astype(object)
    #Snapshot all original values before modifying anything, then assign shifted in one step. Region column for misaligned rows will now be empty but fixed below
    #TODO Why is .values needed?
    original_values = df.loc[misaligned, cols[:-1]].values
    df.loc[misaligned, cols[1:]] = original_values
    df.loc[misaligned, cols[0]] = None
    # Re-infer column types now that the shift is done
    df = df.infer_objects()

    #Need to explicitly cast to int because infer_objects can only cast to float
    df['2024 rank'] = df['2024 rank'].astype(int)
    # Forward-fill Region: propagates each region name down until the next region starts,
    # recovering the NaN regions caused by the rowspan expiring mid-group
    df['Region'] = df['Region'].ffill()
    return df.reset_index(drop=True)

#TODO fix up comments
def extract_components(soup):
    """
    Used to extract Components heading of the Wikipedia page into a pandas dataframe. Fetches table immediately after Components
    heading, casts table to StringIO file-like object, reads into pandas df, 
    """
    table = get_table_after_heading(soup, 'Components')
    # header=[0] tells pandas to use only the first row as column names, ignoring
    # the extra "Full democracies / Flawed democracies" section rows pandas would otherwise treat as a second header level
    df = pd.read_html(StringIO(str(table)), header=[0])[0]
    # Remove soft-hyphen characters (\xad) that Wikipedia uses for word-breaking in long column names like "Electoral process and pluralism"
    df.columns = [col.replace('\xad', '') for col in df.columns]
    # Same duplicate-row cleanup as above: coerce non-numeric ranks to NaN
    #Need to do this to get rid of flag lines AND lines which contain a regime type repeated (like "Full democracies,Full democracies,Full democracies,.....")
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df = df.dropna(subset=['Rank'])
    #Casting back to int from float
    df['Rank'] = df['Rank'].astype(int)
    #Dropping these two columns for now because wikipedia page has red arrow for decrease and green arrow for decrease, skipping for now
    #TODO implement increase and decrease in score and rank
    df = df.drop(['Δ Rank', 'Δ Score'], axis=1)
    #For some reason on the wikipedia page Gambia is "Gambia" in list by country but "The Gambia" in components so renaming that here
    df['Country'] = df['Country'].replace({"The Gambia": "Gambia"})
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()