import pandas as pd
from pathlib import Path


def del_duplicates(filepath_1, filepath_2, col='Title'):
    """deletes duplicates in two csv files. Assumes same columns in both files
    as well as a title column where equality test is performed on."""

    file_1 = Path(filepath_1)
    file_2 = Path(filepath_2)

    df_1 = pd.read_csv(file_1)
    df_2 = pd.read_csv(file_2)

    df = pd.concat([df_1, df_2], ignore_index=True)

    df = df.drop_duplicates(subset=[col], ignore_index=True)

    return df


def del_little_citations(dataframe, citation_threshold=7, citation_per_year=False):
    """Drop entries in the dataframe that have little citations. Default is papers with
    <7 citations will be dropped. If citation_per_year is False, the deletion will
    consider the absolut number of citations, if it is True, citations per year will be considered.
    Default is False."""

    if citation_per_year:
        col = 'cit/year'
    else:
        col = 'Citations'

    df = dataframe[dataframe[col] > citation_threshold]

    return df


def del_by_keyword(dataframe, keywords, and_=True, included=True):
    """Drops entries in the dataframe by given list keywords. By default, all keywords have to appear in
    the title to keep the entry, if and_=False, it is an or operation. If included is True, the words have
    to appear in the title for an entry to be kept, if it is False, the keywords are not allowed to be
    in the title."""
    df_copy = dataframe.copy()
    keywords = [keyword.lower() for keyword in keywords]

    for index, row, in dataframe.iterrows():
        title = row['Title'].lower()

        if included:
            if and_ and all(keyword in title for keyword in keywords):
                continue
            elif not and_ and any(keyword in title for keyword in keywords):
                continue
            else:
                df_copy.drop([index], inplace=True)
        else:
            if and_ and all(keyword not in title for keyword in keywords):
                continue
            elif not and_ and any(keyword not in title for keyword in keywords):
                continue
            else:
                df_copy.drop([index], inplace=True)

    return df_copy


if __name__ == "__main__":
    file_1 = "'emotion'_+_'classification'_+_'EEG'_+_'transformer'_original.csv"
    file_2 = "'emotion'_+_'recognition'_+_'EEG'_+_'transformer'_original.csv"

    df_no_duplicates = del_duplicates(file_1, file_2)
    df_no_duplicates.to_csv('papers_no_duplicates.csv')

    df_no_little_citations = del_little_citations(df_no_duplicates)
    df_no_little_citations.to_csv('papers_citations>7.csv')

    df_transformer = del_by_keyword(df_no_little_citations, ['transformer', 'bert'], and_=False)
    df_transformer.to_csv('papers_transformer.csv')
