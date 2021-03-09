import pandas as pd
import os
from shutil import copyfile


# dataset indir and outdir
indir = os.path.join('dataset', 'painter-by-numbers')
outdir_baroque = os.path.join('dataset', 'painter-of-baroque')
outdir_nude = os.path.join('dataset', 'painter-of-nude')

# csv file
infile = os.path.join(indir, 'all_data_info.csv')


def extract_date(df, col):

    # extract 4 digits anywhere: e.g., c.1982, November 1982
    df[col] = df[col].str.extract(r'(\d{4})')

    # convert to numeric
    df[col] = pd.to_numeric(df[col])

    return df


def extract_painters(df):

    df_painters = df[['artist', 'style', 'genre', 'new_filename']]

    # aggregate for 'artist' by the total count of the paintings
    df_painters = df_painters.groupby(['artist']).size().reset_index(name='counts').sort_values(by='counts',ascending=False)

    # print all the rows
    pd.set_option('display.max_rows', df_painters.shape[0] + 1)
    print(df_painters)


def filter_baroque(infile):

    df = pd.read_csv(infile)

    # filter columns
    df_baroque = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    # filter by 'date'
    # df_baroque = extract_date(df_baroque, 'date')

    # is_baroque_start = df_baroque['date'] > 1580
    # is_baroque_end = df_baroque['date'] < 1750

    # df_baroque = df_baroque[is_baroque_start]
    # df_baroque = df_baroque[is_baroque_end]

    # filter by 'style'
    is_baroque = df_baroque['style'] == 'Baroque'
    df_baroque = df_baroque[is_baroque]

    # filter by 'artist' == caravaggio
    # is_caravaggio = df_baroque['artist'] == 'Caravaggio'
    # df_caravaggio = df_baroque[is_caravaggio]

    extract_painters(df_baroque)

    return df_baroque


def filter_nude(infile):

    df = pd.read_csv(infile)

    # filter columns
    df_nude = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    # filter by 'genre'
    is_nude = df_nude['genre'] == 'nude painting (nu)'
    df_nude = df_nude[is_nude]

    # filter by 'style' == 'Baroque'
    # is_baroque = df_nude['style'] == 'Baroque'
    # df_nude = df_nude[is_baroque]

    extract_painters(df_nude)

    return df_nude


def analyze_statistics(infile, col):

    df = pd.read_csv(infile)

    # filter columns
    df = df[['artist', 'date', 'genre', 'style', 'new_filename']]

    # statistics for desired column
    df = df.groupby([col]).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)


def create_dataset(df, indir, outdir):

    # make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for index, row in df[['artist_group', 'new_filename']].iterrows():

        group = row['artist_group']
        fn = row['new_filename']

        print(group, fn)

        if group.find('train') != -1:
            # train images are from the 'train' folder; but sometimes it is put in the 'test' folder, given 'artist_group' == 'train_and_test'.
            src = os.path.join(indir, 'train', fn)
            dst = os.path.join(outdir, fn)

            try:
                copyfile(src, dst)
            except:
                src = os.path.join(indir, 'test', fn)
                copyfile(src, dst)
        else:
            # test images are from the 'test' folder; images are not duplicate in 'train' and 'test' folders.
            src = os.path.join(indir, 'test', fn)
            dst = os.path.join(outdir, fn)

            try:
                copyfile(src, dst)
            except:
                src = os.path.join(indir, 'train', fn)
                copyfile(src, dst)


if __name__ == '__main__':

    # filter the painters during the Baroque period
    # df = filter_baroque(infile=infile)
    # create_dataset(df, indir=indir, outdir=outdir_baroque)

    # analyze statistics of dataset
    # analyze_statistics(infile, 'style')
    # analyze_statistics(infile, 'genre')

    # filter the painters for nude paintings
    df = filter_nude(infile=infile)
    # create_dataset(df, indir=indir, outdir=outdir_nude)