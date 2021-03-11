import pandas as pd
import os
from shutil import copyfile, move
import numpy as np
import matplotlib.pyplot as plt


# dataset indir and outdir
indir = os.path.join('dataset', 'painter-by-numbers')
outdir_baroque = os.path.join('dataset', 'painter-of-baroque')
outdir_nude = os.path.join('dataset', 'painter-of-nude')

# csv file
infile = os.path.join(indir, 'all_data_info.csv')


def convert_date_to_numeric(df):

    # extract 4 digits anywhere: e.g., c.1982, November 1982
    df['date'] = df['date'].str.extract(r'(\d{4})')

    # convert to numeric
    df['date'] = pd.to_numeric(df['date'])

    return df


def group_by_col(df, is_grouped_by_artist=False, is_grouped_by_style=False, is_grouped_by_genre=False, show_all=False):

    df = df[['artist', 'style', 'genre', 'new_filename']]

    if is_grouped_by_artist and is_grouped_by_style and is_grouped_by_genre:
        df = df.groupby(['artist', 'style', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'style', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist and is_grouped_by_style:
        df = df.groupby(['artist', 'style']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'style', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist and is_grouped_by_genre:
        df = df.groupby(['artist', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['artist', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_style and is_grouped_by_genre:
        df = df.groupby(['style', 'genre']).agg({'new_filename': lambda x: x.unique().tolist()}).reset_index()[['style', 'genre', 'new_filename']]
        df['counts'] = [len(row) for row in df['new_filename'].tolist()]
    elif is_grouped_by_artist:
        # aggregate for 'artist' by the total count of the paintings
        df = df.groupby(['artist']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    elif is_grouped_by_style:
        # aggregate for 'style' by the total count of the paintings
        df = df.groupby(['style']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    elif is_grouped_by_genre:
        # aggregate for 'genre' by the total count of the paintings
        df = df.groupby(['genre']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    else:
        pass

    # print all the rows
    pd.set_option('display.max_rows', df.shape[0] + 1)
    # print all columns
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    if show_all:
        pd.set_option('display.max_colwidth', None)

    print(df)


def filter_baroque(infile, is_grouped_by_artist=False, is_grouped_by_style=False, is_grouped_by_genre=False, show_all=False):

    df = pd.read_csv(infile)

    # filter columns
    df_baroque = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    # filter by 'date'
    # df_baroque = convert_date_to_numeric(df_baroque)

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

    # filter by 'genre' == 'nude painting (nu)'
    # is_nude = df_baroque['genre'] == 'nude painting (nu)'
    # df_baroque = df_baroque[is_nude]

    group_by_col(df_baroque, is_grouped_by_artist, is_grouped_by_style, is_grouped_by_genre, show_all)

    return df_baroque


def filter_nude(infile, is_grouped_by_artist=False, is_grouped_by_style=False, is_grouped_by_genre=False, show_all=False):

    df = pd.read_csv(infile)

    # filter columns
    df_nude = df[['artist', 'date', 'genre', 'style', 'artist_group', 'new_filename']]

    # filter by 'genre'
    is_nude = df_nude['genre'] == 'nude painting (nu)'
    df_nude = df_nude[is_nude]

    # filter by 'style' == 'Baroque'
    # is_baroque = df_nude['style'] == 'Baroque'
    # df_nude = df_nude[is_baroque]

    group_by_col(df_nude, is_grouped_by_artist, is_grouped_by_style, is_grouped_by_genre, show_all)

    return df_nude


def analyze_statistics(infile, col, is_grouped_by_date=False, is_ascending=False, is_displayed=True):

    df = pd.read_csv(infile)

    # filter columns
    df = df[['artist', 'date', 'genre', 'style', 'new_filename']]

    # convert date to numeric
    df = convert_date_to_numeric(df)

    # statistics for desired column
    if is_grouped_by_date:
        df = df.groupby([col, 'date']).size().reset_index(name='counts').sort_values(by='date', ascending=is_ascending)
    else:
        df = df.groupby([col]).size().reset_index(name='counts').sort_values(by='counts', ascending=is_ascending)

    if is_displayed:
        pd.set_option('display.max_rows', df.shape[0] + 1)
        print(df)

    return df


def filter_top_style_ordered_by_date(infile):

    df = pd.read_csv(infile)

    # filter columns
    df = df[['artist', 'date', 'genre', 'style', 'new_filename']]

    # convert date to numeric
    df = convert_date_to_numeric(df)

    # filter 'df' -> 'df_style_count': style + count
    df_style_count = df.groupby(['style']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    # filter 'df_style_count' -> 'df_top_style_count': counts > 1000
    df_top_style_count = df_style_count[df_style_count['counts'] > 1000]

    # filter 'df': style in 'df_top_style_count'
    is_style = df['style'].apply(lambda x: True if x in df_top_style_count['style'].unique().tolist() else False)
    df = df[is_style]

    # aggregate 'df' by date: mean of date -> date
    df = df.groupby(['style']).agg({'date': 'mean'}).reset_index()[['style', 'date']].sort_values(by='date', ascending=True)

    # add 'counts' in 'df': from 'df_top_style_count'
    df['counts'] = [df_top_style_count[df_top_style_count['style']==style]['counts'].values[0] for style in df['style'].tolist()]

    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)

    return df


def plot_stacked_barchart(df):

    style_names = df['style'].tolist()
    results = {'Style': df['counts'].tolist()}

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    style_colors = plt.get_cmap('rainbow')(np.linspace(0.15, 0.85, data.shape[1])) # cmap = RdYlGn or rainbow

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(style_names, style_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color, rotation=90)

    ax.legend(ncol=3, loc='upper right', fontsize='small')

    # save the plot
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join('pix', 'top-21-styles.png'), bbox_inches='tight', pad_inches=0)

    # show the plot
    # plt.show()


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


def create_painter_dir(df, indir):

    for index, row in df.iterrows():

        artist = row['artist']
        fn = row['new_filename']

        painter_dir = os.path.join(indir, artist)

        if not os.path.exists(painter_dir):
            os.makedirs(painter_dir)

        src = os.path.join(indir, fn)
        dst = os.path.join(painter_dir, fn)
        move(src, dst)


if __name__ == '__main__':

    # analyze statistics of dataset
    # df = analyze_statistics(infile, col='style', is_grouped_by_date=False, is_ascending=True)
    # analyze_statistics(infile, col='genre')

    # filter top styles (counts of style > 1000) ordered by date
    df = filter_top_style_ordered_by_date(infile)
    # plot stacked barchart for style
    plot_stacked_barchart(df)

    # filter the painters during the Baroque period
    # df = filter_baroque(infile=infile, is_grouped_by_artist=True, is_grouped_by_genre=False, show_all=False)
    # create_dataset(df, indir=indir, outdir=outdir_baroque)
    # create_painter_dir(df, indir=outdir_baroque)

    # filter the painters for nude paintings
    # df = filter_nude(infile=infile, is_grouped_by_artist=True, is_grouped_by_style=True)
    # create_dataset(df, indir=indir, outdir=outdir_nude)
    # create_painter_dir(df, indir=outdir_nude)