import sys
import pandas as pd
from collections import OrderedDict
import os.path


class FaginsAlgorithm:
    def __init__(self, total_files, k):
        self.total_files = total_files
        self.k = k

    def get_topk(self, listsnames, datasets, weights):
        self.R = {}
        self.n_total_hit = 0
        self.result = []
        i = 0
        while True:
            # read at the same time the list name, the dataset and the weight of each occurrence of
            # the list in the total score
            for listname, ds, w in zip(listsnames, datasets, weights):
                # read the same line 'i' of each dataset at the "same" time
                item = ds.iloc[i]
                doc_id = str(item['Doc_ID'])

                # check if the doc_id was already included in the visited-set R. Add the file if not, otherwise
                # updates the lists which the doc_id was seen, the total_score (considering the different weights
                # between title and text, for examle) and the number of times the doc_id was seen in overall
                # the datasets
                if doc_id not in self.R:
                    self.R[doc_id] = (item['Query_ID'], item['Doc_ID'], item['Rank'], w * item['Score'], listname, 1)
                else:
                    if listname not in self.R[doc_id][4]:
                        self.R[doc_id] = self.R[doc_id][:3] + \
                                         tuple(map(my_sum, zip(self.R[doc_id][3:], (w * item['Score'], listname, 1))))

                    # this is a total hit (a doc_id was seen in all files)
                    if self.R[doc_id][5] == self.total_files:
                        self.n_total_hit += 1

                        # stop the execution when K total hit is reached
                        if self.n_total_hit == self.k:
                            self.select_top_k()
                            self.set_rank()
                            return self.result
            i += 1

    def select_top_k(self):
        l = []

        # creates a list in the shape [(doc_id, (query_id, doc_id, etc...))]
        for key, value in self.R.items():
            l.append((key, value))
        # order the list by the 4th element, which is the score
        l = sorted(l, key=lambda x: x[1][3], reverse=True)

        # add the top k elements to an ordered dictionary, considering just the query_id, doc_id, rank and score.
        # in the end the od will be like this, for example:
        # od = { ('486.0', (1.0, 486.0, 1.0, 20.119134237971807);
        #        ('13.0', (1.0, 13.0, 1.0, 18.029652117379523));
        #        ('184.0', (1.0, 184.0, 2.0, 17.861152656619193));
        #        ('51.0', (1.0, 51.0, 3.0, 16.102725220535181));
        #        ('359.0', (1.0, 359.0, 7.0, 14.590144564103138)).
        od = OrderedDict()
        for i in range(self.k):
            # get the first 4 elements of the tuple (which are the values present in the input dataframe)
            od[l[i][0]] = l[i][1][:4]
        self.result = od

    # After getting the top k elements, the Rank field has the old values and has to be correct
    def set_rank(self):
        i = 1
        for item in self.result.keys():
            l = list(self.result[item])
            l[2] = i
            self.result[item] = tuple(l)
            i += 1


# a new 'sum' function was defined because the built-in sum doesn't implicitly sum string and number in the
# same tuple
def my_sum(t):
    return t[0] + t[1]


# Save in a tsv file in the format:
# [Query_ID \t Doc_ID \t Rank \t Score]
def output(top_k_dict, output_file):
    new_df = pd.DataFrame.from_dict(top_k_dict, orient='index')
    columns=['Query_ID', 'Doc_ID', 'Rank', 'Score']
    new_df.columns = columns
    new_df['Query_ID'] = new_df['Query_ID'].astype(int)
    new_df['Doc_ID'] = new_df['Doc_ID'].astype(int)
    new_df['Rank'] = new_df['Rank'].astype(int)

    if os.path.isfile(output_file):
        with open(output_file, 'a+') as f:
            new_df.to_csv(f, index=False, header=False, sep='\t')
    else:
        new_df.to_csv(output_file, index=False, sep='\t')


def main():
    # reading all the parameters from the terminal
    # parameters' shape:
    # [k] [number of files/dataset] [weight of the i-th dataframe score separete by space] [output directory]
    # Ex:
    # 5 2 ./data/output-stopwords-BM25Scorer-title.tsv ./data/output-stopwords-BM25Scorer-text.tsv 2 1 ./data/output-fagins.tsv
    k = int(sys.argv[1])
    total_files = int(sys.argv[2])
    filenames = []
    Ls = []
    for i in range(total_files):
        filenames.append(sys.argv[3+i])
        Ls.append('L'+str(i))
    weights_vector = []
    for i in range(total_files):
        weights_vector.append(int(sys.argv[3+total_files+i]))
    output_file = sys.argv[3+total_files+i+1]

    if total_files != len(filenames) != len(weights_vector):
        raise Exception("Number of files or elements in the vector of weights don't match")

    fa = FaginsAlgorithm(total_files, k)
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, sep='\t', usecols=['Query_ID', 'Doc_ID', 'Rank', 'Score']))

    # clean the output file
    columns = ['Query_ID', 'Doc_ID', 'Rank', 'Score']
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(output_file, index=False, sep='\t')

    # execute the Fagin's Algorithm for each query of each document
    query_ids = dfs[0]['Query_ID'].unique()
    for q in query_ids:
        df_query = []
        for df in dfs:
            df_query.append(df.loc[df['Query_ID'] == q])
        top_k = fa.get_topk(listsnames=Ls, datasets=df_query, weights=weights_vector)
        output(top_k, output_file)


if __name__ == '__main__':
    main()
