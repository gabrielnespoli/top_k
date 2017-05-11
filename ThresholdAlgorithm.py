import sys
import pandas as pd


class ThresholdAlgorithm:
    def __init__(self, total_files, k):
        self.total_files = total_files
        self.k = k
        self.worst_score = 0

    def get_topk(self, datasets, weights):
        self.result = pd.DataFrame(columns=['Query_ID', 'Doc_ID', 'Rank', 'Score'])
        i = 0  # i marks the position of the pointer in the dataframe (i is the row being read)
        while True:
            # according to Threshold Algorithm, delta is the stop-condition, it is the sum of all elements pointed
            # by the pointer i, which means, the i-th element in each dataset
            delta = 0

            # read at the same time the dataset and the weight of each occurrence of
            # the list in the total score
            # j marks the dataframe that is being read (and the weight of it)
            for j in range(self.total_files):
                df_seq_acc, w_seq_acc = datasets[j], weights[j]

                # to perform the random access to elements of the other dataframes
                list_df_rand_acc = list(datasets)
                del list_df_rand_acc[j]
                list_weight_rand_acc = list(weights)
                del list_weight_rand_acc[j]

                # read the same line 'i' of each dataset at the "same" time
                new_item = df_seq_acc.iloc[i]
                doc_id = new_item['Doc_ID']
                delta += new_item['Score']

                # perform the random access of the doc_id in the "others" dataframe and sum up with the score of the
                # item of the sequential access (considering the weight of each score - a doc_id present in title
                # could have more weight than in text, for example
                for df_rand_acc, weight_rand_acc in zip(list_df_rand_acc, list_weight_rand_acc):
                    value_rand_acc = 0
                    if doc_id in df_rand_acc['Doc_ID'].values:
                        value_rand_acc = df_rand_acc.loc[df_rand_acc['Doc_ID'] == doc_id]['Score'].values[0]
                    new_item['Score'] = w_seq_acc*new_item['Score'] + weight_rand_acc*value_rand_acc

                    self.update_result(new_item)

            # check  the stop condition, ie. when the scores of the top-k are greater or equal to the threshold
            if self.result.shape[0] == self.k and self.worst_score >= delta:
                self.set_rank()
                return self.result

            # move to the next element in each dataset
            i += 1

    def update_result(self, new_item):
        doc_id = new_item['Doc_ID']

        # if the doc_id is still in the result but with a smaller score, updates it
        if doc_id in self.result['Doc_ID'].values:
            return

        self.result = self.result.append(new_item)
        self.result.sort_values(by='Score', ascending=False, inplace=True)
        self.result.reset_index(drop=True, inplace=True)

        if self.result.shape[0] > self.k:
            self.result.drop(self.result.index[self.result.shape[0] - 1], inplace=True)  # eliminates the old worst score
        self.worst_score = self.result.iloc[self.result.shape[0] - 1]['Score']  # updates the worst score in the top-k result

    # After getting the top k elements, the Rank field has the old values and has to be correct
    def set_rank(self):
        for i in range(0, self.result.shape[0]):
            self.result.iloc[i]['Rank'] = i + 1


# Save in a tsv file in the format:
# [Query_ID \t Doc_ID \t Rank \t Score]
def output(top_k, output_file):
    with open(output_file, 'a+') as f:
            top_k.to_csv(f, index=False, header=False, sep='\t')


def main():
    # reading all the parameters from the terminal
    # parameters' shape:
    # [k] [number of files/dataset] [weight of the i-th dataframe score separete by space] [output directory]
    # Ex:
    # 5 2 ./data/output-stopwords-BM25Scorer-title.tsv ./data/output-stopwords-BM25Scorer-text.tsv 2 1 ./data/output-threshold.tsv
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

    fa = ThresholdAlgorithm(total_files, k)
    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, sep='\t', usecols=['Query_ID', 'Doc_ID', 'Rank', 'Score']))

    # clean the output file
    columns = ['Query_ID', 'Doc_ID', 'Rank', 'Score']
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(output_file, index=False, sep='\t')

    # execute the Threshold's Algorithm for each query of each document
    query_ids = dfs[0]['Query_ID'].unique()
    for q in query_ids:
        df_query = []
        for df in dfs:
            df_query.append(df.loc[df['Query_ID'] == q])
        top_k = fa.get_topk(datasets=df_query, weights=weights_vector)
        output(top_k, output_file)


if __name__ == '__main__':
    main()
