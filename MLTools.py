import numpy as np

class InputData:
    num_columns = -1
    num_rows = 0
    original_data = []
    modified_data = []

    skip_columns = []
    categorize_columns = []

    column_categories = []

    def __init__(self, data, delimeter=",", skip_columns=[], categorize_columns=[]):

        self.skip_columns = skip_columns
        self.categorize_columns = categorize_columns

        self.num_columns = len(data[0].split(delimeter))


        for row in data:
            split_row = row.strip().split(delimeter)
            split_row = [item.strip() for item in split_row]

            if len(split_row) == self.num_columns:
                self.original_data.append(split_row)
                self.num_rows += 1

        self.original_data = np.array(self.original_data)

        # Set up the category data, used to make one hot encodings
        for i in range(0, self.num_columns):

            if i in self.categorize_columns:
                column = self.original_data[:, i]
                self.column_categories.append([])

                for unique_item in set(column):
                    self.column_categories[-1].append(unique_item)

            else:
                self.column_categories.append(None)

        for row in self.original_data:
            modified_row = []

            for i, cell in enumerate(row):
                if i in self.skip_columns:
                    continue

                if i in self.categorize_columns:
                    one_hot = np.zeros(len(self.column_categories[i]), np.int8)
                    one_hot[self.column_categories[i].index(cell)] = 1
                    modified_row.extend(one_hot)

                else:
                    modified_row.append(int(cell))

            self.modified_data.append(modified_row)






