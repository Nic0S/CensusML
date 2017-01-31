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

        for row in data:
            split_row = row.strip().split(delimeter)
            self.original_data.append(split_row)

        self.original_data = np.array(self.original_data)

        self.num_columns = self.original_data.shape[1]

        for i in range(0, self.num_columns):
            if i in self.skip_columns:
                continue

            if i in self.categorize_columns:
                column = self.original_data[:, i]
                self.column_categories.append([])

                for unique_item in set(column):
                    self.column_categories[-1].append(unique_item)

            else:
                self.column_categories.append(None)

        print(self.column_categories)




        # for row in data:
        #     split_row = row.strip().split(delimeter)
        #
        #     for i, cell in enumerate(split_row):
        #
        #         # This row needs to be categorized
        #         if i in categorize_columns:
        #
        #             # This category has already been made
        #             if cell in self.column_categories[i]:
        #
        #                 # Get the category value from column_categories and add it to the row
        #                 modified_row.append(self.column_categories[i].index(cell))
        #
        #             # Category has not been seen yet, create it
        #             else:
        #                 self.column_categories[i].append(cell)
        #                 modified_row.append(len(self.column_categories[i]) - 1)
        #
        #         else:
        #             modified_row.append(cell)
        #
        #     print(modified_row)









