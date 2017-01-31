

class InputData:
    numColumns = -1
    numRows = 0
    origData = None
    modifiedData = []

    skip_columns = []
    categorize_columns = []

    column_categories = []

    def __init__(self, data, delimeter=",", skip_columns=[], categorize_columns=[]):

        # Copy this to avoid the user changing data?
        self.origData = data

        self.skip_columns = skip_columns
        self.categorize_columns = categorize_columns

        for row in data:
            split = row.strip().split(delimeter)

            # First row, determine column info
            if self.numColumns == -1:
                self.init_columns(len(split))

    def init_columns(self, num_columns):
        self.numColumns = num_columns

        for i in range(0, num_columns):
            if i in self.categorize_columns:
                self.column_categories.append([])
            else:
                self.column_categories.append(None)

        print(self.column_categories)



