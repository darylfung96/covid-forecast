class RepHoldout:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, data):
        current_test_percentage = 10

        all_index = []

        for i in range(self.n_splits):
            total_length = data.shape[0]
            test_size = round(total_length * current_test_percentage / 100)
            train_size = total_length - test_size
            train_index = range(train_size)
            test_index = range(train_size, total_length)

            all_index.append([train_index, test_index])
            current_test_percentage += 10

        return all_index
