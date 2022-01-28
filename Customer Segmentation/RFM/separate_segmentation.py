import pandas as pd


def segmentation(dataset: pd.DataFrame, rfm: list, d: int):
    """
    Separate RFM Segmentation function
    :param dataset: given dataset
    :param rfm: a list of three column name R, F, M
    :param d: number of delimiters to divide data based on each factor
    :return: dataset with new segment column
    """
    def segment_calculator(customer_data, delimiters, ds, factors):
        """
        Calculator of each customer segment based on factors
        :param customer_data: a row of data related to a customer
        :param delimiters: a dictionary of the length between each group based on each factor
        :param ds: given dataset
        :param factors: a list of three column name R, F, M
        :return: segment_id of the customer
        """
        segment_id = 0
        i = 9
        for factor in factors:
            j = 1
            min_value = ds[factor].min()
            while j != 3 and min_value + j * delimiters[factor] <= customer_data[factor]:
                j += 1
            segment_id += (j - 1) * i
            i //= 3
        segment_id += 1
        return segment_id

    # Calculating delimiters to make
    factor_group_delimiters = {}
    for factor in rfm:
        factor_range = dataset[factor].max() - dataset[factor].min()
        factor_group_delimiters[factor] = factor_range / d

    # Adding segmentation results to dataset
    dataset['Separate_Approch_Segment'] = dataset.apply(
        lambda row: segment_calculator(row, factor_group_delimiters, dataset, rfm), axis=1)

    return dataset
