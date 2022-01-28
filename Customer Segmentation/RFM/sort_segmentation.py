import operator
import pandas as pd


def segmentation(dataset: pd.DataFrame, rfm: list, d: int):
    """
    Sort RFM Segmentation function
    :param dataset: given dataset
    :param rfm: a list of three column name R, F, M
    :param d: number of delimiters to divide data based on each factor
    :return: dataset with new segment column
    """
    datalists = [dataset.values.tolist()]
    for factor in rfm:
        new_datalists = []
        for datalist in datalists:
            datalist.sort(key=operator.itemgetter(dataset.columns.get_loc(factor)))
            size = len(datalist)
            low_index = 0
            rem = size % d
            step = size / d
            for i in range(d):
                up_index = low_index + int(step) + (1 if rem > 0 else 0)
                new_datalists.append(datalist[low_index: up_index])
                rem -= 1
                low_index = up_index
        datalists = new_datalists

    # Determining customer segments
    customer_segment = dict()
    id_index = dataset.columns.get_loc('id')
    for segment_id in range(d**3):
        for customer in datalists[segment_id]:
            customer_segment[customer[id_index]] = segment_id + 1

    # Adding segmentation results to dataset
    dataset['Sort_Approch_Segment'] = dataset.apply(lambda row: customer_segment[row['id']], axis=1)

    return dataset
